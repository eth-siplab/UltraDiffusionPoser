import glob
import os
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch.optim as optim
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer

from torch.utils.data import DataLoader

from modules.dataset.dataset import *
from modules.evaluate.eval_utils import PoseEvaluator
from modules.evaluate.evaluator import evaluate_model, print_eval_result
from modules.log.logging_utils import log_video, log_metrics
from modules.loss.loss_utils import *
from modules.model import get_model
from modules.util.mds_util import compute_mds, plot_points, plot_mds_points_on_body, normalize_mds, reflect_mds, \
    resolve_reflections
from modules.utils import *

import wandb

from aitviewer.configuration import CONFIG as C



class Trainer:
    def __init__(self, opt, parser, debug=False) -> None:
        self.batch_size = opt.batch_size
        self.device = opt.device
        self.w_eval = opt.eval
        self.dry_run = opt.dry_run
        self.eval_dataset_name = opt.eval_dataset
        self.debug = debug
        timestamp = opt.timestamp
        model_cls = get_model(opt, parser)
        args = parser.parse_known_args()[0]
        args.timestamp = timestamp
        opt.timestamp = timestamp
        self.model = model_cls(args=args).to(device=self.device)
        self.model_name = opt.network
        self.use_uwb = "vuwb" in self.model.imu_m
        self.save_interval = opt.save_interval
        self.flatten_uwb = opt.flatten_uwb
        self.use_virtual_uwb = opt.use_virtual_uwb
        self.exclude_tc_amass = opt.exclude_tc_amass
        self.include_amass_in_ft = opt.include_amass_in_ft
        if self.dry_run:
            print("###########You are in dry-run, which is only for quick testing!!!###############")

        # Training_phase (supplied via the config; default to the diffusion phase)
        self.training_phase = opt.training_phase if opt.training_phase is not None else ["baseline_diffusion_all"]

        # Load pretrain Model
        if opt.pretrain_model:
            weight_loaded = torch.load(opt.pretrain_model)
            strict = True
            if "net" in weight_loaded:
                self.model.load_state_dict(weight_loaded["net"], strict=strict)
            else:
                self.model.load_state_dict(weight_loaded, strict=strict)

            print(f"Load model weight from {opt.pretrain_model}")

        # Initialize optimizer
        self.epochs = opt.epochs if not self.dry_run else 1
        self.lr = opt.lr
        self.lr_scalar = {}  # per-phase LR multiplier; UDP phases default to 1.0 (see _init_optimizer)
        self.grad_clip = opt.grad_clip
        self.scheduler_step = opt.scheduler_step
        self.weight_decay = opt.weight_decay
        self.early_stop_delt = opt.early_stop_delt

        # Dataset
        self.dataset = None
        self.downsample_rate = opt.downsample_rate
        self.batch_size = opt.batch_size
        self.resampling_interval = opt.resample_interval
        self.normalize_uwb = opt.normalize_uwb
        self.remove_node = opt.remove_node
        self.dataset_common_kwargs = {"official_model_file": paths.smpl_file,
                                      "seq_length": opt.data_seq_len,
                                      "device": "cpu",
                                      "add_uwb": self.use_uwb,
                                      "imu_m": self.model.imu_m,
                                      "static_uwb_noise": opt.static_uwb_noise,
                                      "imu_acc_noise": opt.imu_acc_noise,
                                      "imu_ori_noise": opt.imu_ori_noise,
                                      "imu_ori_bias_noise": opt.imu_ori_bias,
                                      "extreme_value_thresh_g": opt.extreme_value_thresh_g,
                                      "normalize_uwb": self.normalize_uwb,
                                      "convert_acc_to_g": opt.convert_acc_to_g,
                                      "predict_height": opt.predict_height,
                                      "use_cached": opt.use_dataset_cached,
                                      "uwb_timesample_ratio": opt.uwb_timesample_ratio,
                                      "flatten_uwb": opt.flatten_uwb,
                                      "dry_run": opt.dry_run,
                                      "remove_node": self.remove_node}
        print(f"Dataset Config: {self.dataset_common_kwargs}")

        # loss func
        Loss_Func.add_args(parser)
        args = parser.parse_known_args()[0]
        self.loss_func = Loss_Func(args)

        # initialize writer
        if self.dry_run:
            opt.log_dir = opt.log_dir + "_dry_run"
        self.ckpt_dir = os.path.join(opt.log_dir, "ckpt")
        self.eval_dir = os.path.join(opt.log_dir, "eval")
        self.lowest_val_loss = float("inf")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.eval_dir, exist_ok=True)
        self.step = 0
        args.timestamp = timestamp
        wandb.init(project=os.getenv("WANDB_PROJECT", "UDP"), name=Path(opt.log_dir).parent.name, config=vars(args))

        self.writer_log = Easy_dict()

        self.model.save_config(args, os.path.join(opt.log_dir, "model_args.json"))

        C.update_conf({"smplx_models": os.getenv("SMPL_MODEL_PATH", "data/smpl_m_lbs_10_207_0_v1.0.0.pkl"),
                       "run_animations": True,
                       "scene_fps": 60,
                       "playback_fps": 60,
                       "device": self.device
                       })

        self.smpl_layer = SMPLLayer(model_type="smpl", gender="male")
        self.renderer = HeadlessRenderer()

    def _init_dataloader(self, phase):
        train_phase, module_name = phase.split("_", 1)
        if train_phase == "finetune":
            if self.eval_dataset_name in ["uwb-imu", 'uwb-mixed']:
                # Finetune on UWB-IMU data
                dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,
                                                                                              "sigma0")
                dataset_real = UWBIMU_real_data_train(dataset_path, down_sample_rate=20, **self.dataset_common_kwargs)
                self.dataset = dataset_real

                self.data_loader = DataLoader(self.dataset, shuffle=True,
                                              pin_memory=True,
                                              batch_size=self.batch_size,
                                              num_workers=3)

                dataset_val = UWBIMU_real_data_val(dataset_path, down_sample_rate=100, train_split=False,
                                                   **self.dataset_common_kwargs)
                self.val_dataloader = DataLoader(dataset_val, shuffle=True, pin_memory=True, batch_size=self.batch_size,
                                                 num_workers=3)
            else:
                # Finetune on DIP-IMU data
                self.dataset = DIPIMU_real_data_train(paths.dipimu_dir, down_sample_rate=self.downsample_rate,
                                                      **self.dataset_common_kwargs)
                if self.include_amass_in_ft:
                    dataset_path = paths.amass_dir if not self.exclude_tc_amass else os.path.join(paths.amass_dir, "no_tc")
                    dataset_amass = AMASS_syn_data(dataset_path, down_sample_rate=600,
                                                  **self.dataset_common_kwargs)
                    self.dataset = torch.utils.data.ConcatDataset([self.dataset, dataset_amass])

                self.data_loader = DataLoader(self.dataset, shuffle=True,
                                              pin_memory=True,
                                              batch_size=self.batch_size,
                                              num_workers=3)

                dataset_val = DIPIMU_real_data_val(paths.dipimu_dir, down_sample_rate=100, train_split=False,
                                                   **self.dataset_common_kwargs)
                self.val_dataloader = DataLoader(dataset_val, shuffle=True, pin_memory=True, batch_size=self.batch_size,
                                                 num_workers=3)

        elif train_phase == "baseline":
            if isinstance(self.dataset, AMASS_syn_data):
                return
            dataset_path = paths.amass_dir if not self.exclude_tc_amass else os.path.join(paths.amass_dir, "no_tc")
            self.dataset = AMASS_syn_data(dataset_path, down_sample_rate=self.downsample_rate,
                                          **self.dataset_common_kwargs)
            self.data_loader = DataLoader(self.dataset, shuffle=True,
                                          pin_memory=True,
                                          batch_size=self.batch_size,
                                          num_workers=3)
            if self.eval_dataset_name in ['uwb-mixed', 'uwb_imu']:
                dataset_val = UWBIMU_real_data_val(paths.uwbimu_dir, down_sample_rate=100, train_split=False,
                                                   **self.dataset_common_kwargs)
                self.val_dataloader = DataLoader(dataset_val, shuffle=True, pin_memory=True, batch_size=self.batch_size,
                                                 num_workers=4)
            else:
                dataset_val = AMASS_syn_data_val(paths.amass_dir, down_sample_rate=100, train_split=False,
                                                 **self.dataset_common_kwargs)
                self.val_dataloader = DataLoader(dataset_val, shuffle=True, pin_memory=True, batch_size=self.batch_size,
                                                 num_workers=4)
        else:
            raise KeyError(f"Invalid training phase {train_phase}")
        return

    def _init_optimizer(self, phase):
        train_phase, module_name = phase.split("_", 1)
        if train_phase in ["finetune", "baseline"]:
            for name, param in self.model.named_parameters():
                if name.startswith(module_name):
                    param.requires_grad = True
                elif "diffusion" in module_name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            raise NotImplementedError(f"Invalid training phase {phase}")

        non_frozen_parameters = [p for p in self.model.parameters() if p.requires_grad]
        num_param = sum(p.numel() for p in non_frozen_parameters)
        frozen_parameter_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"Initialize Training phase {phase} -- Number of Parameter: {num_param}, Frozen Parameter: {frozen_parameter_count}")

        # print number of parameters per layer
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"Layer: {name} | Number of Parameters: {param.numel()}")

        self.optimizer = optim.Adam(non_frozen_parameters, lr=self.lr * self.lr_scalar.setdefault(phase, 1.0),
                                    weight_decay=self.weight_decay)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=0.33)

    def wandb_logging(self, epoch, phase, **log_data):
        assert "train_loss" in log_data
        assert "val_loss" in log_data

        # Log validation losses
        for key in log_data["val_loss"].keys():
            wandb.log({f"{phase}/val/{key}": log_data["val_loss"][key], "epoch": epoch}, step=self.step, commit=False)

        # Log training losses
        for key in log_data["train_loss"].keys():
            wandb.log({f"{phase}/train/{key}": log_data["train_loss"][key], "epoch": epoch}, step=self.step, commit=False)

        # Log other metrics
        for key in log_data.keys():
            if key in ["train_loss", "val_loss"]:
                continue
            wandb.log({f"{phase}/{key}": log_data[key], "epoch": epoch}, step=self.step, commit=False)

        wandb.log({}, step=self.step)

    def train(self):
        self.model.train()
        for phase in self.training_phase:
            self.early_stop_check = EarlyStop(delta=self.early_stop_delt)
            self._init_dataloader(phase)
            self._init_optimizer(phase)
            self.loss_func.set_training_phase(phase)

            for epoch in range(self.epochs):
                self.train_one_epoch(epoch, phase=phase)
                self.lr_scheduler.step()
                wandb.log({"lr": self.lr_scheduler.get_last_lr()[0], "epoch": epoch}, step=self.step)
                if self.early_stop_check.early_stop:
                    if phase == self.training_phase[-1]:
                        print("evaluating model")
                        self.evaluate_model(epoch, phase)
                    print(f"Early stop {phase} @ Epoch {epoch}")
                    break
                if (epoch != 0 and epoch % self.save_interval == 0) or epoch == self.epochs - 1:
                    self.eval(epoch, phase)

    def eval(self, epoch, phase):
        # Evaluating the current model on full dataset
        self.model.eval()
        if epoch == self.epochs - 1:
            file_name = os.path.join(self.ckpt_dir, f"{phase}_last_model_{str(epoch).zfill(3)}.pt")
        else:
            file_name = os.path.join(self.ckpt_dir, f"{phase}_ckpt_{str(epoch).zfill(3)}.pt")
        self.save_checkpoint(file_name=file_name, epoch=epoch)

        # delete all previous _ckpt_ files that are not the current one
        ckpts = glob.glob(os.path.join(self.ckpt_dir, f"{phase}_ckpt_*.pt"))
        for ckpt in ckpts:
            if ckpt != file_name:
                os.remove(ckpt)

        # To save time only evaluate in the last epoch of the final phase
        if phase == self.training_phase[-1] and epoch == self.epochs - 1:
            self.evaluate_model(epoch, phase)

        print(f"Model saved at {file_name}")

        self.model.train()

    def evaluate_model(self, epoch, phase):
        if self.w_eval:
            device = self.device

            model_path = os.path.join(self.ckpt_dir, f"{phase}_best_model*.pt")
            model_path = list(glob.glob(model_path))
            if len(model_path) == 0:
                print(f"No model found for {phase}")
                return
            elif len(model_path) > 1:
                print(f"Multiple model found for {phase}")
                return
            else:
                model_path = model_path[0]
            best_model = torch.load(model_path, map_location=device)
            self.model.load_state_dict(best_model["net"])
            epoch_save = best_model["epoch"]

            print(f"Eval Best Model {self.model.name} @ Epoch {epoch_save} ...")
            if self.eval_dataset_name == "dip-imu":
                seq_ids = [0] if self.dry_run else None
                eval_dip = evaluate_model(self.model, paths.dipimu_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True,
                                          evaluate_zmp=False, \
                                          flush_cache=True, sequence_ids=seq_ids, normalize_uwb=self.normalize_uwb,
                                          flatten_uwb=self.flatten_uwb, remove_node=self.remove_node,
                                            device=device)
                eval_tc = evaluate_model(self.model, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(),
                                         evaluate_pose=True, evaluate_zmp=True, \
                                         flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                         normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                         remove_node=self.remove_node,
                                            device=device)
                table = print_eval_result([eval_dip, eval_tc], filter_keys=set(eval_dip) & set(eval_tc))
                eval_trans = eval_tc
            elif self.eval_dataset_name == "tc-imu":
                seq_ids = [0] if self.dry_run else None
                eval_tc = evaluate_model(self.model, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(),
                                         evaluate_pose=True, evaluate_zmp=True, \
                                         flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                         normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                         remove_node=self.remove_node,
                                            device=device)
                table = print_eval_result([eval_tc], filter_keys=set(eval_tc), title=["    ", "TotalCapture"])
                eval_trans = eval_tc
            elif self.eval_dataset_name == "amass":
                seq_ids = [0] if self.dry_run else list(range(30))
                eval_amass = evaluate_model(self.model, os.path.join(paths.amass_dir, "test_split"),
                                            pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
                                            flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                            normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                            remove_node=self.remove_node,
                                            device=device)
                table = print_eval_result([eval_amass], filter_keys=set(eval_amass), title=["    ", "AMASS Dance-DB"])
                eval_trans = eval_amass
            elif self.eval_dataset_name == "uwb-imu":
                seq_ids = [0] if self.dry_run else None
                dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,
                                                                                              "sigma0")
                eval_uwb_imu = evaluate_model(self.model, dataset_path, pose_evaluator=PoseEvaluator(), evaluate_pose=True,
                                              evaluate_zmp=True, \
                                              flush_cache=True, evaluate_tran=True, plt_tran=False,
                                              sequence_ids=seq_ids, normalize_uwb=self.normalize_uwb,
                                              flatten_uwb=self.flatten_uwb, remove_node=self.remove_node,
                                            device=device)
                table = print_eval_result([eval_uwb_imu], filter_keys=set(eval_uwb_imu), title=["    ", "UWB-IMU Test"])
                eval_trans = eval_uwb_imu
            elif self.eval_dataset_name == "uwb-mixed":
                seq_ids = [0] if self.dry_run else list(range(30))
                eval_amass = evaluate_model(self.model, os.path.join(paths.amass_dir, "test_split"),
                                            pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
                                            flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                            normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                            remove_node=self.remove_node,
                                            device=device)
                seq_ids = [0] if self.dry_run else None
                dataset_path = paths.uwbimu_dir if not self.use_virtual_uwb else os.path.join(paths.uwbimu_dir,
                                                                                              "sigma0")
                eval_uwb_imu = evaluate_model(self.model, dataset_path, pose_evaluator=PoseEvaluator(), evaluate_pose=True,
                                              evaluate_zmp=True, \
                                              flush_cache=True, evaluate_tran=True, plt_tran=False,
                                              sequence_ids=seq_ids, normalize_uwb=self.normalize_uwb,
                                              flatten_uwb=self.flatten_uwb, remove_node=self.remove_node,
                                              device=device)
                table = print_eval_result([eval_amass, eval_uwb_imu], filter_keys=set(eval_uwb_imu) & set(eval_amass),
                                          title=["    ", "AMASS Dance-DB", "UWB-IMU Test"])
                eval_trans = eval_uwb_imu
            elif self.eval_dataset_name == "uwb-syn":
                seq_ids = [0] if self.dry_run else list(range(30))
                eval_amass = evaluate_model(self.model, os.path.join(paths.amass_dir, "test_split"),
                                            pose_evaluator=PoseEvaluator(), evaluate_pose=True, evaluate_zmp=True, \
                                            flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                            normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                            remove_node=self.remove_node,
                                            device=device)
                seq_ids = [0] if self.dry_run else None
                eval_dip = evaluate_model(self.model, paths.dipimu_dir, pose_evaluator=PoseEvaluator(), evaluate_pose=True,
                                          evaluate_zmp=False, \
                                          flush_cache=True, sequence_ids=seq_ids, normalize_uwb=self.normalize_uwb,
                                          flatten_uwb=self.flatten_uwb,
                                            device=device)
                eval_tc = evaluate_model(self.model, paths.totalcapture_dir, pose_evaluator=PoseEvaluator(),
                                         evaluate_pose=True, evaluate_zmp=True, \
                                         flush_cache=True, evaluate_tran=True, plt_tran=False, sequence_ids=seq_ids,
                                         normalize_uwb=self.normalize_uwb, flatten_uwb=self.flatten_uwb,
                                         remove_node=self.remove_node,
                                            device=device)
                table = print_eval_result([eval_amass, eval_dip, eval_tc],
                                          filter_keys=set(eval_dip) & set(eval_tc) & set(eval_amass),
                                          title=["    ", "AMASS Dance-DB", "DIP-IMU Test", "TotalCapture"])
                eval_trans = eval_tc
            else:
                raise KeyError("Invalid eval dataset name")

            with open(os.path.join(self.eval_dir, f"{phase}_e{epoch}_error_table.csv"), 'w', newline='') as fid:
                fid.write(table.get_csv_string())

            plt.plot([0] + [_ for _ in eval_trans["trans_error"].keys()],
                     [0] + [torch.tensor(_).mean() for _ in eval_trans["trans_error"].values()], label=self.model.name)
            plt.legend(fontsize=15)
            plt.savefig(os.path.join(self.eval_dir, f"{phase}_e{epoch}_translation_error.png"))
            plt.close("all")

    def save_checkpoint(self, file_name, epoch):
        state = {'epoch': epoch,
                 'net': self.model.state_dict(),
                 'optim': self.optimizer.state_dict()
                 }
        torch.save(state, file_name)

    def preprocess(self, data: Batch):
        return data.get_listed_batch(keys=["x_imu", "lj_init", "jvel_init"])

    def forward_model(self, batch: Batch):
        y_pred = self.model(batch)
        assert len(y_pred) == len(self.model.model_output)
        tmp = {k: v for k, v in zip(self.model.model_output, y_pred)}
        return D_Batch(tmp)

    def train_one_epoch(self, epoch, phase=''):

            total_time = 0
            self.model.train()
            if self.resampling_interval > 0 and epoch % self.resampling_interval == 0 and epoch != 0:
                self.dataset.resubsampling()
                self.data_loader = DataLoader(self.dataset, shuffle=True,
                                              pin_memory=True,
                                              batch_size=self.batch_size,
                                              num_workers=3)

            batch_idx = 0
            loss_train = Easy_dict({l.__name__: 0 for l in self.loss_func.loss_func})
            grad_norm = 0
            loop_bar = tqdm(self.data_loader)
            for data_dict in loop_bar:
                self.step += 1
                data = Batch(**data_dict).to_device(self.device)
                data.uwb_normalized = self.normalize_uwb

                y_pred = self.forward_model(data)
                if hasattr(y_pred, "smpl_6d") and y_pred.smpl_6d.requires_grad:
                    y_pred.smpl_6d.retain_grad()

                if 'lgd' in phase:
                    # learnable gradient descent need history data in model
                    loss_dict = self.model.backward(y_pred, data)
                else:
                    loss_dict = self.loss_func.compute_total_loss(y_pred, data, self.step, epoch)

                self.log_losses(loss_dict)

                # Gradient Analysis: Check which loss term contributes most to smpl_6d gradient
                if self.step % 10 == 0 and hasattr(y_pred, "smpl_6d") and y_pred.smpl_6d.requires_grad:
                    for name, loss_val in loss_dict.items():
                        if name == "total_loss" or not isinstance(loss_val, torch.Tensor) or loss_val.numel() == 0:
                            continue
                        
                        if not loss_val.requires_grad:
                            continue

                        # Calculate gradient of this specific loss w.r.t smpl_6d
                        # retain_graph=True is essential as we need the graph for subsequent gradients and the final backward
                        grads = torch.autograd.grad(loss_val, y_pred.smpl_6d, retain_graph=True, allow_unused=True)

                        if grads[0] is not None:
                            wandb.log({
                                f"grad_components/{name}_norm": grads[0].norm().item(),
                                f"grad_components/{name}_max": grads[0].abs().max().item()
                            }, step=self.step, commit=False)

                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()

                if hasattr(y_pred, "smpl_6d") and y_pred.smpl_6d.grad is not None and self.step % 10 == 0:
                    wandb.log({
                        "train/grad_smpl_6d_norm": y_pred.smpl_6d.grad.norm().item(),
                        "train/grad_smpl_6d_var": y_pred.smpl_6d.grad.var().item()
                    }, step=self.step, commit=False)

                if hasattr(y_pred, "smpl_6d"):
                    log_video(self.renderer, self.smpl_layer,
                             y_pred.smpl_6d[0], data.smpl_6d[0],
                              data.shape[0],
                              y_pred.smpl_tran[0], data.smpl_tran[0],
                              "train")

                    log_metrics(y_pred.smpl_tran, data.smpl_tran,
                                y_pred.smpl_6d, data.smpl_6d,
                                y_pred.contact, data.contact_p,
                                self.smpl_layer,"train")

                total_grad_norm = None
                if self.grad_clip > 0:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"],
                                                                     self.grad_clip)
                else:
                    total_grad_norm = torch.linalg.norm(
                        torch.cat([param.grad.view(-1) for param in self.optimizer.param_groups[0]["params"]]))

                self.optimizer.step()

                batch_idx += 1

                # logging
                loss_train._add_item(loss_dict)
                grad_norm += total_grad_norm.item()
                total_loss = loss_dict["total_loss"].item()
                loop_bar.set_description(
                    f"Phase:{phase}||Epoch:{epoch}/{self.epochs}||lr: {self.lr_scheduler.get_last_lr()[0]:.2E}||Loss: {total_loss:.4f}")

            loss_train._div(deno=batch_idx)
            grad_norm /= batch_idx

            val_loss = self.validation(epoch, phase)

            logging_dict = {
                "train_loss": loss_train,
                "grad_norm": grad_norm,
                "val_loss": val_loss
            }

            if self.early_stop_check(val_loss.sum()):
                file_name = os.path.join(self.ckpt_dir, f"{phase}_last_model_{str(epoch).zfill(3)}.pt")
                self.save_checkpoint(file_name=file_name, epoch=epoch)

            if val_loss["total_loss"] < self.lowest_val_loss:
                self.lowest_val_loss = val_loss["total_loss"]
                file_name = os.path.join(self.ckpt_dir, f"{phase}_best_model_{str(epoch).zfill(3)}.pt")
                self.save_checkpoint(file_name=file_name, epoch=epoch)
                # delete previous best model
                for file in glob.glob(os.path.join(self.ckpt_dir, f"{phase}_best_model_*.pt")):
                    if file != file_name:
                        os.remove(file)

            if wandb.run is not None:
                self.wandb_logging(epoch, phase, **logging_dict)

    @torch.no_grad()
    def validation(self, epoch, phase):
        self.model.eval()
        batch_idx = 0
        loss_val = Easy_dict({loss_func.__name__: 0 for loss_func in self.loss_func.loss_func})
        loop_bar = tqdm(self.val_dataloader)
        for data in loop_bar:
            data = Batch(**data).to_device(self.device)

            y_pred = self.forward_model(data)

            if hasattr(y_pred, "smpl_6d"):
                log_video(self.renderer, self.smpl_layer,
                          y_pred.smpl_6d[0], data.smpl_6d[0],
                          data.shape[0],
                          y_pred.smpl_tran[0], data.smpl_tran[0],
                          "val")

                log_metrics(y_pred.smpl_tran, data.smpl_tran,
                            y_pred.smpl_6d, data.smpl_6d,
                            y_pred.contact, data.contact_p,
                            self.smpl_layer, "val")

            loss = self.loss_func.compute_total_loss(y_pred, data, self.step, epoch)

            loss_val._add_item(loss)
            loop_bar.set_description(f"[Validation]Phase:{phase}||Epoch:{epoch}||")

            batch_idx += 1

        loss_val._div(deno=batch_idx)
        self.model.train()
        return loss_val

    def log_losses(self, loss_dict, mode: Literal["train", "val", "test"] = "train"):
        if self.step % 30 != 0:
            return
        log_loss_dict = {f"{mode}/loss/{loss_name}": loss_val.item() for loss_name, loss_val in loss_dict.items()}

        magnitude_loss_dict = {}

        wandb.log(log_loss_dict, step=self.step, commit=False)
