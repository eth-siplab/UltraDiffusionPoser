# Copyright (c) Meta Platforms, Inc. and affiliates.
# Part of this code is based on https://github.com/GuyTevet/motion-diffusion-model

import torch
from typing import Literal

import config.config
from articulate import ParametricModel
from articulate.math import r6d_to_rotation_matrix, rotation_matrix_to_r6d
from config.config import paths, IMU_NUM, get_input_size
from modules.dataset.preprocess import _syn_uwb, vi_mask
from modules.model.diffusion.heads import *

SMPL_6D_LOCAL_DIM = 6 * 24

class PoseNet(nn.Module):
    def __init__(self, dataset, diff_dim, cond_dim, out_dim,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1, activation="gelu",
                 body_model_path='',
                 device=None,
                 traj_feat_dim=4,
                 weight_loss_rec_repr_full_body=0.0,
                 weight_loss_repr_foot_contact_mse=0.0,
                 weight_loss_joint_pos_global=0.0,
                 weight_loss_joint_vel_global=0.0, weight_loss_joint_smooth=0.0,
                 weight_loss_foot_skating=0.0,
                 start_skating_loss_epoch=0,
                 model_type: Literal["lstm", "transformer_encoder"] = "lstm",
                 positional_encoding: Literal["sinusoidal", "learned"] = "sinusoidal",
                 ):
        super().__init__()
        self.model_type = model_type
        self.dataset = dataset
        self.diff_dim = diff_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.traj_feat_dim = traj_feat_dim

        # contact lbl dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
        # we use only two contact points
        self.foot_joint_index_list = config.config.joint_set.foot_indices
        self.foot_skating_vel_thres = 0.02
        self.fps = 60

        self.latent_dim = latent_dim  # 512
        self.ff_size = ff_size  # 1024
        self.num_layers = num_layers  # 8
        self.num_heads = num_heads  # 4
        self.dropout = dropout  # 0.1
        self.activation = activation
        self.normalize_output = False

        self.mse_loss = nn.MSELoss(reduction='none').to(device)
        self.l1_loss = nn.L1Loss(reduction='none').to(device)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)
        self.device = device

        self.weight_loss_rec_repr_full_body = weight_loss_rec_repr_full_body
        self.weight_loss_repr_foot_contact_mse = weight_loss_repr_foot_contact_mse
        self.weight_loss_joint_pos_global = weight_loss_joint_pos_global
        self.weight_loss_joint_vel_global = weight_loss_joint_vel_global
        self.weight_loss_joint_smooth = weight_loss_joint_smooth
        self.weight_loss_foot_skating = weight_loss_foot_skating
        self.start_skating_loss_epoch = start_skating_loss_epoch

        self.input_process = InputProcess(self.diff_dim, self.latent_dim)
        self.input_process_cond = InputProcess(self.cond_dim, self.latent_dim)
        self.input_process_history = InputProcess(self.diff_dim, self.latent_dim) # global traj + smpl parameters
        if positional_encoding == "sinusoidal":
            self.sequence_pos_encoder = SinusoidalPositionalEncoding(self.latent_dim, self.dropout, max_len=5000)
        elif positional_encoding == "learned":
            self.sequence_pos_encoder = LearnedPositionalEncoding(self.latent_dim, self.dropout, max_len=5000)
        else:
            raise ValueError("positional_encoding should be either 'sinusoidal' or 'learned'")
        if self.model_type == "transformer_encoder":
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                         num_layers=self.num_layers
                                                         )
        elif self.model_type == "lstm":
            self.lstm = nn.LSTM(input_size=self.latent_dim, hidden_size=self.latent_dim, num_layers=self.num_layers,
                                batch_first=False, dropout=self.dropout)
        else:
            raise ValueError("model_type should be either 'lstm' or 'transformer_encoder'")

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.output_process = OutputProcess(self.out_dim, self.latent_dim)

        self.history_token = nn.Parameter(torch.randn(1, self.latent_dim)*0.5, requires_grad=True)
        self.test_seq = nn.Parameter(torch.randn(201, 1, self.latent_dim)*0.5, requires_grad=True)
        self.test_input = nn.Parameter(torch.randn(201, 1, self.latent_dim)*0.5, requires_grad=True)

        self.random_numbers = nn.Parameter(torch.randn((201, 1, 512)), requires_grad=False)

        self.smpl_model = ParametricModel(paths.smpl_file, device="cuda")



    # @profile
    def forward(self, batch, timesteps):
        """
        Processes an input batch of features and conditions on previous motion features
        (if available) to output a sequence of motion predictions.

        Args:
            batch (dict): Dictionary containing input features:
                - 'x_t' (torch.Tensor): The current motion features of shape [batch_size, body_feat_dim, 1, T].
                - 'cond' (torch.Tensor): Conditioning features of shape [batch_size, cond_feat_dim, 1, T].
                - 'history' (optional, torch.Tensor): Auto-regressive history of shape [T', batch_size, feature_dim].
                - 'motion_mask' (torch.Tensor): Mask for the current motion features, of shape [batch_size, T].
            timesteps (torch.Tensor): Tensor of timestep embeddings, of shape [batch_size].

        Returns:
            torch.Tensor: The predicted motion sequence, of shape [batch_size, pose_feat_dim, 1, T].
        """
        t, bs = batch['x_t'].shape[-1], batch['x_t'].shape[0]
        diff_ts_embed = self.embed_timestep(timesteps)  # [1, bs, 512]

        x = self.input_process(batch['x_t'])
        cond = self.input_process_cond(batch['cond'])
        x = x + cond

        # adding the timestep embed
        if "history" in batch:
            attend_hist_mask = batch["auto_regressive_mask"]  # [B] - the mask on which auto-regressive history is present
            history = self.input_process_history(batch["history"]) # T', bs, d
            history[:, attend_hist_mask] = 0
            history = history + self.history_token
            history_length = history.shape[0]

            xseq = torch.cat([diff_ts_embed, history, x], dim=0) # [1+T'+T, bs, d]
            if self.model_type != "transformer_encoder":
                xseq[1:] = xseq[1:] + diff_ts_embed
            if self.model_type == "transformer_encoder":
                xseq[1:, attend_hist_mask] = self.sequence_pos_encoder(xseq[1:, attend_hist_mask]) # apply positional encoding to all (when history is present)

            # ignore history in positional encoding if auto-regressive mask is false
            x_seq_non_history = torch.cat([xseq[0, ~attend_hist_mask].unsqueeze(0), xseq[history_length+1:, ~attend_hist_mask]], dim=0) # [1+T, hist_mask_count, bs, d]
            if self.model_type == "transformer_encoder":
                x_seq_non_history[1:] = self.sequence_pos_encoder(x_seq_non_history[1:])
            xseq[0, ~attend_hist_mask] = x_seq_non_history[0, :]
            xseq[history_length+1:, ~attend_hist_mask] = x_seq_non_history[1:, :] # N, hist_mask_count, d

            # build the attention mask, if auto regression is used the history is attended to
            history_attention_mask = attend_hist_mask.unsqueeze(1).expand(bs, history_length) # [bs, history_length]
            motion_mask = batch["motion_mask"] # [B, N]

            attention_mask = torch.cat((torch.ones(bs, 1, device=history_attention_mask.device), history_attention_mask, motion_mask), dim=1) # [bs, 1+T'+T]
            attention_mask = ~attention_mask.bool() # 0 is attended to, 1 is not attended to
        else:
            xseq = torch.cat((diff_ts_embed, x), dim=0)  # [T+1, bs, d]
            if self.model_type != "transformer_encoder":
                xseq[1:] = xseq[1:] + diff_ts_embed
            
            if self.model_type == "transformer_encoder":
                xseq[1:] = self.sequence_pos_encoder(xseq[1:])  # [T+1, bs, d]

            attention_mask = torch.zeros(bs, t+1, device=xseq.device, dtype=torch.bool)  # [bs, T+1]

        seq_len, batch_size, dim = xseq.shape

        if self.model_type == "transformer_encoder":
            output = self.seqTransEncoder(xseq, src_key_padding_mask=attention_mask) #, src_key_padding_mask=attention_mask) # [1 + T (+ T'), bs, d]
        elif self.model_type == "lstm":
            output = self.lstm(xseq)[0] # [1 + T (+ T'), bs, d]

        # remove the timestep embedding token
        output = output[1:] # [(T' + ) T , bs, d] = [(hist|motion_seq, bs, d]

        output = self.output_process(output)  # [bs, pose_feat_dim, 1, T]

        model_output = {}
        if "history" in batch:
            # extract 0:T from output if history is present
            # extract T:T+T' from output if history is present
            model_output["motion_seq"] = output[:, :, :, history_length:]
            model_output["history_seq"] = batch["history"]

        else:
            model_output["motion_seq"] = output

        return model_output["motion_seq"]


    def compute_losses_with_smpl(self, batch, model_output, smplx_model=None, epoch=0):
        # model_output: [bs, pose_feat_dim, 1, T]
        loss_dict = {}

        ###################### loss on full body repr
        loss_rec_repr_all = self.mse_loss(batch['motion_repr_clean'], model_output)
        loss_rec_repr_all = loss_rec_repr_all[:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        loss_dict['loss_repr_full_body'] = loss_rec_repr_all[:, :, self.traj_feat_dim:-4].mean()

        ###################### loss on global joint coordinate
        full_repr_clean = batch['motion_repr_clean'][:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        full_repr_clean = full_repr_clean * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_clean = {}
        for repr_name in REPR_LIST:
            repr_dict_clean[repr_name] = full_repr_clean[..., cur_total_dim:(cur_total_dim+REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        joint_pos_clean = recover_from_repr_smpl(repr_dict_clean, recover_mode='joint_abs_traj', smplx_model=smplx_model)

        full_repr_rec = model_output[:, :, 0].permute(0, 2, 1)  # [bs, T, 263]
        full_repr_rec = full_repr_rec * torch.from_numpy(self.dataset.Std).to(self.device) + torch.from_numpy(self.dataset.Mean).to(self.device)
        # reconstruct joint positions
        cur_total_dim = 0
        repr_dict_rec = {}
        for repr_name in REPR_LIST:
            repr_dict_rec[repr_name] = full_repr_rec[..., cur_total_dim:(cur_total_dim + REPR_DIM_DICT[repr_name])]
            cur_total_dim += REPR_DIM_DICT[repr_name]
        joint_pos_rec_from_abs_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_abs_traj', smplx_model=smplx_model)  # [bs, clip_len, 22, 3]
        joint_pos_rec_from_rel_traj = recover_from_repr_smpl(repr_dict_rec, recover_mode='joint_rel_traj', smplx_model=smplx_model)
        joint_pos_rec_from_smpl = recover_from_repr_smpl(repr_dict_rec, recover_mode='smplx_params', smplx_model=smplx_model)
        loss_dict['loss_joint_pos_global_from_abs_traj'] = self.mse_loss(joint_pos_rec_from_abs_traj, joint_pos_clean).mean()
        loss_dict['loss_joint_pos_global_from_rel_traj'] = self.mse_loss(joint_pos_rec_from_rel_traj, joint_pos_clean).mean()
        loss_dict['loss_joint_pos_global_from_smpl'] = self.mse_loss(joint_pos_rec_from_smpl, joint_pos_clean).mean()

        ###################### loss on global joint velocity
        joint_vel_clean = joint_pos_clean[:, 1:] - joint_pos_clean[:, 0:-1]
        joint_vel_rec_from_abs_traj = joint_pos_rec_from_abs_traj[:, 1:] - joint_pos_rec_from_abs_traj[:, 0:-1]  # [bs, clip_len-1, 22, 3]
        joint_vel_rec_from_rel_traj = joint_pos_rec_from_rel_traj[:, 1:] - joint_pos_rec_from_rel_traj[:, 0:-1]
        joint_vel_rec_from_smpl = joint_pos_rec_from_smpl[:, 1:] - joint_pos_rec_from_smpl[:, 0:-1]
        loss_dict['loss_joint_vel_global_from_abs_traj'] = self.mse_loss(joint_vel_rec_from_abs_traj, joint_vel_clean).mean()
        loss_dict['loss_joint_vel_global_from_rel_traj'] = self.mse_loss(joint_vel_rec_from_rel_traj, joint_vel_clean).mean()
        loss_dict['loss_joint_vel_global_from_smpl'] = self.mse_loss(joint_vel_rec_from_smpl, joint_vel_clean).mean()

        ###################### accel smooth regularizor
        joint_acc_rec_from_abs_traj = joint_vel_rec_from_abs_traj[:, 1:] - joint_vel_rec_from_abs_traj[:, 0:-1]
        joint_acc_rec_from_rel_traj = joint_vel_rec_from_rel_traj[:, 1:] - joint_vel_rec_from_rel_traj[:, 0:-1]
        joint_acc_rec_from_smpl = joint_vel_rec_from_smpl[:, 1:] - joint_vel_rec_from_smpl[:, 0:-1]
        loss_dict['loss_joint_smooth_from_abs_traj'] = torch.mean(joint_acc_rec_from_abs_traj ** 2)
        loss_dict['loss_joint_smooth_from_rel_traj'] = torch.mean(joint_acc_rec_from_rel_traj ** 2)
        loss_dict['loss_joint_smooth_from_smpl'] = torch.mean(joint_acc_rec_from_smpl ** 2)

        ###################### contact lbl loss
        loss_dict['loss_repr_foot_contact_mse'] = self.mse_loss(batch['motion_repr_clean'][:, -4:, :, :], model_output[:, -4:, :, :]).mean()

        ###################### foot skating loss
        contact_lbl_gt = full_repr_clean[:, :, -4:]  # [bs, clip_len, 4] 1 - in contact, 0 - not in contact

        foot_joint_rec_vel_from_abs_traj = (joint_pos_rec_from_abs_traj[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_abs_traj[:, 0:-1, self.foot_joint_index_list]) * self.fps  # [bs, clip_len-1, 4, 3]
        foot_joint_rec_vel_from_abs_traj = torch.norm(foot_joint_rec_vel_from_abs_traj, dim=-1)  # [bs, clip_len-1, 4]
        mask_skating_from_abs_traj = (foot_joint_rec_vel_from_abs_traj - self.foot_skating_vel_thres).gt(0)  # [bs, clip_len-1, 4]  True/False
        mask_skating_from_abs_traj = mask_skating_from_abs_traj * contact_lbl_gt[:, 0:-1]  # [bs, clip_len-1, 4]
        masked_foot_joint_rec_vel_from_abs_traj = foot_joint_rec_vel_from_abs_traj * mask_skating_from_abs_traj  # [bs, clip_len-1, 4]
        loss_dict['loss_foot_skating_from_abs_traj'] = masked_foot_joint_rec_vel_from_abs_traj.sum() / mask_skating_from_abs_traj.sum()

        foot_joint_rec_vel_from_rel_traj = (joint_pos_rec_from_rel_traj[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_rel_traj[:, 0:-1, self.foot_joint_index_list]) * self.fps
        foot_joint_rec_vel_from_rel_traj = torch.norm(foot_joint_rec_vel_from_rel_traj, dim=-1)
        mask_skating_from_rel_traj = (foot_joint_rec_vel_from_rel_traj - self.foot_skating_vel_thres).gt(0)
        mask_skating_from_rel_traj = mask_skating_from_rel_traj * contact_lbl_gt[:, 0:-1]
        masked_foot_joint_rec_vel_from_rel_traj = foot_joint_rec_vel_from_rel_traj * mask_skating_from_rel_traj
        loss_dict['loss_foot_skating_from_rel_traj'] = masked_foot_joint_rec_vel_from_rel_traj.sum() / mask_skating_from_rel_traj.sum()

        foot_joint_rec_vel_from_smpl = (joint_pos_rec_from_smpl[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_rec_from_smpl[:, 0:-1, self.foot_joint_index_list]) * self.fps
        foot_joint_rec_vel_from_smpl = torch.norm(foot_joint_rec_vel_from_smpl, dim=-1)
        mask_skating_from_smpl = (foot_joint_rec_vel_from_smpl - self.foot_skating_vel_thres).gt(0)
        mask_skating_from_smpl = mask_skating_from_smpl * contact_lbl_gt[:, 0:-1]
        masked_foot_joint_rec_vel_from_smpl = foot_joint_rec_vel_from_smpl * mask_skating_from_smpl
        loss_dict['loss_foot_skating_from_smpl'] = masked_foot_joint_rec_vel_from_smpl.sum() / mask_skating_from_smpl.sum()

        if epoch >= self.start_skating_loss_epoch:
            weight_loss_foot_skating = self.weight_loss_foot_skating
        else:
            weight_loss_foot_skating = 0.0


        loss_dict["loss"] = self.weight_loss_rec_repr_full_body * loss_dict['loss_repr_full_body'] + \
                            self.weight_loss_repr_foot_contact_mse * loss_dict['loss_repr_foot_contact_mse'] + \
                            self.weight_loss_joint_pos_global * (loss_dict['loss_joint_pos_global_from_abs_traj'] + loss_dict['loss_joint_pos_global_from_rel_traj'] + loss_dict['loss_joint_pos_global_from_smpl']) + \
                            self.weight_loss_joint_vel_global * (loss_dict['loss_joint_vel_global_from_abs_traj'] + loss_dict['loss_joint_vel_global_from_rel_traj'] + loss_dict['loss_joint_vel_global_from_smpl'])+ \
                            self.weight_loss_joint_smooth * (loss_dict['loss_joint_smooth_from_abs_traj'] + loss_dict['loss_joint_smooth_from_rel_traj'] + loss_dict['loss_joint_smooth_from_smpl']) + \
                            weight_loss_foot_skating * (loss_dict['loss_foot_skating_from_abs_traj'] + loss_dict['loss_foot_skating_from_rel_traj'] + loss_dict['loss_foot_skating_from_smpl'])
        return loss_dict

    def fk_from_motion_repr(self, full_repr_rec, batch, return_vertices=False):
        B, T, pose_feat_dim = full_repr_rec.shape
        cond = batch["cond"]
        SMPL_SHAPE_DIM = 10
        SMPL_6D_LOCAL_DIM = 6 * 24
        shape = cond[:, -SMPL_SHAPE_DIM:, 0, 0]

        joint_angle_glb = []
        joint_pos_glb = []
        vertex_pos_glb = []
        for i in range(B):
            smple_6d_local = full_repr_rec[i, :, :SMPL_6D_LOCAL_DIM].reshape(-1, 6)
            smpl_rot_mat_local = r6d_to_rotation_matrix(smple_6d_local).reshape(T, 24, 3, 3)
            tran = full_repr_rec[i, :, SMPL_6D_LOCAL_DIM:SMPL_6D_LOCAL_DIM+3]
            shape = shape[i]
            if return_vertices:
                pose_global_p, joint_p, vertex_p = self.smpl_model.forward_kinematics(smpl_rot_mat_local, shape, tran, calc_mesh=True)
            else:
                pose_global_p, joint_p = self.smpl_model.forward_kinematics(smpl_rot_mat_local, shape, tran, calc_mesh=False)

            joint_angle_glb.append(pose_global_p)
            joint_pos_glb.append(joint_p)
            if return_vertices:
                vertex_pos_glb.append(vertex_p)

        joint_angle_glb = torch.stack(joint_angle_glb, dim=0)
        joint_pos_glb = torch.stack(joint_pos_glb, dim=0)
        if return_vertices:
            vertex_pos_glb = torch.stack(vertex_pos_glb, dim=0)
            return joint_angle_glb, joint_pos_glb, vertex_pos_glb
        return joint_angle_glb, joint_pos_glb


    def guide_with_uwb(self, batch, out, denoise_t, compute_grad='x_t'):
        # model_output: [bs, pose_feat_dim, 1, T]
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()

            # obtain global joint coordinate
            full_repr_rec = x_t[:, :, 0].permute(0, 2, 1)  # [bs, T, body_feat_dim]
            B, T, _ = full_repr_rec.shape

            # run fk to get joint pos
            _, joint_pos_glb, vertices_glb = self.fk_from_motion_repr(full_repr_rec, batch, return_vertices=True)

            # compute pairwise distances # WARNING not batch compatible
            uwb_smpl_pred = _syn_uwb(vertices_glb[:, :, vi_mask]).unsqueeze(0) * 100

            UWB_START = get_input_size(["vacc", "vrot"])
            UWB_END = get_input_size(["vacc", "vrot", "vuwb"])
            uwb_gt = batch["cond"][:, UWB_START:UWB_END, 0, :].permute(0, 2, 1).view(B, T, IMU_NUM, IMU_NUM) * 100 # [bs, T, NUM_IMU*NUM_IMU]

            # set lower triangular to 0 to not count distances twice due to symmetry
            mask = torch.tril(torch.ones(IMU_NUM, IMU_NUM), diagonal=-1).to(x_t.device).unsqueeze(0).unsqueeze(0)
            uwb_gt = uwb_gt * mask
            uwb_smpl_pred = uwb_smpl_pred * mask

            # compute mse loss
            loss_uwb = self.mse_loss(uwb_smpl_pred, uwb_gt).mean()

            grad_uwb = torch.autograd.grad([-loss_uwb], [x_t])[0]  # [bs, body_feat_dim, 1, T] same shape as x_t, body_feat_dim=294
            grad_uwb[:, :6, :, :] = 0  # root rotation should stay
            grad_uwb[:, SMPL_6D_LOCAL_DIM:SMPL_6D_LOCAL_DIM + 3, :, :] = 0  # for traj repr part
            grad_uwb[:, -2:, :, :] = 0  # for contact label

            x_t.detach()

        return grad_uwb


    def guide_skating_with_smpl(self, batch, out, denoise_t, compute_grad='x_t'):

        # model_output: [bs, pose_feat_dim, 1, T]
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, body_feat_dim, 1, T]

            # obtain global joint coordinate
            full_repr_rec = x_t[:, :, 0].permute(0, 2, 1)  # [bs, T, body_feat_dim]

            # run fk to get joint pos
            _, joint_pos_glb = self.fk_from_motion_repr(full_repr_rec, batch, return_vertices=False)

            ############# foot skating loss
            # extract contact label from motion representation and run softmax if needed
            contact_pred = full_repr_rec[:, :, -len(config.config.joint_set.foot_indices):].detach().clone()
            # apply element-wise sigmoid to get contact label
            contact_pred = torch.sigmoid(contact_pred)
            contact_mask = torch.zeros_like(contact_pred)
            contact_mask[contact_pred > 0.5] = 1.0
            contact_mask[contact_pred <= 0.5] = 0.0

            ### joints from smpl
            foot_joint_rec_vel_from_smpl = (joint_pos_glb[:, 1:, self.foot_joint_index_list] -
                                            joint_pos_glb[:, :-1, self.foot_joint_index_list])
            foot_joint_rec_vel_from_smpl = torch.norm(foot_joint_rec_vel_from_smpl, dim=-1)
            mask_skating_from_smpl = (foot_joint_rec_vel_from_smpl - self.foot_skating_vel_thres).gt(0)
            total_mask = mask_skating_from_smpl * contact_mask[:, :-1]
            masked_foot_joint_rec_vel_from_smpl = foot_joint_rec_vel_from_smpl * total_mask
            if total_mask.sum() != 0:
                loss_foot_skating_from_smpl = masked_foot_joint_rec_vel_from_smpl.sum()
            else:
                loss_foot_skating_from_smpl = torch.tensor(0.0).to(self.device)

            if total_mask.sum() != 0:
                loss_foot_skating = loss_foot_skating_from_smpl
                grad_skating = torch.autograd.grad([-loss_foot_skating], [x_t])[0]  # [bs, body_feat_dim, 1, T] same shape as x_t, body_feat_dim=294
                grad_skating[:, -len(config.config.joint_set.foot_indices):, :, :] = 0  # for contact label
            else:
                grad_skating = torch.tensor(0.0).to(self.device)
            x_t.detach()

        return grad_skating


    def guide_with_orientations(self, batch, out, denoise_t, compute_grad='x_t'):
        """
        We assume to have sensor orientations as input, so our output should be aligned with them.

        This guidance loss tries to the smpl output with the orientation conditioning.
        """
        with torch.enable_grad():
            if compute_grad == 'x_t':
                x_t = batch['x_t']
            elif compute_grad == 'x_0':
                x_t = out['pred_xstart']
            x_t = x_t.detach().requires_grad_()  # [bs, body_feat_dim, 1, T]

            # obtain global joint coordinate of the joints where sensors are attached to
            full_repr_rec = x_t[:, :, 0].permute(0, 2, 1)  # [bs, T, body_feat_dim]
            B, T, _ = full_repr_rec.shape

            # run fk to get joint angles
            joints_angle_glb, _ = self.fk_from_motion_repr(full_repr_rec, batch, return_vertices=False)

            joints_ori_glb = joints_angle_glb[:, :, config.config.IMU_placement.ji_mask]

            ROT_START = get_input_size(["vacc"])
            ROT_END = get_input_size(["vacc", "vrot"])
            cond_rot = batch["cond"][:, ROT_START:ROT_END, 0, :].permute(0, 2, 1).view(B, T, IMU_NUM, 3, 3)

            # both to 6d representation
            joints_ori_glb_6d = rotation_matrix_to_r6d(joints_ori_glb.reshape(-1, 3, 3)).reshape(B, T, IMU_NUM, 6)
            sensor_ori_glb_6d = rotation_matrix_to_r6d(cond_rot.reshape(-1, 3, 3)).reshape(B, T, IMU_NUM, 6)

            # compute mse loss
            loss_ori = self.mse_loss(joints_ori_glb_6d, sensor_ori_glb_6d).mean()

            # Compute gradient with respect to x_t
            grad_ori = torch.autograd.grad([-loss_ori], [x_t])[0]  # [bs, body_feat_dim, 1, T]

            # Apply masks to preserve certain parts of the motion
            # Don't affect translation part
            grad_ori[:, SMPL_6D_LOCAL_DIM:SMPL_6D_LOCAL_DIM + 3, :, :] = 0

            # Don't affect contact labels
            grad_ori[:, -len(config.config.joint_set.foot_indices):, :, :] = 0
            x_t.detach()

        return grad_ori







