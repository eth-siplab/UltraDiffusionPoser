import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.dataset.data_utils import Batch, D_Batch

__all__ = ["mapping_loss_func", "registered_loss", "Loss_Func", "Easy_dict"]

# Fallback loss set, used only when `--losses` is not supplied. UDP normally
# passes `losses`/`loss_weights`/`loss_start_epoch` explicitly via the config.
mapping_loss_func = {
    "diffusion_all": ["simple_diff_loss", "smpl_tran_loss", "smpl_6d_loss", "contact_loss", "uwb_loss"],
}
mapping_loss_weight = {}

registered_loss = {}


def register_loss_func(func):
    registered_loss[func.__name__] = func
    return func


@register_loss_func
def simple_diff_loss(batch: Batch, pred: D_Batch, **kwargs):
    motion_repr_clean = pred.motion_repr_clean
    motion_repr_pred = pred.motion_repr_pred
    return nn.MSELoss()(motion_repr_clean.cuda(), motion_repr_pred.cuda())


@register_loss_func
def uwb_loss(batch: Batch, pred: D_Batch, **kwargs):
    y_uwb_pred = pred.vuwb_pred
    y_uwb_gt = batch.uwb_gt
    return nn.MSELoss()(y_uwb_gt.cuda(), y_uwb_pred)


@register_loss_func
def smpl_6d_loss(batch: Batch, pred: D_Batch, **kwargs):
    y_full_joint_pred = pred.smpl_6d
    y_full_joint_gt = batch.smpl_6d
    return F.l1_loss(y_full_joint_gt.cuda(), y_full_joint_pred)


@register_loss_func
def smpl_tran_loss(batch: Batch, pred: D_Batch, **kwargs):
    y_full_joint_pred = pred.smpl_tran
    y_full_joint_gt = batch.smpl_tran
    return F.mse_loss(y_full_joint_gt.cuda(), y_full_joint_pred)


@register_loss_func
def smpl_tran_vel_loss(batch: Batch, pred: D_Batch, **kwargs):
    fps = 60
    y_full_joint_pred = pred.smpl_tran
    y_full_joint_gt = batch.smpl_tran

    # Per-frame translation deltas -> speed (m/s).
    y_full_rel_tran_pred = y_full_joint_pred[:, 1:] - y_full_joint_pred[:, :-1]
    y_full_rel_tran_gt = y_full_joint_gt[:, 1:] - y_full_joint_gt[:, :-1]
    y_full_vel_pred = torch.norm(y_full_rel_tran_pred, dim=-1) * fps
    y_full_vel_gt = torch.norm(y_full_rel_tran_gt, dim=-1) * fps
    return F.mse_loss(y_full_vel_gt.cuda(), y_full_vel_pred)


@register_loss_func
def contact_loss(batch: Batch, pred: D_Batch, **kwargs):
    y_contact = pred.contact
    y_contact_gt = batch.contact_p
    if isinstance(y_contact, list):
        y_contact = torch.stack(y_contact)
    if y_contact.shape[2] == 2:
        # if only two contact points are predicted, run loss on joints 10/11 only
        # (y_contact_gt is in order of joint_set foot indices)
        y_contact_gt = y_contact_gt[:, :, [1, 3]]
    return contact_loss_func(y_contact_gt.cuda(), y_contact)


def contact_loss_func(c_gt, c_pred):
    c_l = F.binary_cross_entropy(torch.sigmoid(c_pred), c_gt.float())
    return c_l


class Easy_dict(dict):

    def _div(self, deno=1):
        for key, value in self.items():
            self[key] = value / deno

    def div(self, deno=1):
        out_dict = Easy_dict()
        for key, value in self.items():
            out_dict[key] = value / deno
        return out_dict

    def sum(self):
        sum = 0
        for value in self.values():
            sum += value
        return sum

    def _add(self, a: dict):
        for key, value in self.items():
            self[key] = value + a[key]

    def _add_item(self, a: dict):
        for key, value in a.items():
            if key in self:
                self[key] = self[key] + value
            else:
                self[key] = value


class Loss_Func:
    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("Loss Function Config")
        group.add_argument('--losses', type=str, nargs='+', default=None, help='Losses to use')
        group.add_argument('--loss_weights', type=float, nargs='+', default=None, help='Weights of losses')
        group.add_argument('--loss_start_epoch', type=int, nargs='+', default=None, help='Epoch to start using losses')

    def __init__(self, opt) -> None:
        self.mapping_loss_func = mapping_loss_func
        self.loss_func_kwargs = {}
        self.losses = opt.losses
        self.loss_weight = opt.loss_weights
        self.loss_start_epoch = opt.loss_start_epoch

    def set_training_phase(self, phase):
        if self.losses is not None:
            self.loss_func = [registered_loss[k] for k in self.losses]
            assert self.loss_weight is not None
            if self.loss_weight is None:
                self.loss_weight = [0] * len(self.loss_func)
        else:
            mode, module = phase.split("_", 1)
            if mode in ['baseline', 'finetune']:
                self.loss_func = [registered_loss[k] for k in self.mapping_loss_func[module]]
            else:
                raise KeyError(f"Invalid phase mode{mode}")

            if module in mapping_loss_weight:
                self.loss_weight = mapping_loss_weight[module]
            else:
                self.loss_weight = [1] * len(self.loss_func)

            self.loss_start_epoch = [0] * len(self.loss_func)

        assert len(self.loss_func) == len(self.loss_weight)
        print(f"Update loss function {self.loss_func[0].__name__} for phase {phase}")

    def compute_total_loss(self, y_pred: D_Batch, batch: Batch, step, epoch):
        def use_loss(epoch, step, start_epoch):
            return epoch >= start_epoch
        kwargs = self.loss_func_kwargs
        loss_dict = Easy_dict({loss_func.__name__: loss_func(batch, y_pred, **kwargs) * weight
                               for weight, loss_func, start_epoch in
                               zip(self.loss_weight, self.loss_func, self.loss_start_epoch)
                               if use_loss(epoch, step, start_epoch)})
        loss_dict["total_loss"] = loss_dict.sum()
        return loss_dict
