import traceback
import uuid
from pathlib import Path
import random
from typing import Optional, Literal, Iterable

import numpy as np
import torch
import wandb
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import PinholeCamera
from articulate.math import r6d_to_axis_angle, r6d_to_rotation_matrix, rotation_matrix_to_axis_angle

ERROR_MARKER_COLOR = (1.0, 0, 0, 1.0)
K = 1.0
MESH_ERR_THR = 0.42
LOG_VIDEO_RATE = 1000
LOG_VIDEO_RATE_VAL = 100

LOG_METRICS_RATE = 150
LOG_METRICS_RATE_VAL = 30



def visualize_mesh_error(poses_gt, poses_hat, betas, smpl_layer, rgb=(0.62, 0.62, 0.62),
                         trans_gt: Optional[torch.Tensor] = None,
                         trans_hat: Optional[torch.Tensor] = None,
                         root_orient_aa_gt: Optional[torch.Tensor] = None,
                         root_orient_aa_hat: Optional[torch.Tensor] = None,
                         ):
    """
    visualizes the mesh error between the predicted and the ground-truth SMPL meshes.
    """
    vertices_gt, joints_gt = smpl_layer(betas=betas, poses_body=poses_gt)
    vertices, joints = smpl_layer(betas=betas, poses_body=poses_hat)

    frame_size = min(vertices.size(0), vertices_gt.size(0))
    mesh_err = torch.norm(vertices[:frame_size] - vertices_gt[:frame_size], dim=-1)
    color_coef = torch.min(torch.ones_like(mesh_err), K * mesh_err)

    rgba = rgb + (1.0,)
    vertices_rgba = torch.stack(
        [color_coef * (ERROR_MARKER_COLOR[i] - rgba[i]) + rgba[i] for i in range(len(rgba))]).permute(1, 2, 0)

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq_gt = SMPLSequence(poses_body=poses_gt, smpl_layer=smpl_layer, betas=betas,
                            trans=trans_gt,
                            poses_root=root_orient_aa_gt,
                            z_up=False)
    print(torch.mean(mesh_err))

    if trans_hat is not None:
        trans_hat = trans_hat + torch.tensor([2, 0, 0], device=trans_hat.device).repeat(poses_hat.shape[0], 1)
    else:
        trans_hat = torch.tensor([2, 0, 0], device=poses_gt.device).repeat(poses_gt.shape[0], 1)
    smpl_seq = SMPLSequence(poses_body=poses_hat, betas=betas, smpl_layer=smpl_layer,
                               trans=trans_hat,
                               poses_root=root_orient_aa_hat,
                               z_up=False
                               )
    smpl_seq.mesh_seq.vertex_colors = vertices_rgba.cpu().numpy()

    return (smpl_seq_gt, smpl_seq), joints


def render_smpl_sequences(smpl_sequences: Iterable[SMPLSequence], renderer: HeadlessRenderer, **kwargs):
    for seq in smpl_sequences:
        renderer.scene.add(seq)

    targets_stack = torch.stack([seq.trans for seq in smpl_sequences], dim=0)
    targets = torch.mean(targets_stack, dim=0)
    targets = torch.matmul(torch.tensor([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]], device=targets.device).float(),
                           targets.transpose(0, 1))

    # Transpose back if necessary to keep the shape as [N, 3]
    targets = targets.transpose(0, 1)
    targets = targets - torch.tensor([0, 0.0, 0], device=targets.device)

    # compute norm between min and max targets stack
    target_dist_norm = torch.norm(targets_stack.max(dim=0).values - targets_stack.min(dim=0).values, dim=-1)
    target_dist_norm[target_dist_norm <= 3.0] = 3.0
    positions = targets + torch.tensor([0, 0.5, 2.0], device=targets.device).repeat(targets.shape[0], 1) * target_dist_norm.unsqueeze(1)

    targets = targets.detach().cpu().numpy()
    positions = positions.detach().cpu().numpy()

    camera = PinholeCamera(positions, targets, 1280, 720, viewer=renderer)

    renderer.scene.add(camera)
    renderer.set_temp_camera(camera)
    renderer.save_video(**kwargs)

    for seq in smpl_sequences:
        renderer.scene.remove(seq)
        renderer.scene.remove(camera)

def log_video(renderer: HeadlessRenderer, smpl_layer,
              sixd_pred: torch.Tensor, sixd_gt: torch.Tensor,
              betas: torch.Tensor,
              trans_pred: Optional[torch.tensor] = None, trans_gt: Optional[torch.tensor] = None,
              prefix: Literal["train", "val", "test"] = "train",
              video_name: Optional[str] = None, log_video_to_wandb: bool = True):
    """
    Logs a video of the mesh error between the predicted and the ground-truth SMPL meshes.
    :param
    renderer: HeadlessRenderer: The renderer to use
    smpl_layer: SMPL: The SMPL layer
    sixd_pred: torch.tensor: (N x 144) Predicted 6D poses
    sixd_gt: torch.tensor: (N x 144) Ground-truth 6D poses
    betas: torch.tensor: (10) Shape parameters
    trans_pred: torch.tensor: (N x 3) Predicted translations
    trans_gt: torch.tensor: (N x 3) Ground-truth translations
    """



    # log video with p = 1 / log_video_rate
    log_video_rate = LOG_VIDEO_RATE if prefix == "train" else LOG_VIDEO_RATE_VAL
    probability = 1 / log_video_rate
    if (random.random() > probability or log_video_rate <= 0) and not prefix == "test":
        return

    vl = sixd_pred.shape[0]
    gt_aa = r6d_to_axis_angle(sixd_gt.contiguous().view(-1, 6)).reshape(vl, 24 * 3)

    # pred
    head_aa = r6d_to_axis_angle(sixd_pred.contiguous().view(-1, 6)).reshape(vl, 24 * 3)

    gt_aa = gt_aa.detach()
    head_aa = head_aa.detach()

    try:
        seqs, joints = visualize_mesh_error(gt_aa[:, 3:], head_aa[:, 3:], betas, smpl_layer,
                                            trans_gt=trans_gt, trans_hat=trans_pred,
                                            root_orient_aa_gt=gt_aa[:, :3], root_orient_aa_hat=head_aa[:, :3])

    except Exception as e:
        print("Error in visualization of mesh error", e)
        traceback.print_exc()
        return

    if video_name is None:
        path = Path(f"/tmp/{str(uuid.uuid4())}.mp4")
    else:
        path = Path("output/videos_predict") / f"{str(video_name).replace('/', '_')}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)

    render_smpl_sequences(seqs, renderer, output_fps=15, video_dir=str(path))
    new_video_path = path.with_name(path.stem + "_0" + path.suffix)

    print(f"Logging video to {new_video_path}")

    if log_video_to_wandb:
        wandb.log({f"{prefix}/video/mesh_error": wandb.Video(str(new_video_path))})

def angle_between(rot_mat1: torch.Tensor, rot_mat2: torch.Tensor):
    r"""
    https://github.dev/Xinyu-Yi/PIP/blob/0e8df58d3b67ac5922626d72e9d6b74a068df108/articulate/math/angular.py#L84

    Calculate the angle in radians between two rotations. (torch, batch)

    :param rot1: Rotation matrix 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation matrix 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    offsets = rot_mat1.transpose(1, 2).bmm(rot_mat2)
    angles = rotation_matrix_to_axis_angle(offsets).norm(dim=1)
    return angles

def radian_to_degree(q):
    r"""
    Convert radians to degrees.
    """
    return q * 180.0 / np.pi


@torch.no_grad()
def get_metrics(trans_pred: torch.tensor, trans_gt: torch.tensor, sixd_pred: torch.tensor, sixd_gt: torch.tensor,
                contact_pred: torch.tensor, contact_gt: torch.tensor, smpl_layer):
    """
    Computes different metrics and returns the results as a dictionary.

    :param
    trans_pred: torch.tensor: (B x N x 3) Predicted translations
    trans_gt: torch.tensor: (B x N x 3) Ground-truth translations
    sixd_pred: torch.tensor: (B x N x 144)Predicted 6D poses
    sixd_gt: torch.tensor: (B x N x 144) Ground-truth 6D poses
    contact_pred: torch.tensor: (B x N x C) Predicted contact points logits
    contact_gt: torch.tensor: (B x N x C) Ground-truth contact points logits
    smpl_layer: SMPL: The SMPL layer
    """



    metrics = {}


    # Compute the mean translation error
    mse = torch.mean(torch.norm(trans_pred - trans_gt, dim=-1))
    metrics["mean_translation_error"] = mse

    # compute joint angle error
    rot_mat_pred = r6d_to_rotation_matrix(sixd_pred.reshape(-1, 6)).reshape(sixd_pred.shape[0], -1, 24, 3, 3)
    rot_mat_gt = r6d_to_rotation_matrix(sixd_gt.reshape(-1, 6)).reshape(sixd_gt.shape[0], -1, 24, 3, 3)
    joint_angle_error = angle_between(rot_mat_pred[:, :, 1:].reshape(-1, 3, 3), rot_mat_gt[:, :, 1:].reshape(-1, 3, 3))
    joint_angle_error_deg = radian_to_degree(joint_angle_error).mean()
    metrics["mean_joint_angle_error"] = joint_angle_error_deg.item()

    # compute the orientation error
    joint_angle_error_orientation = angle_between(rot_mat_pred[:, :, 0].view(-1 ,3 ,3), rot_mat_gt[:, :, 0].view(-1 ,3 ,3))
    joint_angle_error_orientation_deg = radian_to_degree(joint_angle_error_orientation).mean()
    metrics["mean_joint_angle_error_orientation"] = joint_angle_error_orientation_deg


    # Calculate the Mean Per Joint Position Error (MPJPE)
    gt_aa = r6d_to_axis_angle(sixd_gt.reshape(-1, 6)).reshape(sixd_gt.shape[0], -1, 24 * 3)
    pred_aa = r6d_to_axis_angle(sixd_pred.reshape(-1, 6)).reshape(sixd_gt.shape[0], -1, 24 * 3)
    pjpe = []
    for pred_aa_s, gt_aa_s in zip(pred_aa, gt_aa):
        # Get the joint positions
        _, joints_pred = smpl_layer(poses_body=pred_aa_s[:, 3:], poses_root=pred_aa_s[:, :3], betas=torch.zeros(10, device=pred_aa_s.device))
        _, joints_gt = smpl_layer(poses_body=gt_aa_s[:, 3:], poses_root=gt_aa_s[:, :3], betas=torch.zeros(10, device=pred_aa_s.device))

        joints_pred = joints_pred[:, :24]
        joints_gt = joints_gt[:, :24]

        # MPJPE calculation
        mpjpe = torch.norm(joints_pred - joints_gt, dim=-1)
        mean_mpjpe = torch.mean(mpjpe)
        pjpe.append(mean_mpjpe)

    mean_mpjpex = torch.mean(torch.stack(pjpe))
    metrics["mean_per_joint_position_error"] = mean_mpjpex.item()

    # Compute the contact point accuracy
    contact_pred = torch.sigmoid(contact_pred) > 0.5
    contact_accuracy = torch.mean((contact_pred == contact_gt).float())
    metrics["contact_accuracy"] = contact_accuracy.item()


    return metrics


def log_metrics(trans_pred: torch.tensor, trans_gt: torch.tensor, sixd_pred: torch.tensor, sixd_gt: torch.tensor,
                contact_pred: torch.tensor, contact_gt: torch.tensor,
                smpl_layer, prefix: Literal["train", "val", "test"] = "train"):
    """
    Logs the metrics for the given predictions and ground-truth values.
    logging metrics is very slow so should only be done at low rate.
    """
    # log video with p = 1 / log_video_rate
    log_video_rate = LOG_METRICS_RATE if prefix == "train" else LOG_METRICS_RATE_VAL
    probability = 1 / log_video_rate
    if (random.random() > probability or log_video_rate <= 0) and not prefix == "test":
        return

    metrics = get_metrics(trans_pred, trans_gt, sixd_pred, sixd_gt, contact_pred, contact_gt, smpl_layer)
    wandb.log({f"{prefix}/metrics/{k}": v for k, v in metrics.items()})
