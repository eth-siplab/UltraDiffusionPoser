# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import argparse
import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer

from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.scene.camera import PinholeCamera
from aitviewer.viewer import Viewer
import plotly.graph_objs as go
from scipy.ndimage import gaussian_filter1d


from config.config import SMPL_JOINTS
from modules.util.rotation_conversions import matrix_to_axis_angle
C.update_conf({"smplx_models": os.getenv("SMPLX_MODEL_PATH", "data/smpl_m_lbs_10_207_0_v1.0.0.pkl"),
               "device": "cpu",
               "auto_set_floor": True,
               "z_up": False,})


def smooth_tensor(tensor, sigma=2):
    """
    Applies Gaussian smoothing along the first dimension of the tensor.

    Args:
        tensor (torch.Tensor): Input tensor of shape (N, feature_dim).
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        torch.Tensor: Smoothed tensor with the same shape as input.
    """
    # Convert tensor to numpy (move to CPU if necessary)
    tensor_np = tensor.cpu().numpy()
    # Apply Gaussian filter along the time axis (axis=0)
    smoothed_np = gaussian_filter1d(tensor_np, sigma=sigma, axis=0)
    # Convert back to torch.Tensor and ensure it's on the original device
    return torch.from_numpy(smoothed_np).to(tensor.device)

def load_uwb_imu_dataset(path, seq_id, rgb, seq_end=-1, stride=2, beta=None):
    data = torch.load(path)

    # Get the data.
    poses = data["pose"][seq_id].view(-1, 72)
    tran = data["tran"][seq_id].view(-1, 3)
    oris = data["ori"][seq_id].numpy().reshape(-1, 6, 3, 3)
    accs = data["acc"][seq_id].numpy().reshape(-1, 6, 3)
    if "vuwb" in data:
        uwb = data["vuwb"][seq_id].numpy().reshape(-1, 6, 6)
        uwb = uwb[:seq_end:2]
    else:
        uwb = None

    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]

    oris = oris[:seq_end:stride]
    tran = tran[:seq_end:stride]
    accs = accs[:seq_end:stride]
    tran = torch.zeros_like(tran)
    # DIP has no shape information, assume the mean shape.
    if beta is None:
        betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    else:
        betas = beta.repeat(poses.shape[0], 1).float().to(C.device)

    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)
    poses[:, 20 * 3:22 * 3] = 0  # zero out hand pose

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device),
    )

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    print(betas)
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3], trans=tran,
                            betas=betas)
    smpl_seq.mesh_seq.color = rgb + (1.0,)

    return smpl_seq, joints, oris, accs, uwb


def visualize_smpl_models(path, rgb=(0.62, 0.62, 0.62), seq_end=-1, stride=2, vis_leaf_joint_position=False):
    data = torch.load(path)
    uwb_imu_rot = np.array([[1, 0, 0], [0, 0, 1.0], [0, -1, 0]])
    # Get the data.
    poses = matrix_to_axis_angle(data[0]).view(-1, 72)
    tran = data[1].view(-1, 3)
    # Subject 6 is female, all others are male (cf. metadata.txt included in the downloaded zip file).
    gender = "male"

    # Downsample to 30 Hz.
    poses = poses[:seq_end:stride]
    tran = tran[:seq_end:stride]
    # DIP has no shape information, assume the mean shape.
    betas = torch.zeros((poses.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender=gender, device=C.device)

    # We need to anchor the IMU orientations somewhere in order to display them.
    # We can do this at the joint locations, so perform one forward pass.
    _, joints = smpl_layer(
        poses_body=poses[:, 3:].to(C.device),
        poses_root=poses[:, :3].to(C.device),
        betas=betas,
        trans=tran.to(C.device)
    )

    # Display the SMPL ground-truth with a semi-transparent mesh so we can see the IMUs.
    smpl_seq = SMPLSequence(poses_body=poses[:, 3:], smpl_layer=smpl_layer, poses_root=poses[:, :3], trans=tran)
    smpl_seq.mesh_seq.color = rgb + (1.0,)

    return smpl_seq, joints


def visualize_leaf_joint_position(path, joints, seq_end=-1, stride=2):
    data = torch.load(path)

    # Get the data.
    tran = data[1].view(-1, 3)
    root_ori = data[0][:, 0]
    leaf_joint_position = data[2].view(-1, 5, 3)

    # Downsample to 30 Hz.
    leaf_joint_position = leaf_joint_position[:seq_end:stride]
    root_ori = root_ori[:seq_end:stride]
    tran = tran[:seq_end:stride]
    f, n, _ = leaf_joint_position.size()
    leaf_joint_position = leaf_joint_position @ root_ori.permute(0, 2, 1)

    root_position = np.tile(joints[:, 0].cpu().numpy(), (1, 5)).reshape(f, n, 3)
    arr_head = Arrows(origins=root_position[:, [2]],
                      tips=root_position[:, [2]] - leaf_joint_position[:, [2]].cpu().numpy(), color=(0, 0, 0.5, 1))  # b
    arr_leg = Arrows(origins=root_position[:, [0, 1]],
                     tips=root_position[:, [0, 1]] - leaf_joint_position[:, [0, 1]].cpu().numpy(),
                     color=(0, 0.5, 0, 1))  # g
    arr_upper = Arrows(origins=root_position[:, [3, 4]],
                       tips=root_position[:, [3, 4]] - leaf_joint_position[:, [3, 4]].cpu().numpy(),
                       color=(0.5, 0, 0, 1))  # r

    return [arr_head, arr_leg, arr_upper]


def show_vertex_positions(vertices_gt, joints_gt):
    # The vertices you want to locate
    vertex_indices = torch.tensor([1961, 5424, 1176, 4662, 411, 3021], device=C.device)

    # Extract positions of these vertices from the first frame
    vertices_gt_positions = vertices_gt[0, vertex_indices, :].cpu().numpy()

    # Extract all vertex positions for the first frame (entire mesh)
    vertices_gt_all = vertices_gt[0].cpu().numpy()

    # Create a scatter plot for the entire mesh (all vertices)
    mesh_scatter = go.Scatter3d(
        x=vertices_gt_all[:, 0], y=vertices_gt_all[:, 1], z=vertices_gt_all[:, 2],
        mode='markers',
        # add hover text of id
        text=[f"ID: {idx}" for idx in range(vertices_gt_all.shape[0])],
        marker=dict(size=2, color='lightgrey'),  # Mesh vertices appear in light grey
        name='SMPL Mesh'
    )

    # Create a scatter plot for the key vertices you want to highlight
    highlight_scatter = go.Scatter3d(
        x=vertices_gt_positions[:, 0], y=vertices_gt_positions[:, 1], z=vertices_gt_positions[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red'),  # Highlighted vertices in red
        text=[f"ID: {idx}" for idx in vertex_indices.cpu().numpy()],  # Add vertex IDs for hover text
        textposition='top center',
        name='Highlighted Vertices',
        hoverinfo='text'  # Display only the ID on hover
    )

    # show the joint positions and there names
    joint_positions = joints_gt[0].cpu().numpy()
    joint_names = SMPL_JOINTS
    joint_scatter = go.Scatter3d(
        x=joint_positions[:, 0], y=joint_positions[:, 1], z=joint_positions[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='blue'),  # Highlighted vertices in red
        text=joint_names,  # Add vertex IDs for hover text
        textposition='top center',
        name='Joints',
        hoverinfo='text'  # Display only the ID on hover
    )

    # Setup the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'  # Keep aspect ratio
        ),
        title='SMPL Mesh with Highlighted Vertices'
    )

    # Create the figure and add both the full mesh and the highlighted vertices
    fig = go.Figure(data=[mesh_scatter, highlight_scatter, joint_scatter], layout=layout)

    # Display the plot
    fig.show()
def visualize_multiple_sequences(paths, smpl_layer, seq_end=-1, stride=2, no_translation=False, use_root_gt=False, rgb=(0.62, 0.62, 0.62), plot_trans=True):
    """
    Visualizes multiple SMPL sequences side by side. The first path is assumed to be the ground truth.
    Each additional sequence is shifted by 2 meters along the x-axis.
    """
    # Load the ground truth data (first path in the list)
    gt_data = torch.load(paths[0])
    gt_data = (torch.tensor(gt_data[0], dtype=torch.float32), torch.tensor(gt_data[1], dtype=torch.float32))
    poses_gt = gt_data[0].view(-1, 72)[:seq_end:stride].to(C.device)
    start_pos_xz = gt_data[1].view(-1, 3)[0].to(C.device).clone()
    trans_gt = gt_data[1].view(-1, 3)[:seq_end:stride].to(C.device) - start_pos_xz
    if no_translation:
        trans_gt = torch.zeros_like(trans_gt)

    # Assuming betas are all zero
    betas = torch.zeros((poses_gt.shape[0], 10)).float().to(C.device)

    # Compute the SMPL vertices and joints for the ground truth
    vertices_gt, joints_gt = smpl_layer(
        poses_body=poses_gt[:, 3:].to(C.device),
        poses_root=poses_gt[:, :3].to(C.device),
        betas=betas,
    )

    minimum_vertex_frame_zero = torch.min(vertices_gt[0], dim=0)[0]

    sequences = []  # Store SMPL sequences
    trans_shift = 2  # Start shifting by 2m along x-axis

    # Create the SMPL sequence for the ground truth
    poses_gt[:, -12:] = 0
    smpl_seq_gt = SMPLSequence(
        poses_body=poses_gt[:, 3:],
        poses_root=poses_gt[:, :3],
        smpl_layer=smpl_layer,
        trans=trans_gt,
        betas=betas,
        name="Ground Truth"
    )
    green_rgb = (0.52, 0.7, 0.52)
    smpl_seq_gt.mesh_seq.color = green_rgb + (1.0,)  # Set the color for the ground-truth sequence
    sequences.append(smpl_seq_gt)

    translations = [trans_gt]
    # Loop over the prediction paths and process each one
    for i, pred_path in enumerate(paths[1:], start=1):
        pred_data = torch.load(pred_path)

        pred_data = (torch.tensor(pred_data[0], dtype=torch.float32), torch.tensor(pred_data[1], dtype=torch.float32))
        if i == 0:
            # smooth tensor
            pred_data = (smooth_tensor(pred_data[0], sigma=6), smooth_tensor(pred_data[1], sigma=2))

        if pred_data[0].shape[1:] == (24, 3, 3):
            pred_aa = matrix_to_axis_angle(pred_data[0]).view(-1, 72)
        else:
            pred_aa = pred_data[0].view(-1, 72)
        poses_pred = pred_aa[:seq_end:stride].to(C.device)
        if use_root_gt:
            poses_pred[:, :3] = poses_gt[:, :3]

        trans_pred = pred_data[1].view(-1, 3)[:seq_end:stride].to(C.device)
        print(f"len of {pred_path}, {trans_pred.shape[0]}")

        trans_pred = trans_pred - trans_pred[0]
        translations.append(trans_pred.clone())
        if no_translation:
            trans_pred = torch.zeros_like(trans_pred)

        # Compute SMPL vertices and joints for the prediction
        betas = torch.zeros((poses_pred.shape[0], 10)).float().to(C.device)
        vertices_pred, _ = smpl_layer(
            poses_body=poses_pred[:, 3:].to(C.device),
            poses_root=poses_pred[:, :3].to(C.device),
            betas=betas,
        )

        # Compute the vertex error and create a color map based on the error
        frame_size = min(vertices_pred.size(0), vertices_gt.size(0))
        mesh_err = torch.norm(vertices_pred[:frame_size] - vertices_gt[:frame_size], dim=-1)
        color_coef = torch.min(torch.ones_like(mesh_err), 1.0 * mesh_err)

        rgba = rgb + (1.0,)
        ERROR_COLOR = [1.0, 0.0, 0.0, 1.0]
        vertices_rgba = torch.stack(
            [color_coef * (ERROR_COLOR[j] - rgba[j]) + rgba[j] for j in range(len(rgba))]
        ).permute(1, 2, 0)

        # Shift each prediction by an additional 2 meters on the x-axis
        trans_pred_shifted = trans_pred + torch.tensor([trans_shift * (i), 0, 0], device=C.device).repeat(frame_size, 1)

        # Create SMPL sequence for the prediction
        smpl_seq_pred = SMPLSequence(
            poses_body=poses_pred[:, 3:],
            poses_root=poses_pred[:, :3],
            smpl_layer=smpl_layer,
            trans=trans_pred_shifted,
            betas=betas,
            name=f"pred: {pred_path.parent.name}"
        )
        smpl_seq_pred.mesh_seq.vertex_colors = vertices_rgba.cpu().numpy()  # Apply vertex-based coloring for the prediction

        sequences.append(smpl_seq_pred)

    if plot_trans:
        plot_translations(translations)

    return sequences, minimum_vertex_frame_zero


def plot_translations(translations):
    """
    Plot translations along the xz-axis using Plotly for scientific visualization.

    Parameters:
    translations: list of 5 tensors (numpy arrays), each of shape (n_frames, 3),
                  where columns represent (x, y, z) coordinates.
    """
    assert len(translations) == 5, "Expected 5 translation tensors"

    label_order = ["GT", "UDP", "UIP", "PIP", "TIP"]
    colors = ["black", "green", "blue", "orange", "purple"]  # Distinct colors for each label

    MAX_FRAMES = 23 * 60
    fig = go.Figure()

    for i, (translation, label, color) in enumerate(zip(translations, label_order, colors)):
        x, z = translation[:MAX_FRAMES, 0], translation[:MAX_FRAMES, 2]  # Extract x and z coordinates
        fig.add_trace(go.Scatter(
            x=x, y=z, mode='lines', name=label, line=dict(color=color, width=2)
        ))

    # Configure layout for a clean scientific-style figure
    fig.update_layout(
        title=dict(
            text="Trajectory of Translations (XZ Plane)",
            font=dict(size=24)  # Increase the title font size
        ),
        xaxis_title="X Coordinate",
        yaxis_title="Z Coordinate",
        xaxis=dict(scaleanchor="y", title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
        template="plotly_white",
        width=600, height=600,  # Maintain 1:1 aspect ratio
        legend=dict(font=dict(size=16)),
    )
    #use raleway font
    fig.update_layout(font=dict(family="Raleway"))
    # set apspect mode to data
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()

    # save figure as pdf
    fig.write_image("translations.pdf")

    fig.write_image("translations.png")

def render_smpl_sequences_viewer(smpl_sequences: Iterable[SMPLSequence], min_vertex: Optional[torch.Tensor] = None):
    """
    Displays the given SMPL sequences using the Viewer.
    """
    if min_vertex is not None:
        C.update_conf({"auto_set_floor": False})

    v = Viewer()
    v.playback_fps = 60.0

    for seq in smpl_sequences:
        v.scene.add(seq)

    v.scene.camera.is_ortho = False

    # set the position of the camera
    v.scene.camera.position = np.array([0 , 0, 15])
    v.scene.camera.target = np.array([0, 100, 15])

    v.scene.floor.enabled = True
    if not C.auto_set_floor:
        axis = 2 if C.z_up else 1
        v.scene.floor.position[axis] = min_vertex[axis]
        v.scene.floor.update_transform(parent_transform=v.scene.model_matrix)
    v.scene.origin.enabled = False

    # Run the viewer to display the sequences
    v.run()

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation process')

    # Modify seq_res_path to accept multiple paths
    parser.add_argument('--seq_res_paths', type=str, nargs='+',
                        help="Specify the sequence paths (ground truth first, followed by predictions)")

    parser.add_argument('--seq_id', type=int, default=1,
                        help='Result sequence id to run')

    parser.add_argument("--show_gt", action="store_true",
                        help="If set, show ground truth alongside predictions")
    parser.add_argument("--no_translation", action="store_true", help="If set, do not show translation, root is fixed")
    parser.add_argument("--plot_translations", action="store_true", help="If set, plot translations")
    parser.add_argument("--no_root_rotation", action="store_true", help="If set, use gt root rotation")
    args = parser.parse_args()
    return args, parser


if __name__ == "__main__":
    args, _ = get_args()
    seq_id = args.seq_id
    paths = args.seq_res_paths  # Now taking the paths directly from the argument list
    paths = [Path(p) / f"{seq_id}.pt" for p in paths]
    stride = 1

    if args.show_gt:

        # Visualize both ground truth and prediction side by side
        smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
        sequences, min_vertex = visualize_multiple_sequences(paths, smpl_layer, seq_end=12000, stride=stride,
                                                 no_translation=args.no_translation,
                                                 use_root_gt=args.no_root_rotation,
                                                 plot_trans=args.plot_translations)
        # Display both sequences in the viewer
        render_smpl_sequences_viewer(sequences, min_vertex)
    else:
        smpl_seqs = []
        for p in paths:
            smpl_s, joint = visualize_smpl_models(p, rgb=(0.348, 0.395, 0.628), seq_end=12000, stride=stride)
            smpl_seqs.append(smpl_s)

        # Add everything to the scene and display at 30 fps
        v = Viewer()
        v.playback_fps = 30.0
        v.scene.add(*smpl_seqs)
        v.run()