import os
import random

import torch
import matplotlib.pyplot as plt
import matplotlib
from aitviewer.headless import HeadlessRenderer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.spheres import Spheres
from aitviewer.viewer import Viewer

import plotly.graph_objs as go
from aitviewer.configuration import CONFIG as C
C.update_conf({"smplx_models": os.getenv("SMPL_MODEL_PATH", "data/smpl_m_lbs_10_207_0_v1.0.0.pkl"),
               "device": "cpu",
               "window_type": "pyglet"})

RENDERER = HeadlessRenderer()

def compute_mds(distances, dim=3):
    """
    Computes the MDS coordinates from Euclidean distance matrices.

    Works for inputs of shape [..., D, D], where `...` represents any number of leading dimensions.

    Args:
    - distances: torch.Tensor of shape [..., D, D], where D is the number of points.
    - dim: int, the number of dimensions to return (default is 3 -> 3D space).

    Returns:
    - torch.Tensor of shape [..., D, dim], containing the MDS coordinates.
    """
    # Get the shape of the input
    *leading_dims, D, _ = distances.shape

    # Create the identity and centering matrices based on D
    I = torch.eye(D, device=distances.device)  # Shape (D, D)
    H = I - torch.ones((D, D), device=distances.device) / D  # Centering matrix (D, D)

    # Reshape distances to (..., D, D)
    distances_flat = distances.view(-1, D, D)  # Flatten the leading dimensions for easier batch processing

    # Compute the B matrix
    B_matrix = -0.5 * H @ distances_flat ** 2 @ H  # Apply centering and compute B

    # Perform SVD
    U, S, Vh = torch.linalg.svd(B_matrix)

    # Compute Y using the square root of singular values
    sqrt_S = torch.sqrt(S)  # Shape (batch_size, D)

    # Ensure the dimensions are broadcastable for matrix multiplication
    Y = U @ torch.diag_embed(sqrt_S)  # Shape (batch_size, D,

    # Reshape Y back to the original leading dimensions
    Y = Y.view(*leading_dims, D, D)

    # Return the first 'dim' components of Y
    return Y[..., :dim]


def resolve_reflections(X, threshold=0.8):
    """
    Resolves the points if a reflection happened in the sequence.
    X: BxNx6x3, each item of the batch is a sequence of 6 joint motions. The problem is that elements of the sequence can be reflected.
    This funcition resolves the reflections by comparing the distance between the previous points and the next points.
    If they are bigger than some threshold, the element of that sequence is reflected.
    """


    diff = torch.diff(X, dim=-3, prepend=X[..., :1, :, :])
    dist = torch.norm(diff, dim=-1)
    sum_dist = torch.sum(dist, dim=-1)

    # Determine if a reflection happened (i.e., if the distance exceeds the threshold)
    reflection = sum_dist > threshold

    # Cumulative reflection toggling
    reflection_cumsum = torch.cumsum(reflection, dim=-1) % 2

    # Create a reflection mask targeting only the X-coordinate
    reflection_mask = reflection_cumsum.unsqueeze(-1).expand_as(X[..., 0])

    # Reflect the X-coordinate while leaving Y and Z intact
    X_reflected = X.clone()
    X_reflected[..., 2] = torch.where(reflection_mask == 1, -X[..., 2], X[..., 2])

    return X_reflected


def reflect_mds(X):
    """
    The MDS recovers point up to a reflection. Since it is unclear if the points are reflected or not, we also compute the reflected points.

    For that we reflex X along the YZ plane.
    """
    X_reflected = X.clone()
    X_reflected[..., :, 2] = -X_reflected[..., :, 2]
    return X_reflected

def normalize_mds(X):
    """
    Normalizes the MDS points of the leaf joints ([lw, rw, lk, rk, head, root])
    such that:
    - The root joint is at the origin (0, 0, 0)
    - The head root vector is aligned with the y-axis
    - the left wrist is on the XY plane and the x component is positive

    Args:
    - X: torch.tensor of shape (B, N, 6, 3), MDS points of the leaf joints

    Returns:
    - torch.tensor of shape (B, N, 6, 3), normalized MDS points
    """
    B, N, _, _ = X.shape
    X = X - X[:, :, -1:, :]  # Subtract the root joint from all joints


    # align head-root vector with y-axis
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    # Head and root vector
    head_root_vec = X[:, :, -2, :]  # Head is the second to last joint (index 4)

    # Normalize the head-root vector
    head_root_vec = head_root_vec / torch.norm(head_root_vec, dim=-1, keepdim=True)

    # Rotation matrix to align head-root vector with the y-axis
    y_axis = torch.tensor([0, 1, 0], dtype=X.dtype, device=X.device).unsqueeze(0).unsqueeze(0)
    v = torch.cross(head_root_vec, y_axis.expand_as(head_root_vec), dim=-1)
    s = torch.norm(v, dim=-1, keepdim=True)
    c = torch.einsum('bij,bij->bi', head_root_vec, y_axis.expand_as(head_root_vec)).unsqueeze(-1)  # Dot product

    def skew_symmetric(v):
        """
        Computes the skew-symmetric matrix for a batch of 3D vectors v.
        v is expected to have shape (B, N, 3) where B is the batch size, N is the number of vectors, and 3 represents the vector components.

        Returns:
        - A skew-symmetric matrix of shape (B, N, 3, 3)
        """
        B, N, _ = v.shape
        v_skew = torch.zeros(B, N, 3, 3, device=v.device, dtype=v.dtype)

        # Fill in the skew-symmetric matrix elements
        v_skew[:, :, 0, 1] = -v[:, :, 2]
        v_skew[:, :, 0, 2] = v[:, :, 1]
        v_skew[:, :, 1, 0] = v[:, :, 2]
        v_skew[:, :, 1, 2] = -v[:, :, 0]
        v_skew[:, :, 2, 0] = -v[:, :, 1]
        v_skew[:, :, 2, 1] = v[:, :, 0]

        return v_skew

    scale_f = ((1 - c) / (s ** 2)).unsqueeze(-1)
    R = torch.eye(3, dtype=X.dtype, device=X.device).unsqueeze(0).unsqueeze(0) + skew_symmetric(v) + \
        skew_symmetric(v) @ skew_symmetric(v) * scale_f

    # rotate the points R @ X
    X_rotated = torch.matmul(R, X.transpose(-1, -2)).transpose(-1, -2)

    ## Second alignment: Align left wrist with XY plane and that
    # Left wrist is the first joint (index 0)
    left_wrist = X_rotated[:, :, 0, :]  # Shape (B, N, 3)

    # Compute the magnitude of the left wrist vector that is projected onto the xz plane
    wrist_magnitude = torch.norm(left_wrist[..., [0, 2]], dim=-1, keepdim=True)  # Shape (B, N, 1)

    # The normal vector to the XY plane is [0, 0, 1], so z-component is relevant
    z_component = left_wrist[:, :, 2]  # Z-component of the left wrist
    x_component = left_wrist[:, :, 0]

    # Compute cos(theta) and theta
    cos_theta = z_component / wrist_magnitude.squeeze(-1)  # Cosine of the angle
    theta = torch.acos(cos_theta)  # Angle to rotate by

    x_positive_mask = left_wrist[:, :, 0] > 0
    z_positive_mask = left_wrist[:, :, 2] > 0

    theta[x_positive_mask & z_positive_mask] = torch.pi / 2 - theta[x_positive_mask & z_positive_mask]
    theta[~x_positive_mask & z_positive_mask] = torch.pi/2 + theta[~x_positive_mask & z_positive_mask]
    theta[~x_positive_mask & ~z_positive_mask] = torch.pi/2 + theta[~x_positive_mask & ~z_positive_mask]
    theta[x_positive_mask & ~z_positive_mask] = - (theta[x_positive_mask & ~z_positive_mask] - torch.pi / 2)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)

    # Rotation matrix for y-axis rotation (B, N, 3, 3)
    R_y = torch.zeros(B, N, 3, 3, dtype=X.dtype, device=X.device)
    R_y[:, :, 0, 0] = cos_theta
    R_y[:, :, 0, 2] = sin_theta
    R_y[:, :, 1, 1] = 1  # Y-axis stays unchanged
    R_y[:, :, 2, 0] = -sin_theta
    R_y[:, :, 2, 2] = cos_theta

    # Rotate all points using the new rotation matrix
    X_aligned = torch.matmul(R_y, X_rotated.transpose(-1, -2)).transpose(-1, -2)

    return X_aligned


def plot_mds_points_on_body(list_of_points, smpl_aa, align=True):
    """
    Uses AIT viewer to plot multiple sets of points on the SMPL body.
    If align is true, they get aligned on the body using a fixed set of indices of the vertex mesh.

    Args:
    - list_of_points: A list of torch.tensors, each of shape (B, N, 3), representing predicted points.
    - smpl_aa: torch.tensor of shape (B, 72), SMPL axis-angle poses.
    - align: bool, whether to align the points using the SMPL mesh vertices.
    """
    if len(smpl_aa.shape) == 3:
        smpl_aa = smpl_aa.view(-1, 24 * 3)
    smpl_aa = smpl_aa.to(C.device)

    smpl_layer = SMPLLayer(model_type="smpl", gender="male", device=C.device)
    smpl_seq = SMPLSequence(poses_body=smpl_aa[:, 3:], poses_root=smpl_aa[:, :3], smpl_layer=smpl_layer)
    smpl_seq.mesh_seq.color = (0.6, 0.6, 0.6, 0.6)

    # Compute ground truth vertices (from SMPL model)
    betas = torch.zeros((smpl_aa.shape[0], 10)).float().to(C.device)
    vertices_gt, joints_gt = smpl_layer(
        poses_body=smpl_aa[:, 3:].to(C.device),
        poses_root=smpl_aa[:, :3].to(C.device),
        betas=betas,
    )

    # Use specific vertex indices for alignment
    vertex_indices = torch.tensor([1961, 5424, 1176, 4662, 411, 3021], device=C.device)
    vertices_gt_positions = vertices_gt[:, vertex_indices, :]  # Shape (B, 6, 3)

    # List to hold all the sphere objects for visualization
    all_spheres = []

    # Define a list of color pairs (left_color, right_color) for the different point sets
    color_pairs = [
        ((0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 0.0, 1.0)),  # Blue / Red
        ((0.0, 1.0, 1.0, 1.0), (1.0, 0.5, 0.0, 1.0)),  # Cyan / Orange
        ((0.5, 0.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0)),  # Purple / Yellow
        ((1.0, 0.5, 0.5, 1.0), (0.5, 1.0, 0.5, 1.0)),  # Salmon / Mint Green
    ]

    # Iterate through each set of points in the input list
    for i, points in enumerate(list_of_points):
        points = points.to(C.device)

        if align:
            # Align points to SMPL mesh vertices using SVD-based alignment
            points_mean = torch.mean(points, dim=1, keepdim=True)  # Shape (B, 1, 3)
            vertices_mean = torch.mean(vertices_gt_positions, dim=1, keepdim=True)  # Shape (B, 1, 3)

            points_centered = points - points_mean  # Shape (B, N, 3)
            vertices_centered = vertices_gt_positions - vertices_mean  # Shape (B, 6, 3)

            # Compute covariance and perform SVD
            H = torch.einsum('bij,bik->bjk', points_centered, vertices_centered)  # Shape (B, 3, 3)
            U, S, Vt = torch.linalg.svd(H)

            R = torch.einsum('bij,bjk->bik', Vt.transpose(-2, -1), U.transpose(-2, -1))  # Shape (B, 3, 3)

            # Apply the rotation and translation to align points
            points_aligned = torch.einsum('bij,bkj->bik', points_centered, R) + vertices_mean
        else:
            points_aligned = points

        # Convert aligned points to numpy for visualization
        left_points = points_aligned[:, [0, 2]].cpu().detach().numpy()
        right_points = points_aligned[:, [1, 3]].cpu().detach().numpy()
        remaining_points = points_aligned[:, [4, 5]].cpu().detach().numpy()

        # Get the color pair for the current set, cycling through the color list if necessary
        left_color, right_color = color_pairs[i % len(color_pairs)]

        # Plot the points in their assigned colors
        spheres_left = Spheres(left_points, color=left_color)
        spheres_right = Spheres(right_points, color=right_color)
        # The original code used the same color for 'right' and 'rest' points
        spheres_rest = Spheres(remaining_points, color=right_color)

        # Add the spheres for the current point set to our list
        all_spheres.extend([spheres_left, spheres_right, spheres_rest])

    # Plot ground truth points in green
    spheres_gt = Spheres(vertices_gt_positions.cpu().detach().numpy(), color=(0.0, 1.0, 0.0, 1.0))

    # Set up AITViewer for visualization
    sequences = [smpl_seq] + all_spheres + [spheres_gt]
    v = Viewer()
    [v.scene.add(seq) for seq in sequences]

    try:
        v.run()
    except KeyboardInterrupt:
        pass
    finally:
        [v.scene.remove(seq) for seq in sequences]






def plot_points(X1, X2, title: str = "Aligned Points Animation", align=True):
    """
    Plots two sets of points in 3D space with animation over the batch dimension.
    As they might not be aligned in the same space, we align them by assuming X1 to be the GT.
    Uses SVD to align them via a rigid body transformation.
    Uses plotly to animate the points across the batch domain.

    Args:
    - X1: torch.tensor of shape (B, N, 3) GT
    - X2: torch.tensor of shape (B, N, 3) Prediction
    - title: str, title of the plot
    - align: bool, whether the points should get alligned
    """
    num_joints = X1.shape[1]
    joint_labels = [str(i) for i in range(num_joints)]

    if align:
        # Compute means for each batch
        X1_mean = torch.mean(X1, dim=1, keepdim=True)  # Shape (B, 1, 3)
        X2_mean = torch.mean(X2, dim=1, keepdim=True)  # Shape (B, 1, 3)

        # Center the points for each batch
        X1_centered = X1 - X1_mean  # Shape (B, N, 3)
        X2_centered = X2 - X2_mean  # Shape (B, N, 3)

        # Compute covariance matrices for each batch
        H = torch.einsum('bij,bik->bjk', X2_centered, X1_centered)  # Shape (B, 3, 3)

        # Perform SVD for each batch
        U, S, Vt = torch.linalg.svd(H)  # U, Vt shapes (B, 3, 3), S shape (B, 3)

        # Compute the rotation matrices for each batch
        R = torch.einsum('bij,bjk->bik', Vt.transpose(-2, -1), U.transpose(-2, -1))  # Shape (B, 3, 3)

        # Fix improper rotations (ensure determinant is positive)
        det_R = torch.det(R)
        negative_det_indices = (det_R < 0).nonzero(as_tuple=True)[0]
        if len(negative_det_indices) > 0:
            Vt[negative_det_indices, 2, :] *= -1
            R = torch.einsum('bij,bjk->bik', Vt.transpose(-2, -1), U.transpose(-2, -1))

        # Apply the rotation and translation to X2
        X2_aligned = torch.einsum('bij,bkj->bik', X2_centered, R) + X1_mean  # Shape (B, N, 3)
    else:
        X2_aligned = X2

    # Create the data for the animation
    frames = []
    for i in range(X1.shape[0]):  # Iterate over the batch dimension
        # Ground truth points (X1) for the current batch
        X1_batch = X1[i].cpu().numpy()
        # Aligned points (X2_aligned) for the current batch
        X2_aligned_batch = X2_aligned[i].cpu().numpy()

        # Create scatter plot traces for both X1 and X2 with joint labels
        trace1 = go.Scatter3d(
            x=X1_batch[:, 0],
            y=X1_batch[:, 1],
            z=X1_batch[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='blue'),
            text=joint_labels,  # Add joint labels to points
            textposition='top center',
            name='Ground Truth (X1)'
        )

        trace2 = go.Scatter3d(
            x=X2_aligned_batch[:, 0],
            y=X2_aligned_batch[:, 1],
            z=X2_aligned_batch[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=joint_labels,  # Add joint labels to points
            textposition='top center',
            name='Aligned Points (X2)'
        )

        # Add frame for each batch
        frames.append(go.Frame(data=[trace1, trace2], name=f"Frame {i}"))

    # Initial plot with the first frame
    X1_initial = X1[0].cpu().numpy()
    X2_aligned_initial = X2_aligned[0].cpu().numpy()

    trace1_initial = go.Scatter3d(
        x=X1_initial[:, 0],
        y=X1_initial[:, 1],
        z=X1_initial[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='blue'),
        text=joint_labels,  # Add joint labels to points
        textposition='top center',
        name='Ground Truth (X1)'
    )

    trace2_initial = go.Scatter3d(
        x=X2_aligned_initial[:, 0],
        y=X2_aligned_initial[:, 1],
        z=X2_aligned_initial[:, 2],
        mode='markers+text',
        marker=dict(size=5, color='red'),
        text=joint_labels,  # Add joint labels to points
        textposition='top center',
        name='Aligned Points (X2)'
    )

    # Create layout with 3D axis titles
    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode = 'data',  # Force manual aspect ratio
        ),
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 30, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate'
                    }]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }]
    )

    # Create the figure with initial data and frames
    fig = go.Figure(data=[trace1_initial, trace2_initial], layout=layout, frames=frames)

    # Show the animated plot
    fig.show()

def test_plot_mds_2d():
    """
    Test if the MDS function returns the correct output shape.
    Visualize the output in 2D.
    """

    points = torch.tensor([[0.0, 0], [1, 1], [0, 1]])
    distances = torch.cdist(points, points, p=2)
    result = compute_mds(distances, dim=2)
    assert result.shape == (3, 2)

    plt.scatter(result[:, 0], result[:, 1])
    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

def test_plot_mds_3d():
    X1 = torch.randn(50, 10, 3)  # Example ground truth points
    cdist = torch.cdist(X1, X1, p=2)
    X2 = compute_mds(cdist, dim=3)  # Example predicted points

    plot_points(X1, X2, "3D Point Alignment Example")

def test_plots_mds_3d_noise():
    X1 = torch.randn(1, 10, 3)  # Example ground truth points
    cdist = torch.cdist(X1, X1, p=2) + torch.randn(10, 10) * 0.1 # Add noise to the distance matrix
    X2 = compute_mds(cdist, dim=3) # Example predicted points

    plot_points(X1, X2, "3D Point Alignment Example with Noise")


if __name__ == '__main__':
    test_plot_mds_3d()
