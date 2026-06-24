import os
import time
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d
from smplx import SMPL
import plotly.graph_objects as go
from torch import no_grad

import articulate as art
from articulate import math, ParametricModel
from articulate.math import r6d_to_rotation_matrix
from articulate.math.angular import r6d_to_axis_angle
from articulate.utils.torch import *
from config.config import *
from modules.dataset.data_utils import Batch
from modules.log.logging_utils import log_video
from modules.model.diffusion.PoseNet import PoseNet
from modules.model.diffusion.resample import create_named_schedule_sampler
from modules.model.diffusion.respace import SpacedDiffusion
from modules.model.diffusion.utils import create_gaussian_diffusion
from modules.model.model_base import BaseModel
from modules.util.mds_util import compute_mds, normalize_mds, resolve_reflections, reflect_mds
from modules.utils import *

from modules.model.diffusion import gaussian_diffusion





class UDP(BaseModel):
    name = "UDP"
    # has to be in order acc, rot, vuwb,
    imu_m = ["vacc", "vrot", "vuwb"]
    model_output = [
        "smpl_6d",
        "smpl_tran",
        "joints_global",
        "contact",
        "mds_points",
        "motion_repr_clean",
        "motion_repr_pred",
        "vuwb_pred",
    ]

    @staticmethod
    def add_args(parser):
        group = parser.add_argument_group("UDP Config")

        group.add_argument('--n_hidden', type=int, default=256, help='Hidden layer width')
        group.add_argument('--n_hidden_rot_est', type=int, default=512, help='Hidden layer width')
        group.add_argument('--diffusion_steps', type=int,
                           help='Number of diffusion steps', default=1000)
        group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine', 'sqrt', 'exponential', "sigmoid"], type=str,
                           help="Noise schedule type")
        group.add_argument("--scale_betas", default=1.0, type=float, help="scales betas, results might depend on noise_schedule.")
        group.add_argument("--timestep_respacing_eval", default='',
                           type=str, help="'' or ddimN, if use ddim, set to 'ddimN', where N denotes ddim sampling steps, empty means ddpm.")
        group.add_argument("--sigma_small", default='True', type=lambda x: x.lower() in ['true', '1'],
                           help="Use smaller sigma values.")
        group.add_argument("--latent_dim", default=512, type=int, help="Latent dimension")
        group.add_argument("--ff_size", default=1024, type=int, help="Feed forward size")
        group.add_argument("--num_layers", default=6, type=int, help="Number of layers")
        group.add_argument("--num_layers_rot_est", default=3, type=int, help="Number of layers for rotation estimator")
        group.add_argument("--num_heads", default=4, type=int, help="Number of heads")
        group.add_argument("--dropout_diff", default=0.1, type=float, help="Dropout rate")
        group.add_argument("--activation", default='gelu', type=str, help="Activation function")
        group.add_argument("--auto_regressive_prob", default=0.5, type=float, help="Auto regressive probability in training.")
        group.add_argument("--auto_regressive_history_length", default=30, type=int, help="Auto regressive history length.")
        group.add_argument("--predict_chunk_length", default=170, type=int, help="Prediction chunk length.")
        group.add_argument("--model_type", default='lstm', type=str, help="Model type")
        group.add_argument("--no_mds", action="store_true", help="Do not use MDS")
        group.add_argument("--positional_encoding", choices=['sinusoidal', 'learned'], default='sinusoidal', help="Positional encoding type")
        group.add_argument("--predict_velocity", action="store_true", help="Typically the model predicts the absolute pose of the body, this flag makes the model predict the velocity that can be integrated to get the pose.")
        group.add_argument("--contact_points", default=4, type=int, help="Number of ground contact points")
        group.add_argument("--clean_uwb_measurement", action="store_true", help="Use a network to clean the UWB measurement")
        group.add_argument("--predict_no_trans", action="store_true", help="Dont predict translation, but set it to 0.")
        group.add_argument("--add_velocity_input", action="store_true", help="Integrates the acceleration to get velocity. This gets added to the conditioning.")
        group.add_argument("--add_position_input", action="store_true", help="Integrates the acceleration twice to get position. This gets added to the conditioning.")
        group.add_argument("--no_rot_est", action="store_true", help="Do not use rotation estimator for MDS correction, just mds, reflection and normalization.")
        group.add_argument("--no_res_est", action="store_true",
                       help="Do not use the residual estimation for MDS correction, just mds, reflection, rotation and normalization.")
        group.add_argument("--no_rotation_spl", action="store_true", help="Dont apply the predicted rotaiton in the mds correction step")
        group.add_argument("--uwb_guidance_lambda", default=5e2, type=float, help="The lambda for the uwb guidance.")

    @staticmethod
    def get_config(args):
        config_str = f"{args.network}-input{UDP.imu_m},-output{UDP.model_output}"
        return config_str

    def _get_input_size(self, sensor_list: Optional[List[str]] = None):
        if sensor_list is None:
            sensor_list = self.imu_m
        return sum([INPUT_DATA_SIZE[k] for k in sensor_list])

    def __init__(self, args) -> None:
        super().__init__(args)
        self.diffusion_type_eval = args.timestep_respacing_eval
        self.use_mds = not args.no_mds
        self.predict_height = args.predict_height
        self.convert_acc_to_g = args.convert_acc_to_g
        self.add_velocity_input = args.add_velocity_input
        self.add_position_input = args.add_position_input

        self.use_uwb = "vuwb" in self.imu_m
        if not self.use_uwb:
            self.use_mds = False

        self.clean_uwb_measurement = args.clean_uwb_measurement
        if self.clean_uwb_measurement:
            self.uwb_smoother = RNN(input_size=IMU_NUM * IMU_NUM,
                                    output_size=IMU_NUM * IMU_NUM,
                                    hidden_size=2 * IMU_NUM * IMU_NUM,
                                    num_rnn_layer=3,
                                    dropout=0.2)

        # imu + mds*2 + shape + (vel) + (pos)
        vel_dim = 3 * IMU_NUM if self.add_velocity_input else 0
        pos_dim = 3 * IMU_NUM if self.add_position_input else 0
        init_rot_estimator_dim = self._get_input_size() + IMU_NUM * 3 * 2 + 10 + vel_dim + pos_dim
        self.init_rot_err_estimator = RNN(input_size=init_rot_estimator_dim,
                                          # Rotations + Initial MDS estimate + smpl shape
                                          output_size=6 + IMU_NUM * 3,
                                          hidden_size=args.n_hidden_rot_est,
                                          num_rnn_layer=args.num_layers_rot_est,
                                          dropout=0.2)

        self.diffusion_train = create_gaussian_diffusion(args, gd=gaussian_diffusion,
                                                         return_class=SpacedDiffusion,
                                                         num_diffusion_timesteps=args.diffusion_steps,
                                                         timestep_respacing=args.timestep_respacing_eval,
                                                         uwb_diffusion_guidance=args.uwb_guidance_lambda
                            )

        # 6d rotation of smpl joints, translation, contact
        self.clean_motion_dim = len(SMPL_JOINTS)*6 + 3 + len(joint_set.foot_indices)
        mds_dim = 3 * 2 * IMU_NUM if self.use_mds else 0

        cond_dim = self._get_input_size() + mds_dim + 10 # ximu + mds estimate + smpl shape
        if self.add_velocity_input:
            cond_dim += 3 * IMU_NUM

        if self.add_position_input:
            cond_dim += 3 * IMU_NUM

        self.model = PoseNet(
            dataset=None,
            diff_dim=self.clean_motion_dim, # smpl 6d + translation + feet contact
            cond_dim=cond_dim,
            out_dim=self.clean_motion_dim,  # smplx_6d + trans + contact
            latent_dim=args.latent_dim,
            ff_size=args.ff_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout_diff,
            activation=args.activation,
            model_type=args.model_type,
            positional_encoding=args.positional_encoding,
        )

        self.bodymodel = ParametricModel(paths.smpl_file, device=args.device)
        self.smpl_bodymodel = SMPL(paths.smpl_file,
                                     batch_size=200,
                                     gender='male',
                                     num_betas=10,
                                     create_transl=False
                                     )

        self.auto_regressive_prob = args.auto_regressive_prob
        self.auto_regressive_history_length = args.auto_regressive_history_length
        self.predict_chunk_length = args.predict_chunk_length
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion_train)
        self.device_param = nn.Parameter(torch.zeros(1))
        self.diffusion_steps = args.diffusion_steps
        self.predict_velocity = args.predict_velocity
        self.ground_contact_points = args.contact_points
        self.predict_no_tran = args.predict_no_trans
        self.no_rot_est = args.no_rot_est
        self.no_res_est = args.no_res_est
        self.no_rotation_spl = args.no_rotation_spl

    @property
    def device(self):
        return self.device_param.device
    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def compute_mds(self, vuwb):
        """
        Compute MDS from data.

        returns:
        - resolved_reflection: B, N, IMU_NUM, 3 # points mds normalized
        - reflected_points_glb: B, N, IMU_NUM, 3 # points mds normalized and reflected
        """
        B, N = vuwb.shape[:2]

        uwb_distance = vuwb.view(B, N, IMU_NUM, IMU_NUM)  # B, N, IMU_NUM, IMU_NUM
        recovered_points_glb = compute_mds(uwb_distance, dim=3)  # B, N, IMU_NUM, 3
        normalized_points_glb = normalize_mds(recovered_points_glb)  # B, N, IMU_NUM, 3
        resolved_reflection = resolve_reflections(normalized_points_glb)  # B, N, IMU_NUM, 3
        reflected_points_glb = reflect_mds(normalized_points_glb)  # B, N, IMU_NUM, 3

        return resolved_reflection, reflected_points_glb

    def correct_mds(self, x_imu, smpl_shape_betas, mds_points, mds_points_reflected):
        """
        MDS is only up to a reflection and a rotation. This function corrects the MDS points by estimating the
        rotation and applies the rotation and some per point error to the MDS points.

        returns:
        - corrected_mds_points: B, N, 3, 2 * IMU_NUM
        """

        B, N = x_imu.shape[:2]
        mds_points_all = torch.cat([mds_points, mds_points_reflected], dim=-2)  # B, N, 2*IMU_NUM, 3

        if self.no_rot_est:
            fixed_mds_points = mds_points_all.permute(0, 1, 3, 2)  # B, N, 3, 2 * IMU_NUM
            return fixed_mds_points

        shape = smpl_shape_betas.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([x_imu, mds_points_all.view(B, N, -1), shape], dim=-1)  # B, N, IMU_data + 2*IMU_NUM*3
        x = self.init_rot_err_estimator(x)  # list of B tensors of shape N, 6+IMU_NUM*3
        rots_6d = torch.stack([x[i][:, :6] for i in range(B)])
        rots_rot_mat = r6d_to_rotation_matrix(rots_6d).view(B, N, 3, 3)
        errors = torch.stack([x[i][:, 6:] for i in range(B)])

        if self.no_rotation_spl:
            fixed_resolved_mds_points = mds_points.transpose(-1, -2)
            fixed_reflected_mds_points = mds_points_reflected.transpose(-1, -2)  # B, N, 3, IMU_NUM
        else:
            fixed_resolved_mds_points = rots_rot_mat @ mds_points.transpose(-1, -2)  # B, N, 3, IMU_NUM
            fixed_reflected_mds_points = rots_rot_mat @ mds_points_reflected.transpose(-1, -2)  # B, N, 3, IMU_NUM
        if not self.no_res_est:
            fixed_resolved_mds_points = fixed_resolved_mds_points + errors.view(B, N, 3, IMU_NUM)  # B, N, 3, IMU_NUM
            fixed_reflected_mds_points = fixed_reflected_mds_points + errors.view(B, N, 3, IMU_NUM)  # B, N, 3, IMU_NUM

        fixed_mds_points = torch.cat([fixed_resolved_mds_points, fixed_reflected_mds_points],
                                     dim=-1)  # B, N, 3, 2 * IMU_NUM
        return fixed_mds_points

    def forward(self, data: Batch, **kwargs):
        """
        Forward pass of the model.

        :param x: A list of length [batch_size], where each element is a 3-tuple:
                  - tensor [num_frames, 72]: IMU data (6 sensors * (9 + 3 + [6 if uwb]) features = 72 [ 108 if uwb])
                  - tensor [15]: Leaf joint position (5 joints * 3 positions = 15)
                  - tensor [72]: Joint velocity in the root frame (24 joints * 3 = 72)

        :return: Tuple containing:
                 - torch.Size([num_frames, 15]): Leaf joint positions (RNN)
                 - torch.Size([num_frames, 15]): Leaf joint positions (GNN)
                 - torch.Size([num_frames, 90]): Global 6D pose
                 - torch.Size([num_frames, 72]): Joint velocity
                 - torch.Size([num_frames, 2]): Contact information
        """
        B, N = data.x_imu.shape[:2]
        device = data.x_imu.device

        #### DIFFUSE THE POSE ####

        # GT for sampling assembling
        smpl_6d_gt = data.smpl_6d
        trans = data.smpl_tran
        smpl_contact_gt = data.contact_p
        motion_repr_clean = torch.cat([smpl_6d_gt, trans, smpl_contact_gt], dim=-1)
        motion_repr_clean = motion_repr_clean.permute(0, 2, 1).unsqueeze(-2)  # B, feat_dim, 1, N
        tran_start_idx = smpl_6d_gt.shape[-1]

        if self.predict_velocity:
            # change the translation representation to velocity between frames (translation difference in x and z). y stays absolute
            motion_repr_clean[:, tran_start_idx, : , :-1] = motion_repr_clean[:, tran_start_idx, : , 1:] - motion_repr_clean[:, tran_start_idx, : , :-1]
            motion_repr_clean[:, tran_start_idx + 2, : , :-1] = motion_repr_clean[:, tran_start_idx + 2, : , 1:] - motion_repr_clean[:, tran_start_idx + 2, : , :-1]

        # build the conditioning per frame

        uwb_start = IMU_NUM * 3 + IMU_NUM * 9
        vuwb = data.x_imu[:, :, uwb_start:]
        if self.clean_uwb_measurement:
            vuwb = self.uwb_smoother(vuwb)
            vuwb = torch.stack(vuwb)
        x_imu = data.x_imu.clone()
        x_imu[:, :, uwb_start:] = vuwb

        shape = data.shape.unsqueeze(1).expand(-1, N, -1) if data.shape is not None else torch.zeros(B, N, 10, device=device)

        # prepare data to be autoregressive
        # This means that we create a history tensor that is the first auto_regressive_history_length
        # frames of the clean motion representation
        auto_regressive_mask = torch.rand(B, device=device) > self.auto_regressive_prob # B

        if self.add_velocity_input:
            dt = 1/60
            # integrate acc to get velocity
            acc = x_imu[:, :, :IMU_NUM * 3]
            if self.convert_acc_to_g:
                # convert the acc in g to m/s^2
                acc = acc * 9.81
            vel = torch.cumsum(acc*dt, dim=-2)
            x_imu = torch.cat([x_imu, vel], dim=-1)  # B, N, IMU_data + IMU_NUM * 3

        if self.add_position_input:
            dt = 1/60
            acc = x_imu[:, :, :IMU_NUM * 3]
            if self.convert_acc_to_g:
                # convert the acc in g to m/s^2
                acc = acc * 9.81
            vel = torch.cumsum(acc*dt, dim=-2)
            pos = torch.cumsum(vel*dt, dim=-2)

            # shift position to start at history with 0,0,0
            pos[auto_regressive_mask] = pos[auto_regressive_mask] - pos[
                auto_regressive_mask, self.auto_regressive_history_length].unsqueeze(1)
            x_imu = torch.cat([x_imu, pos], dim=-1)  # B, N, IMU_data + IMU_NUM * 3

        if self.use_mds:
            mds_points, mds_points_reflected = self.compute_mds(data.vuwb)

            fixed_mds_points = self.correct_mds(x_imu, data.shape, mds_points, mds_points_reflected)
            cond = torch.cat([x_imu, fixed_mds_points.reshape(B, N, -1), shape], dim=-1) # B, N, IMU_data + 2*IMU_NUM*3 + 10
        else:
            cond = torch.cat([x_imu, shape], dim=-1)  # B, N, IMU_data + 10


        cond = cond.view(B, N, -1).permute(0, 2, 1).unsqueeze(-2)  # B, feat_dim, 1, N



        # The autoregressive history is the shifted clean motion representation
        history = torch.zeros(*motion_repr_clean.shape[:3], self.auto_regressive_history_length, device=device, requires_grad=False) # B, feat_dim, 1, history_length
        history[auto_regressive_mask] = motion_repr_clean[auto_regressive_mask, :, :, :self.auto_regressive_history_length]

        if not self.predict_velocity:
            # subtract the last position in the translation from the history to make it end at origin, keep height
            history[auto_regressive_mask, tran_start_idx] = history[auto_regressive_mask, tran_start_idx] - history[auto_regressive_mask, tran_start_idx, :, -1][..., None]
            if not self.predict_height:
                history[auto_regressive_mask, tran_start_idx + 1] = history[auto_regressive_mask, tran_start_idx + 1] - history[auto_regressive_mask, tran_start_idx + 1, :, -1][..., None]
            history[auto_regressive_mask, tran_start_idx + 2] = history[auto_regressive_mask, tran_start_idx + 2] - history[auto_regressive_mask, tran_start_idx + 2, :, -1][..., None]


        # Need to shift the clean motion representation and conditioning to the left
        # the "new" motions when using autoregression start only after the history length
        # we only infer the motion after the history length
        motion_repr_clean[auto_regressive_mask, :, :, :-self.auto_regressive_history_length] = motion_repr_clean[auto_regressive_mask, :, :, self.auto_regressive_history_length:]
        motion_repr_clean[auto_regressive_mask, :, :, -self.auto_regressive_history_length:] = 0

        if not self.predict_velocity:
            # subtract the first position from the clean motion representation to make it start at origin
            motion_repr_clean[:, tran_start_idx] = (motion_repr_clean[:, tran_start_idx] - motion_repr_clean[:, tran_start_idx, :, 0][..., None])
            motion_repr_clean[:, tran_start_idx + 1] = (motion_repr_clean[:, tran_start_idx + 1] - motion_repr_clean[:, tran_start_idx + 1, :, 0][..., None])
            motion_repr_clean[:, tran_start_idx + 2] = (motion_repr_clean[:, tran_start_idx + 2] - motion_repr_clean[:, tran_start_idx + 2, :, 0][..., None])

        cond[auto_regressive_mask, :, :, :-self.auto_regressive_history_length] = cond[auto_regressive_mask, :, :, self.auto_regressive_history_length:]
        cond[auto_regressive_mask, :, :, -self.auto_regressive_history_length:] = 0

        # create the attention mask for the transformer to ignore the 0 padded values
        motion_mask = torch.ones(B, N, device=device)
        motion_mask[auto_regressive_mask, -self.auto_regressive_history_length:] = 0


        batch = {
            "motion_repr_clean": motion_repr_clean, # B, feat_dim, 1, N
            "cond": cond, # B, feat_dim, 1, N
            "history": history, # B, feat_dim, 1, history_length
            "motion_mask": motion_mask, # B, N
            "auto_regressive_mask": auto_regressive_mask, # B
        }
        t, weights = self.schedule_sampler.sample(B, device)
        batch, model_output = self.diffusion_train.generate_training_output(model=self.model, batch=batch, t=t,
                                                                            noise=None)

        if self.predict_no_tran:
            # set the translation to 0
            model_output[:, tran_start_idx:tran_start_idx + 3] = 0
        rt_motion_repr_clean = motion_repr_clean.clone()
        rt_model_output = model_output.clone()

        history = history.squeeze(-2).permute(0, 2, 1)  # B, N, feat_dim
        model_output = model_output.squeeze(-2).permute(0, 2, 1)  # B, N, feat_dim

        # shift the output back to the original position based on the batch that used autoregressive sampling
        model_output[auto_regressive_mask, self.auto_regressive_history_length:] = model_output[auto_regressive_mask, :-self.auto_regressive_history_length]
        model_output[auto_regressive_mask, :self.auto_regressive_history_length] = history[auto_regressive_mask]

        if self.predict_velocity:
            model_output[:, :, tran_start_idx] = torch.cumsum(model_output[:, :, tran_start_idx], dim=-1)
            model_output[:, :, tran_start_idx + 2] = torch.cumsum(model_output[:, :, tran_start_idx + 2], dim=-1)

        # CHANGE THE GROUNDTRUTH,
        if not self.predict_velocity:
            self.modify_gt(data, auto_regressive_mask, motion_mask)



        smpl_6d = model_output[:, :, :24 * 6] # B, N, 24*6
        trans = model_output[:, :, 24 * 6:24 * 6 + 3] # B, N, 3
        contact = model_output[:, :, 24 * 6 + 3:] # B, N, 2

        smpl_rot_mat = r6d_to_rotation_matrix(smpl_6d.reshape(-1, 6)).view(B, N, 24, 3, 3)
        joints_global_list = []
        for i in range(B):
            # use bodymodel
            _, joints_global = self.bodymodel.forward_kinematics(smpl_rot_mat[i],
                                                                 shape[i, 0, :],
                                                                 trans[i]
                                                                 )
            joints_global_list.append(joints_global)


        joints_global = torch.stack(joints_global_list, dim=0)

        fixed_mds_points = fixed_mds_points if self.use_mds else None
        return smpl_6d, trans, joints_global, contact, fixed_mds_points, rt_motion_repr_clean, rt_model_output, vuwb

    def modify_gt(self, data, auto_regressive_mask, motion_mask):
        """
        We need to modify the ground truth to be shifted to align with the autogressive shifted values.
        Also modify the ground truth such that the smpl trajectories always start at 0.

        :param data: the input data
        :param auto_regressive_mask: the mask that indicates which batch elements used autoregressive sampling
        :param motion_mask: the mask that indicates which frames are used in the diffusion

        :return: the modified data
        """

        # Center the smpl joint trajectories to start at 0 with the prediction window
        joints_glb = data.joints_glb # B, T, J, 3
        
        # center joints at (0, y, 0)
        joints_glb[..., 0] = joints_glb[..., 0] - joints_glb[:, 0, 0, 0][:, None, None] 
        joints_glb[..., 2] = joints_glb[..., 2] - joints_glb[:, 0, 0, 2][:, None, None]

        # prediction starts at the auto_regressive history length
        joints_glb[auto_regressive_mask, :, :, 0] = joints_glb[auto_regressive_mask, :, :, 0] - joints_glb[auto_regressive_mask, self.auto_regressive_history_length, :, 0][:, None]
        if not self.predict_height:
            joints_glb[auto_regressive_mask, :, :, 1] = joints_glb[auto_regressive_mask, :, :, 1] - joints_glb[auto_regressive_mask, self.auto_regressive_history_length, :, 1][:, None]
        joints_glb[auto_regressive_mask, :, :, 2] = joints_glb[auto_regressive_mask, :, :, 2] - joints_glb[auto_regressive_mask, self.auto_regressive_history_length, :, 2][:, None]

        # center the smpl translation
        smpl_tran = data.smpl_tran
        # the prediction starts at self.auto_regressive_history_length if autoregressive sampling is used
        smpl_tran[auto_regressive_mask, :, 0] = smpl_tran[auto_regressive_mask, :, 0] - smpl_tran[auto_regressive_mask, self.auto_regressive_history_length, 0][:, None]
        if not self.predict_height:
            smpl_tran[auto_regressive_mask, :, 1] = smpl_tran[auto_regressive_mask, :, 1] - smpl_tran[auto_regressive_mask, self.auto_regressive_history_length, 1][:, None]
        smpl_tran[auto_regressive_mask, :, 2] = smpl_tran[auto_regressive_mask, :, 2] - smpl_tran[auto_regressive_mask, self.auto_regressive_history_length, 2][:, None]
        data.smpl_tran = smpl_tran

        return data

    def _process_input_imu(self, glb_acc, glb_rot, glb_uwb):
        if glb_uwb is None:
            return torch.cat([normalize_and_concat(glb_acc, glb_rot)], dim=1)
        return torch.cat([normalize_and_concat(glb_acc, glb_rot), glb_uwb.flatten(1)], dim=1)

    # @profile
    @torch.no_grad()
    def predict(self, glb_acc, glb_rot, init_pose, **kwargs):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :param glb_uwb: A tensor that can reshape to [num_frames, 6, 6].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """

        ONESHOT = False
        GUIDANCE = True
        if "vuwb" in self.imu_m:
            vuwb = kwargs["glb_uwb"]
            offset = kwargs["offset"]
        else:
            vuwb = None
            offset = torch.zeros(6, 3, device=glb_acc.device)

        if "shape" in kwargs:
            smpl_shape = kwargs["shape"]
        else:
            print("No shape provided, using default shape")
            smpl_shape = torch.zeros(10, device=glb_acc.device)

        sample_loop = self.diffusion_train.ddim_sample_loop if "ddim" in self.diffusion_type_eval else self.diffusion_train.p_sample_loop

        # Move device
        glb_acc = glb_acc.to(self.device)
        glb_rot = glb_rot.to(self.device)
        if vuwb is not None:
            vuwb = vuwb.to(self.device)
        smpl_shape = smpl_shape.to(self.device)
        offset = offset.to(self.device)
        seq_len = glb_acc.shape[0]

        # convert acc to g's
        if self.convert_acc_to_g:
            glb_acc = glb_acc / 9.81

        x_imu = self._process_input_imu(glb_acc, glb_rot, vuwb).unsqueeze(0) # 1, N, 108


        if self.clean_uwb_measurement and self.use_uwb:
            uwb_start_idx = IMU_NUM * 9 + IMU_NUM * 3
            v_uwb = x_imu[:, :, uwb_start_idx:]
            vuwb_pred = self.uwb_smoother(v_uwb)
            vuwb_pred = torch.stack(vuwb_pred)
            x_imu[:, :, uwb_start_idx:] = vuwb_pred

        x_imu_initial_window = x_imu[:, :self.predict_chunk_length] # 1, inital_window_length, 108

        if self.add_velocity_input:
            # integrate acc to get velocity
            dt = 1 / 60  # time step in seconds
            acc = x_imu_initial_window[:, :, :IMU_NUM * 3]
            if self.convert_acc_to_g:
                # convert the acc in g to m/s^2
                acc = acc * 9.81
            # Properly scale by dt when integrating
            vel = torch.cumsum(acc * dt, dim=-2)
            x_imu_initial_window = torch.cat([x_imu_initial_window, vel], dim=-1)  # B, N, IMU_data + IMU_NUM * 3

        if self.add_position_input:
            dt = 1 / 60  # time step in seconds
            acc = x_imu_initial_window[:, :, :IMU_NUM * 3]
            if self.convert_acc_to_g:
                # convert the acc in g to m/s^2
                acc = acc * 9.81
            # Properly scale by dt for both integrations
            vel = torch.cumsum(acc * dt, dim=-2)
            pos = torch.cumsum(vel * dt, dim=-2)

            x_imu_initial_window = torch.cat([x_imu_initial_window, pos], dim=-1)  # B, N, IMU_data + IMU_NUM * 3


        inital_window_length = x_imu_initial_window.shape[1]
        smpl_shape_ext = smpl_shape[None, None, :].expand(-1, inital_window_length, -1) # 1, N, SMPL_SHAPE_DIM

        if self.use_mds:
            vuwb = vuwb.view(1, -1, IMU_NUM * IMU_NUM)  # 1, N, 36

            mds_points, mds_points_reflected = self.compute_mds(vuwb)

            fixed_mds_points_initial_window = self.correct_mds(x_imu_initial_window, smpl_shape.unsqueeze(0), mds_points[:, :inital_window_length], mds_points_reflected[:, :inital_window_length]) # 1, inital_window_length, 3, 2 * IMU_NUM

            cond = torch.cat([x_imu_initial_window, fixed_mds_points_initial_window.reshape(1, inital_window_length, -1), smpl_shape_ext], dim=-1) # B=1, N, IMU_data + 2*IMU_NUM*3 + 10
        else:
            cond = torch.cat([x_imu_initial_window, smpl_shape_ext], dim=-1)  # B=1, N, IMU_data + 10

        cond = cond.view(1, inital_window_length, -1).permute(0, 2, 1).unsqueeze(-2)  # B, feat_dim, 1, N


        batch = {
            "cond": cond, # B, feat_dim, 1, N
            "history": torch.zeros(1, self.clean_motion_dim, 1, inital_window_length, device=self.device),  # B, feat_dim, 1, history_length
            "motion_mask": torch.ones(1, inital_window_length, dtype=torch.bool, device=self.device), # B, N
            "auto_regressive_mask": torch.zeros(1, dtype=torch.bool, device=self.device), # B
        }
        if ONESHOT:
            batch["motion_repr_clean"] = torch.zeros(1, self.clean_motion_dim, 1, self.predict_chunk_length, device=self.device) # B, feat_dim, 1, N
            batch, model_output = self.diffusion_train.generate_training_output(model=self.model, batch=batch,
                                                                                t=torch.tensor([self.diffusion_steps-1], device=self.device),
                                                                                noise=None)
        else:
            model_output = sample_loop(model=self.model,
                                              batch=batch,
                                              shape=(1, self.clean_motion_dim, 1, inital_window_length),
                                              progress=False,
                                              clip_denoised=False,
                                              cond_fn_with_grad=GUIDANCE,
                                              grad_type="amass",
                                              early_stop=False,
                                              save_intermediate_result=False) # B, F, 1, N

        model_outputs = [model_output]
        times = []
        idx = 1
        for start_idx in range(self.predict_chunk_length, glb_acc.shape[0], self.predict_chunk_length):
            end_idx = min(start_idx + self.predict_chunk_length, glb_acc.shape[0])
            window_length = end_idx - start_idx

            x_imu_window = x_imu[:, start_idx:end_idx] # B, N, F

            if self.add_velocity_input:
                # integrate acc to get velocity
                dt = 1 / 60
                acc = x_imu_window[:, :, :IMU_NUM * 3]
                if self.convert_acc_to_g:
                    # convert the acc in g to m/s^2
                    acc = acc * 9.81
                vel = torch.cumsum(acc * dt, dim=-2)
                x_imu_window = torch.cat([x_imu_window, vel], dim=-1)  # B, N, IMU_data + IMU_NUM * 3

            if self.add_position_input:
                dt = 1 / 60
                acc = x_imu_window[:, :, :IMU_NUM * 3]
                if self.convert_acc_to_g:
                    # convert the acc in g to m/s^2
                    acc = acc * 9.81
                vel = torch.cumsum(acc * dt, dim=-2)
                pos = torch.cumsum(vel * dt, dim=-2)

                x_imu_window = torch.cat([x_imu_window, pos], dim=-1)

            smpl_shape_exp = smpl_shape[None, None, :].expand(-1, window_length, -1) # B, N, SMPL_SHAPE_DIM
            if self.use_mds:
                fixed_mds_points_window = self.correct_mds(x_imu_window, smpl_shape.unsqueeze(0),
                                                                   mds_points[:, start_idx:end_idx],
                                                                   mds_points_reflected[:,
                                                                   start_idx:end_idx])  # 1, inital_window_length, 3, 2 * IMU_NUM

                cond = torch.cat([x_imu_window, fixed_mds_points_window.reshape(1, window_length, -1), smpl_shape_exp], dim=-1) # B, N, IMU_data + 2*IMU_NUM*3 + 10
            else:
                cond = torch.cat([x_imu_window, smpl_shape_exp], dim=-1)  # B, N, IMU_data + 10

            cond = cond.view(1, window_length, -1).permute(0, 2, 1).unsqueeze(-2)  # B, feat_dim, 1, N

            history = model_outputs[-1][0, :, :, -self.auto_regressive_history_length:].unsqueeze(0).clone() # 1, F, 1, inital_window_length
            # subtract the last position in the translation from the history to make it end at origin
            tran_start_idx = 24 * 6
            if not self.predict_velocity:
                history[:, tran_start_idx] = (history[:, tran_start_idx] - history[:, tran_start_idx, :, -1][..., None])
                if not self.predict_height:
                    history[:, tran_start_idx + 1] = (history[:, tran_start_idx + 1] - history[:, tran_start_idx + 1, :, -1][..., None])
                history[:, tran_start_idx + 2] = (history[:, tran_start_idx + 2] - history[:, tran_start_idx + 2, :, -1][..., None])


            motion_mask = torch.ones(1, x_imu_window.shape[1], device=glb_acc.device, dtype=torch.bool)
            auto_regressive_mask = torch.ones(1, device=glb_acc.device, dtype=torch.bool)

            batch = {
                "cond": cond, # B=1, feat_dim, 1, N
                "history": history,  # B, feat_dim, 1, history_length
                "motion_mask": motion_mask,  # B=1, N
                "auto_regressive_mask": auto_regressive_mask, # B=1
            }

            if ONESHOT:
                batch["motion_repr_clean"] =  torch.zeros(1, self.clean_motion_dim, 1, window_length,
                                                 device=self.device)  # B, feat_dim, 1, N
                batch, model_output = self.diffusion_train.generate_training_output(model=self.model, batch=batch,
                                                                                    t=torch.tensor([self.diffusion_steps-1],
                                                                                                   device=self.device),
                                                                                    noise=None)

            else:
                model_output = sample_loop(model=self.model,
                                                  batch=batch,
                                                  shape=(1, self.clean_motion_dim, 1, window_length),
                                                  progress=False,
                                                  clip_denoised=False,
                                                  cond_fn_with_grad=GUIDANCE,
                                                  grad_type="amass",
                                                  early_stop=False,
                                                  save_intermediate_result=False)
            idx += 1

            # add back the previous last postion to the model output translation
            if not self.predict_velocity:
                model_output[:, tran_start_idx] = model_output[:, tran_start_idx] + model_outputs[-1][:, tran_start_idx, :, -1][..., None]
                if not self.predict_height:
                    model_output[:, tran_start_idx + 1] = model_output[:, tran_start_idx + 1] + model_outputs[-1][:, tran_start_idx + 1, :, -1][..., None]
                model_output[:, tran_start_idx + 2] = model_output[:, tran_start_idx + 2] + model_outputs[-1][:, tran_start_idx + 2, :, -1][..., None]


            model_outputs.append(model_output)


        model_outputs = torch.cat(model_outputs, dim=-1)
        model_outputs = model_outputs.squeeze(-2).permute(0, 2, 1) # B, N, feat_dim
        smpl_6d = model_outputs[..., :24 * 6] # B, N, 24*6
        trans = model_outputs[..., 24 * 6:24 * 6 + 3] # B, N, 3
        contact = model_outputs[..., 24 * 6 + 3:] # B, N, 2

        smpl_6d = smpl_6d.squeeze(0)
        trans = trans.squeeze(0)
        contact = contact.squeeze(0)

        if self.predict_velocity:
            # integrate velocity to translation
            trans[:, 0] = torch.cumsum(trans[:, 0], dim=0)
            trans[:, 2] = torch.cumsum(trans[:, 2], dim=0)

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

        # smoothing filter smpl pose and translation
        smpl_6d = smooth_tensor(smpl_6d, sigma=2)
        trans = smooth_tensor(trans, sigma=2)


        smpl_rot_mat = r6d_to_rotation_matrix(smpl_6d.reshape(-1, 6)).view(seq_len, 24, 3, 3)
        _, joints_global = self.bodymodel.forward_kinematics(smpl_rot_mat, smpl_shape, trans)

        smpl_rot_mat = r6d_to_rotation_matrix(smpl_6d.reshape(-1, 6)).reshape(smpl_6d.size(0), -1, 3, 3)
        return smpl_rot_mat, trans, joints_global, contact



    def visualize_intermediate_result(self, x0, xt, t, titel:str = ""):
        """
        Visualize the intermediate results of the diffusion sampling.

        x0: (pred_xstart) is the prediction for x_0.
        """
        x0 = x0.squeeze(-2).permute(1, 0)  # a prediction of the model for timestamp 0
        xt = xt.squeeze(-2).permute(1, 0) # a sample from the model at timestamp t

        smpl_x0_6d = x0[:, :24 * 6]
        smpl_xt_6d = xt[:, :24 * 6]
        smpl_x0_tran = x0[:, 24 * 6:24 * 6 + 3]
        smpl_xt_tran = xt[:, 24 * 6:24 * 6 + 3]

        from aitviewer.configuration import CONFIG as C
        from aitviewer.models.smpl import SMPLLayer
        from aitviewer.headless import HeadlessRenderer

        C.update_conf({"smplx_models": os.getenv("SMPL_MODEL_PATH", "data/smpl_m_lbs_10_207_0_v1.0.0.pkl"),
                       "run_animations": True,
                       "scene_fps": 60,
                       "playback_fps": 60,
                       "device": str(self.device)
                       })

        smpl_layer = SMPLLayer(model_type="smpl", gender="male")
        if not hasattr(self, "renderer"):
            self.renderer = HeadlessRenderer()
        log_video(self.renderer, smpl_layer, smpl_xt_6d, smpl_x0_6d, torch.zeros(10, device=smpl_xt_tran.device),
                    smpl_xt_tran, smpl_x0_tran, prefix="test", video_name=f"{titel}_predict_{t}", log_video_to_wandb=False)
