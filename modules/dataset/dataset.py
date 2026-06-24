import gc
import os
import random
from collections import defaultdict
from typing import Literal

from torch.utils.data import Dataset
from tqdm import tqdm

from articulate.math import *
from articulate.model import ParametricModel
from config.config import *
from modules.dataset.data_utils import *


class _Dataset(Dataset):
    def __init__(self,
                 official_model_file: str,
                 rep=RotationRepresentation.AXIS_ANGLE,
                 device=torch.device('cpu'),
                 seq_length=200,
                 down_sample_rate=60,
                 use_cached=True,
                 add_uwb=False,
                 train_split=True,
                 imu_m=[""],
                 static_uwb_noise=-1,
                 imu_acc_noise=0.0,
                 imu_ori_noise=0.0,
                 imu_ori_bias_noise=0.0,
                 extreme_value_thresh_g=np.nan,
                 normalize_uwb=False,
                 predict_height=False,
                 convert_acc_to_g=False,
                 uwb_timesample_ratio=None,
                 flatten_uwb=False,
                 dry_run=False,
                 remove_node=-1,
                 **kwargs) -> None:
        super().__init__()
        self.model = ParametricModel(official_model_file, use_pose_blendshape=False, device=device)
        self.seq_length = seq_length
        self.down_sample_rate = down_sample_rate
        self.rep = rep
        self.use_cached = use_cached
        self.device = device
        self.add_uwb = add_uwb
        self.normalize_uwb = normalize_uwb
        self.convert_acc_to_g = convert_acc_to_g
        self.predict_height = predict_height
        self.train_split = train_split
        self.imu_m = imu_m
        self.static_uwb_noise = static_uwb_noise
        self.imu_acc_noise = imu_acc_noise
        self.imu_ori_noise = imu_ori_noise
        self.imu_ori_bias_noise = imu_ori_bias_noise
        self.extreme_value_thresh_g = extreme_value_thresh_g
        self.with_acc_sum = "acc_sum" in imu_m
        self.uwb_timesample_ratio = uwb_timesample_ratio
        self.flatten_uwb = flatten_uwb
        self.remove_node = remove_node
        self.dry_run = dry_run


    def _preprocess(self, pose, shape=None, tran=None):
        pose = to_rotation_matrix(pose.to(self.device), self.rep).view(pose.shape[0], -1)
        shape = shape.to(self.device) if shape is not None else shape
        tran = tran.to(self.device) if tran is not None else tran
        return pose, shape, tran

    def __len__(self):
        return self.seq_num

    def _get_subsample_seq_range(self, N_frames):
        frame_range = range(self.seq_length, N_frames)
        num_samples = np.maximum(round(len(frame_range) / self.down_sample_rate), 1)
        for t_end in random.sample(frame_range, k=num_samples):
            yield t_end

    def _check_cached(self, dir, file):
        file_path = os.path.join(dir, file)
        if os.path.exists(file_path) and self.use_cached:
            print("Use cached dataset from ", file_path)
            return True
        return False

    @staticmethod
    def add_syn_uwb_noise(uwb, sigma=-1, uwb_collision=None, normalize=False):
        # this method is used during sampling it expects B, 6, 6
        convert_format = False
        if len(uwb.shape) == 2:
            convert_format = True
            uwb = uwb.view(-1, IMU_NUM, IMU_NUM)

        if sigma >= 0:
            # Map the sensors in use to their indices in the global sensor order.
            used_indices = [IMU_placement.SENSOR_ORDER.index(sensor) for sensor in IMU_USE]
            # Extract the corresponding noise level submatrix.
            noise_level_used = UIP_UWB_NOISE_LEVEL[used_indices][:, used_indices].to(uwb.device)

            # Scale the noise level submatrix by sigma and expand it to the batch dimension.
            noise_std = sigma * noise_level_used  # shape (IMU_NUM, IMU_NUM)
            noise_std = noise_std.unsqueeze(0).expand(uwb.size(0), -1, -1)
            noise_full = torch.normal(mean=torch.zeros_like(noise_std), std=noise_std)

            # Create a mask for the upper-triangular part (excluding the diagonal).
            mask_upper = torch.triu(torch.ones(uwb.size(1), uwb.size(2), dtype=torch.bool, device=uwb.device),
                                    diagonal=1)
            # Initialize an empty symmetric noise tensor.
            noise_sym = torch.zeros_like(noise_full)
            # Fill only the upper triangular part with the sampled noise.
            noise_sym[:, mask_upper] = noise_full[:, mask_upper]
            # Mirror the upper triangle to the lower triangle.
            noise_sym = noise_sym + noise_sym.transpose(1, 2)
            noise = noise_sym
        else:
            # Collision branch ignored.
            noise = torch.zeros_like(uwb)

        # Symmetrize the noise and zero-out the diagonal.
        noise = (noise + noise.transpose(1, 2)) / np.sqrt(2)
        noise = torch.clamp(noise, min=0.00)

        torch.diagonal(noise, dim1=1, dim2=2).zero_()
        # clip noise at 0

        if normalize:
            # Normalize by the distance between head and root.
            # First, determine the indices for head and root in the used sensor list.
            try:
                head_idx = IMU_USE.index("head")
                root_idx = IMU_USE.index("root")
                d_root2head = uwb[0, head_idx, root_idx].clone()  # using first sample
            except ValueError:
                # If either 'head' or 'root' is not used, skip normalization.
                d_root2head = 1.0
            uwb = (uwb + noise) / d_root2head
        else:
            uwb = uwb + noise

        if convert_format:
            return uwb.view(-1, IMU_NUM * IMU_NUM)
        return uwb

    @staticmethod
    def preprocess(pose, shape=None, tran=None, device="cpu", rep=RotationRepresentation.AXIS_ANGLE):
        pose = to_rotation_matrix(pose.to(device), rep).view(pose.shape[0], -1)
        shape = shape.to(device) if shape is not None else shape
        tran = tran.to(device) if tran is not None else tran
        return pose, shape, tran

    @staticmethod
    def get_non_root_joint_pos(glb_joint_pos, glb_root_ori):
        _r = glb_joint_pos[:, 1:, :] - glb_joint_pos[:, joint_set.root, :]
        return torch.einsum('bij,bjk->bik', _r, glb_root_ori.view(-1, 3, 3))

    @staticmethod
    def get_leaf_joint(glb_joint_pos, glb_root_ori):
        # get leaf joint position relative to root
        _r = (glb_joint_pos[:, joint_set.leaf, :] - glb_joint_pos[:, joint_set.root])
        return torch.einsum('bij,bjk->bik', _r, glb_root_ori.view(-1, 3, 3))

    @staticmethod
    def get_6d_rotation(global_rotmat, glb_root_ori):
        # get 6d rotation relative to root
        N_f = global_rotmat.size(0)
        global_rotmat = torch.einsum('abij,abjk->abik', glb_root_ori.view(-1, 1, 3, 3).transpose(2, 3), global_rotmat)
        return global_rotmat.view(N_f, -1, 3, 3)[:, :, :, :2].transpose(2, 3).contiguous().view(N_f, -1, 6)

    @staticmethod
    def get_joint_vel(glb_joint_pos, glb_root_ori):
        N_f = glb_joint_pos.size(0)
        joint_vel = torch.zeros_like(glb_joint_pos)
        joint_vel[1:] = (glb_joint_pos[1:, :, :] - glb_joint_pos[:-1, :, :]) * 20
        joint_vel = torch.einsum('bij,bjk->bik', joint_vel, glb_root_ori.view(N_f, 3, 3))
        joint_vel[0] = joint_vel[1]
        return joint_vel

    @staticmethod
    def get_contact_points(glb_joint_pos, th=0.0125):
        feet_contact = torch.zeros(glb_joint_pos.size(0), len(joint_set.foot_indices))
        feet_contact[1:, :] = torch.linalg.norm(glb_joint_pos[1:, joint_set.foot_indices, :] - glb_joint_pos[:-1, joint_set.foot_indices, :], dim=2) < th
        feet_contact[0] = feet_contact[1]
        feet_contact = feet_contact.to(torch.int)
        return feet_contact

    @staticmethod
    def get_acc_sum(acc, window_size=40, down_scale=15):
        b = torch.cumsum(acc.view(-1, 18), dim=0)
        b[window_size:, :] = b[window_size:, :] - b[:-window_size, :]
        b = b / down_scale  # down scale to acc scale
        return b.view_as(acc)

    @staticmethod
    def get_local_imu(glb_acc, glb_rot):
        glb_acc = glb_acc.view(-1, 6, 3)
        glb_rot = glb_rot.view(-1, 6, 3, 3)
        acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
        ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
        return acc, ori

    def _data_sampling(self, seq_info):
        """
        Subsample data sequence of different length into chunks of the same length(self.seq_length) for training.

        Args:
            seq_info seq_id:(leaf_joint,joint_vel,pose_global_6d,feet_contact,non_root_joint,imu_ori,imu_acc,*res)

        Returns:
            data_info,data_idx
        """
        seq_list = sorted(list(seq_info.keys()))  # list of seq_ids

        data_info = defaultdict(list)
        data_info["vuwb"] = []
        data_info["uwb_gt"] = []
        data_info["offset"] = []
        data_info["acc_sum"] = []

        data_idx = 0
        print(f"Sampling Dataset {self.__class__.__name__}")
        for seq_id in tqdm(seq_list):

            seq_d = seq_info[seq_id]
            assert isinstance(seq_d, SeqInfo)
            for t_end in self._get_subsample_seq_range(N_frames=seq_d.frame_size):
                assert t_end >= self.seq_length
                data_info["joints_glb"].append(seq_d.joints_glb[t_end - self.seq_length: t_end].to(device=self.device))  # N,24,3
                data_info["lfj"].append(
                    seq_d.leaf_joint[t_end - self.seq_length: t_end].to(device=self.device))  # N,5,3
                data_info["jv"].append(seq_d.joint_vel[t_end - self.seq_length: t_end].to(device=self.device))  # N,J,3
                data_info["smpl_aa"].append(seq_d.smpl_aa[t_end - self.seq_length: t_end].to(device=self.device))
                data_info["smpl_6d"].append(seq_d.smpl_6d[t_end - self.seq_length: t_end].to(device=self.device))
                data_info["shape"].append(seq_d.shape.to(device=self.device))
                # start pos with height, y is up
                if self.predict_height:
                    # dont normalize the height
                    start_pos = torch.tensor([seq_d.smpl_tran[t_end - self.seq_length, 0], 0, seq_d.smpl_tran[t_end - self.seq_length, 2]]).to(device=self.device)
                else:
                    # start pos, completely normalized
                    start_pos = seq_d.smpl_tran[t_end - self.seq_length, :3].to(device=self.device)
                data_info["smpl_tran"].append(seq_d.smpl_tran[t_end - self.seq_length: t_end].to(device=self.device) - start_pos) 
                data_info["jrot_6d"].append(
                    seq_d.pose_global_6d[t_end - self.seq_length: t_end, joint_set.reduced, :].to(
                        device=self.device))  # N,J,6
                data_info["jrot_6d_full"].append(
                    seq_d.pose_global_6d[t_end - self.seq_length:t_end].to(
                        device=self.device))
                data_info["contact"].append(
                    seq_d.feet_contact[t_end - self.seq_length: t_end].to(device=self.device))  # N,2
                data_info["joint"].append(
                    seq_d.non_root_joint[t_end - self.seq_length: t_end, :, :].to(device=self.device))
                data_info["vrot"].append(seq_d.imu_ori[t_end - self.seq_length: t_end].to(device=self.device))
                data_info["vacc"].append(seq_d.imu_acc[t_end - self.seq_length: t_end].to(device=self.device))
                if seq_d.with_uwb():
                    data_info["vuwb"].append(seq_d.uwb_m[t_end - self.seq_length: t_end].to(device=self.device))
                    data_info["uwb_gt"].append(seq_d.uwb_gt[t_end - self.seq_length: t_end].to(device=self.device))
                    data_info["offset"].append(seq_d.offset.to(device=self.device))
                if seq_d.with_acc_sum():
                    data_info["acc_sum"].append(seq_d.acc_sum[t_end - self.seq_length: t_end].to(device=self.device))
                data_idx += 1
        return data_info, data_idx

    def resubsampling(self):
        """
        Resubsampling the data sequence with the cached preprocessed data

        Returns:
            None
        """

        data_info, data_idx = self._data_sampling(self.seq_info)

        self.seq_num = data_idx
        self.data_info = data_info

    def add_ori_noise(self, rot_mats: torch.tensor):
        # Rot_mats: T, NUM_IMU*9
        # apply random rotation noise
        # with std noise = self.imu_ori_noise
        # and a random bias with std self.imu_ori_bias_noise

        T = rot_mats.size(0)
        num_imu = rot_mats.size(1) // 9
        # Reshape to (T, num_imu, 3, 3)
        rot_mats = rot_mats.view(T, num_imu, 3, 3)

        def rodrigues(rotvec: torch.Tensor) -> torch.Tensor:
            """
            Convert a rotation vector to a rotation matrix using the Rodrigues formula.

            Args:
                rotvec (torch.Tensor): Tensor of shape (..., 3) representing rotation vectors.

            Returns:
                torch.Tensor: Rotation matrices of shape (..., 3, 3).
            """
            eps = 1e-6
            theta = torch.norm(rotvec, dim=-1, keepdim=True)  # shape (..., 1)
            # Avoid division by zero: normalize the rotation vector safely.
            axis = rotvec / (theta + eps)

            # Create cross-product matrix K for each rotation vector.
            zero = torch.zeros_like(axis[..., 0])
            K = torch.stack([
                torch.stack([zero, -axis[..., 2], axis[..., 1]], dim=-1),
                torch.stack([axis[..., 2], zero, -axis[..., 0]], dim=-1),
                torch.stack([-axis[..., 1], axis[..., 0], zero], dim=-1)
            ], dim=-2)  # shape (..., 3, 3)

            # Compute sin and cos terms and reshape for proper broadcasting.
            sin_theta = torch.sin(theta)[..., None]  # now shape (..., 1, 1)
            cos_theta = torch.cos(theta)[..., None]  # now shape (..., 1, 1)

            # Identity matrix, broadcasted to match K's shape.
            I = torch.eye(3, device=rotvec.device).expand(K.shape[:-2] + (3, 3))

            # Rodrigues formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
            K2 = torch.matmul(K, K)
            R_mat = I + sin_theta * K + (1 - cos_theta) * K2
            return R_mat

        # Generate constant bias rotation per sensor.
        bias_rotvec = torch.randn(num_imu, 3, device=rot_mats.device) * self.imu_ori_bias_noise
        bias_rot_mats = rodrigues(bias_rotvec)  # shape: (num_imu, 3, 3)

        # Generate noise rotation per time step and sensor.
        noise_rotvec = torch.randn(T, num_imu, 3, device=rot_mats.device) * self.imu_ori_noise
        noise_rot_mats = rodrigues(noise_rotvec)  # shape: (T, num_imu, 3, 3)

        # Combine bias and noise rotations.
        # bias_rot_mats: (num_imu, 3, 3) -> unsqueeze to (1, num_imu, 3, 3) to broadcast over T.
        combined_error = torch.matmul(bias_rot_mats.unsqueeze(0), noise_rot_mats)  # (T, num_imu, 3, 3)

        # Apply the error rotation: new_rot = original_rot * combined_error.
        new_rot_mats = torch.matmul(rot_mats, combined_error)  # (T, num_imu, 3, 3)

        # Flatten back to (T, num_imu*9)
        return new_rot_mats.view(T, num_imu * 9)

    def add_acc_noise(self, data):
        return data + torch.normal(0, std=self.imu_acc_noise, size=data.size()).to(data.device)

    def apply_noise(self, data, type: Literal["vacc", "vrot", "vuwb"]):
        if not self.train_split:
            return data
        if type == "vacc":
            return self.add_acc_noise(data)
        elif type == "vrot":
            return self.add_ori_noise(data)
        elif type == "vuwb":
            return self.add_syn_uwb_noise(data, sigma=self.static_uwb_noise, uwb_collision=None,
                                         normalize=self.normalize_uwb)


    def __getitem__(self, index):
        """_summary_

        Args:
            index (_type_): _description_

        Returns:
            X: IMU, lj_init, jvel_init (tensor [num_frames, 72], tensor [15], tensor [72])
            IMU: imu_acc (3 channel acc, 9=3x3 orientation rot mat, + optional 6 pairwise distances uwb)

            Y: Joint_position, Joint_velocity, Joint_leaf_position
            ([num_frames, 24,3]) ([num_frames, 24,3]) ([num_frames, 5,3])
        """
        kwargs_data = {}
        sensor_measure = []

        if self.imu_m:
            for k in self.imu_m:
                sensor_data = self.data_info[k][index].flatten(1).clone()
                if k == "vacc":
                    # convert to g
                    if self.convert_acc_to_g:
                        sensor_data = sensor_data / 9.81
                    if not np.isnan(self.extreme_value_thresh_g):
                        sensor_data = torch.clamp(sensor_data, min=-self.extreme_value_thresh_g, max=self.extreme_value_thresh_g)
                noisy_data = self.apply_noise(sensor_data, k)
                sensor_measure.append(noisy_data)

        kwargs_data["joints_glb"] = self.data_info["joints_glb"][index]
        kwargs_data["x_imu"] = torch.cat(sensor_measure, dim=1)
        kwargs_data["lj_init"] = self.data_info["lfj"][index][0].view(-1)
        kwargs_data["jvel_init"] = self.data_info["jv"][index][0].view(-1)

        # Make it adjustable given the training phase for it
        kwargs_data["lfj_gt"] = self.data_info["lfj"][index].view(self.seq_length, 15)
        kwargs_data["joint_gt"] = self.data_info["joint"][index].view(self.seq_length, 69)
        kwargs_data["jrot_6d"] = self.data_info["jrot_6d"][index].view(self.seq_length, 90)
        kwargs_data["jrot_6d_full"] = self.data_info["jrot_6d_full"][index]
        kwargs_data["jvel_gt"] = self.data_info["jv"][index].view(self.seq_length, 72)
        kwargs_data["contact_p"] = self.data_info["contact"][index].view(self.seq_length, len(joint_set.foot_indices))
        kwargs_data["smpl_aa"] = self.data_info["smpl_aa"][index].view(self.seq_length, 72)
        kwargs_data["smpl_6d"] = self.data_info["smpl_6d"][index].view(self.seq_length, 144)
        kwargs_data["shape"] = self.data_info["shape"][index]
        kwargs_data["smpl_tran"] = self.data_info["smpl_tran"][index].view(self.seq_length, 3)

        if "uwb_gt" in self.data_info:
            kwargs_data["uwb_gt"] = self.data_info["uwb_gt"][index].view(self.seq_length, -1)
            vuwb = self.data_info["vuwb"][index].view(self.seq_length, -1).clone()
            kwargs_data["vuwb"] = vuwb
            kwargs_data["uwb_offset"] = self.data_info["offset"][index].view(6, 3)

        return kwargs_data


'''
AMASS Dataset
'''


class AMASS_syn_data(_Dataset):
    def __init__(self, amass_dir: str, **superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = amass_dir

        self.seq_info = self._data_preprocess(self.dataset_dir)

        data_info, data_idx = self._data_sampling(self.seq_info)

        self.seq_num = data_idx

        self.data_info = data_info

    def _data_preprocess(self, amass_dir):

        seq_info = {}

        keys = ["pose", "shape", "tran", "joint", "vrot", "vacc", "vuwb", "uwb_collision", "offset"]

        raw_data = {
            k: torch.load(os.path.join(amass_dir, f"{k}.pt"))
            for k in keys if os.path.exists(os.path.join(amass_dir, f"{k}.pt"))
        }

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            joint_pos = raw_data["joint"][seq_id]
            N_frame = joint_pos.size(0)

            if N_frame <= self.seq_length:
                continue

            seq_kwargs = {}
            seq_kwargs["imu_acc"], seq_kwargs["imu_ori"] = \
                self.get_local_imu(self.add_syn_noise(raw_data["vacc"][seq_id]), raw_data["vrot"][seq_id])

            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                       raw_data["shape"][seq_id],
                                                       raw_data["tran"][seq_id])


            # Compute pose/joint in world coordinate
            pose_global_p, _ = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)

            tran_p_origin = tran_p - tran_p[0]
            _, joint_glb = self.model.forward_kinematics(pose_p, shape_p, tran_p_origin, calc_mesh=False)

            glb_root_ori = raw_data["vrot"][seq_id][:, [-1], ...]

            seq_kwargs["smpl_aa"] = raw_data["pose"][seq_id]
            seq_kwargs["smpl_6d"] = rotation_matrix_to_r6d(axis_angle_to_rotation_matrix(raw_data["pose"][seq_id].view(-1, 3))).view(N_frame, -1, 6)
            seq_kwargs["shape"] = raw_data["shape"][seq_id]
            seq_kwargs["smpl_tran"] = raw_data["tran"][seq_id]

            seq_kwargs["joints_glb"] = joint_pos

            # get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_pos, glb_root_ori)

            # non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_pos, glb_root_ori)

            # get 6D joint rotation
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)

            # get joint velocity
            seq_kwargs["joint_vel"] = self.get_joint_vel(joint_pos, glb_root_ori)

            # contact points
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_pos)

            # uwb
            uwb_c = raw_data["uwb_collision"][seq_id] if "uwb_collision" in raw_data else torch.zeros_like(
                raw_data["vuwb"][seq_id])
            uwb = self.add_syn_uwb_noise(raw_data["vuwb"][seq_id], sigma=-1.0, uwb_collision=uwb_c,
                                         normalize=self.normalize_uwb)
            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:, self.remove_node, :] = 0
                uwb[:, :, self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb
            seq_kwargs["uwb_gt"] = raw_data["vuwb"][seq_id]
            seq_kwargs["uwb_c"] = uwb_c
            seq_kwargs["offset"] = raw_data["offset"][seq_id]
            # acceleration sum
            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])

            seq_info_ = SeqInfo(seq_id=seq_id, frame_size=N_frame, **seq_kwargs)
            if self.uwb_timesample_ratio is not None and self.uwb_timesample_ratio != 1:
                seq_info_.syc_measurement(sampling_ratio=self.uwb_timesample_ratio)

            if self.flatten_uwb:
                seq_info_.flatten_uwb_measurement()

            seq_info[seq_id] = seq_info_

        if not self.dry_run:
            torch.save(seq_info, os.path.join(amass_dir, "dataset_cached.pt"))

        return seq_info

    def add_syn_noise(self, acc, bias_noise=0.1):
        noise = 2 * torch.rand_like(acc) * bias_noise - bias_noise
        return acc + noise

    def add_syn_gaussian_noise(self, acc, std=0.1, bias_noise=0.1):
        noise = torch.normal(bias_noise, std=std, size=acc.size()).to(acc.device)
        return acc + noise

'''
AMASS Val Data
'''


class AMASS_syn_data_val(AMASS_syn_data):
    def __init__(self, amass_dir: str, **superkwargs) -> None:
        amass_dir = os.path.join(amass_dir, "test_split")
        super().__init__(amass_dir, **superkwargs)


class AMASS_syn_data_train_tc(AMASS_syn_data):
    def __init__(self, amass_dir: str, **superkwargs) -> None:
        amass_dir = os.path.join(amass_dir, "syn_tc_1.0")
        super().__init__(amass_dir, **superkwargs)


class UWBIMU_real_data(_Dataset):
    def __init__(self, data_dir: str, split, **superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = data_dir

        seq_info = self._data_preprocess(self.dataset_dir, split=split)

        data_info, data_idx = self._data_sampling(seq_info)

        self.seq_num = data_idx
        self.data_info = data_info
        self.seq_info = seq_info
        print(f"Loading data size of {self.seq_num}..")

    def _data_preprocess(self, uwb_dir, split="train"):

        seq_info = {}

        print("Load raw data from ", os.path.join(uwb_dir, f"{split}.pt"))
        raw_data = torch.load(os.path.join(uwb_dir, f"{split}.pt"))

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            joint_pos = raw_data["joint"][seq_id]
            N_frame = joint_pos.size(0)

            if N_frame <= self.seq_length:
                continue

            seq_kwargs = {}
            seq_kwargs["imu_acc"], seq_kwargs["imu_ori"] = \
                self.get_local_imu(raw_data["acc"][seq_id], raw_data["ori"][seq_id])
            # Forward kinematics
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                       shape=None,
                                                       tran=raw_data["tran"][
                                                           seq_id])  # DIP IMU does not contain translations
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)
            tran_p_origin = tran_p - tran_p[0]
            _, joint_glb = self.model.forward_kinematics(pose_p, shape_p, tran_p_origin, calc_mesh=False)


            seq_kwargs["joints_glb"] = joint_pos


            glb_root_ori = raw_data["ori"][seq_id][:, [-1], ...]

            seq_kwargs["smpl_aa"] = raw_data["pose"][seq_id]
            seq_kwargs["smpl_6d"] = rotation_matrix_to_r6d(axis_angle_to_rotation_matrix(raw_data["pose"][seq_id].view(-1, 3))).view(N_frame, -1, 6)
            seq_kwargs["shape"] = raw_data["shape"][seq_id] if "shape" in raw_data else torch.zeros(10)
            seq_kwargs["smpl_tran"] = raw_data["tran"][seq_id]

            seq_kwargs["joints_glb"] = joint_pos

            # get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_p, glb_root_ori)

            # non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_p, glb_root_ori)

            # get 6D joint rotation
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)

            # get joint velocity (for DIP joint vel label is not used)
            seq_kwargs["joint_vel"] = self.get_joint_vel(joint_p, glb_root_ori)

            # contact points (for DIP contact label is not used)
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_p)

            # uwb
            uwb = raw_data["vuwb"][seq_id]
            if self.normalize_uwb:
                d_root2head = uwb[0, 4, 5].clone()  # normalized by distance between head and root
                uwb = uwb / d_root2head

            if self.flatten_uwb:
                index = torch.triu_indices(IMU_NUM, IMU_NUM, 1)
                uwb = uwb.view(-1, IMU_NUM, IMU_NUM)[:, index[0], index[1]]

            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:, self.remove_node, :] = 0
                uwb[:, :, self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb

            seq_kwargs["uwb_c"] = torch.zeros_like(uwb)
            seq_kwargs["uwb_gt"] = raw_data["uwb_gt"][seq_id] if "uwb_gt" in raw_data else uwb
            seq_kwargs["offset"] = torch.zeros(6, 3)

            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])

            seq_info_ = SeqInfo(seq_id=seq_id, frame_size=N_frame, **seq_kwargs)

            seq_info[seq_id] = seq_info_

        if self.train_split and not self.dry_run:
            torch.save(seq_info, os.path.join(uwb_dir, "dataset_cached.pt"))

        return seq_info


class UWBIMU_real_data_train(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="train", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = True


class UWBIMU_real_data_test(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False


class UWBIMU_real_data_val(UWBIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False


'''
DIP IMU Dataset
'''


class DIPIMU_real_data(_Dataset):
    def __init__(self, dip_dir: str, split, **superkwargs) -> None:
        super().__init__(**superkwargs)
        self.dataset_dir = dip_dir

        seq_info = self._data_preprocess(self.dataset_dir, split=split)

        data_info, data_idx = self._data_sampling(seq_info)

        self.seq_num = data_idx
        self.data_info = data_info
        self.seq_info = seq_info
        print(f"Loading data size of {self.seq_num}..")

    def _get_uwb_(self, joint_p):
        ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
        imu_joint = joint_p[:, ji_mask, ...]
        return torch.cdist(imu_joint, imu_joint)

    def _data_preprocess(self, dip_dir, split="train"):

        seq_info = {}

        if self.add_uwb:
            try:
                uwb_collision = torch.load(os.path.join(dip_dir, f"{split}_uwb_collision.pt"))
            except:
                print("No uwb collision file found, set to zero")
                uwb_collision = None

        print("Load raw data from ", os.path.join(dip_dir, f"{split}.pt"))
        raw_data = torch.load(os.path.join(dip_dir, f"{split}.pt"))

        N_seq = len(raw_data["pose"]) if not self.dry_run else 1
        print(f"Preprocessing Dataset {self.__class__.__name__}")
        for seq_id in tqdm(range(N_seq)):
            joint_pos = raw_data["joint"][seq_id]
            N_frame = joint_pos.size(0)

            if N_frame <= self.seq_length:
                continue

            seq_kwargs = {}
            seq_kwargs["imu_acc"], seq_kwargs["imu_ori"] = \
                self.get_local_imu(raw_data["acc"][seq_id], raw_data["ori"][seq_id])
            # Forward kinematics
            pose_p, shape_p, tran_p = self._preprocess(raw_data["pose"][seq_id],
                                                       shape=None,
                                                       tran=None)  # DIP IMU does not contain translations
            pose_global_p, joint_p = self.model.forward_kinematics(pose_p, shape_p, tran_p, calc_mesh=False)

            seq_kwargs["smpl_aa"] = raw_data["pose"][seq_id]

            seq_kwargs["joints_glb"] = joint_pos


            glb_root_ori = raw_data["ori"][seq_id][:, [-1], ...]

            seq_kwargs["smpl_aa"] = raw_data["pose"][seq_id]
            seq_kwargs["smpl_6d"] = rotation_matrix_to_r6d(
            axis_angle_to_rotation_matrix(raw_data["pose"][seq_id].view(-1, 3))).view(N_frame, -1, 6)
            seq_kwargs["shape"] = raw_data["shape"][seq_id] if "shape" in raw_data else torch.zeros(10)
            seq_kwargs["smpl_tran"] = raw_data["tran"][seq_id]

            # get leaf joint position
            seq_kwargs["leaf_joint"] = self.get_leaf_joint(joint_p, glb_root_ori)

            # non-root joint position wrt root position
            seq_kwargs["non_root_joint"] = self.get_non_root_joint_pos(joint_p, glb_root_ori)

            # get 6D joint rotation
            seq_kwargs["pose_global_6d"] = self.get_6d_rotation(pose_global_p, glb_root_ori)

            # get joint velocity (for DIP joint vel label is not used)
            seq_kwargs["joint_vel"] = self.get_joint_vel(joint_p, glb_root_ori)

            # contact points (for DIP contact label is not used)
            seq_kwargs["feet_contact"] = self.get_contact_points(joint_p)

            # uwb
            v_uwb = raw_data["vuwb"][seq_id]
            seq_kwargs["uwb_gt"] = v_uwb.clone()
            uwb_collision_seq = uwb_collision[seq_id] if uwb_collision is not None else torch.zeros_like(
                raw_data["vuwb"][seq_id])
            uwb = self.add_syn_uwb_noise(v_uwb, sigma=self.static_uwb_noise, uwb_collision=uwb_collision_seq,
                                         normalize=self.normalize_uwb)

            if self.remove_node == -1:
                seq_kwargs["uwb_m"] = uwb
            else:
                uwb[:, self.remove_node, :] = 0
                uwb[:, :, self.remove_node] = 0
                seq_kwargs["uwb_m"] = uwb

            seq_kwargs["uwb_c"] = uwb_collision_seq
            seq_kwargs["offset"] = raw_data["offset"][seq_id]
            # acc_sum
            seq_kwargs["acc_sum"] = self.get_acc_sum(seq_kwargs["imu_acc"])

            seq_info_ = SeqInfo(seq_id=seq_id, frame_size=N_frame, **seq_kwargs)

            if self.uwb_timesample_ratio is not None and self.uwb_timesample_ratio != 1:
                seq_info_.syc_measurement(sampling_ratio=self.uwb_timesample_ratio)

            if self.flatten_uwb:
                seq_info_.flatten_uwb_measurement()

            seq_info[seq_id] = seq_info_

        if self.train_split and not self.dry_run:
            torch.save(seq_info, os.path.join(dip_dir, "dataset_cached.pt"))

        return seq_info


class DIPIMU_real_data_train(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="train", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = True


class DIPIMU_real_data_test(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="test", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False


class DIPIMU_real_data_val(DIPIMU_real_data):
    def __init__(self, dip_dir: str, split="validation", **superkwargs) -> None:
        super().__init__(dip_dir, split, **superkwargs)
        self.train_split = False
