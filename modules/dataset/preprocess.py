r"""
    Preprocess DIP-IMU and TotalCapture test dataset.
    Synthesize AMASS dataset.
    
    Adapted from https://github.com/Xinyu-Yi/PIP/blob/main/preprocess.py
"""
import glob
import os
import pickle
import sys

# Make the repo root importable so `articulate`, `config`, `modules`, ... resolve
# when run directly as `python modules/dataset/preprocess.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from tqdm import tqdm
from scipy.interpolate import interp1d

import articulate as art
from config.config import paths, amass_data, amass_test_data
from modules.dataset.dataset import _Dataset
from visualizer.visualize_result import show_vertex_positions
from scipy.spatial.transform import Slerp, Rotation

from pathlib import Path
try:
    from fairmotion.ops import conversions, quaternion
except:
    print("Did not find package fairmotion.ops")

vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])  # lr wrist, lr knee, head, pelvis
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) # [left elbow, right elbow, left knee, right knee, head, pelvis]
uwb_m_mapping = torch.tensor([[1, 2], [1, 4], [1, 5], [1, 3], [1, 0]
                                 , [2, 4], [2, 5], [2, 3], [2, 0]
                                 , [4, 5], [4, 3], [4, 0]
                                 , [5, 3], [5, 0]
                                 , [3, 0]])
uwb_f_mapping = torch.tensor([5, 7, 8, 6, 0, 10, 11, 9, 1, 14, 12, 3, 13, 4, 2])
uwb_imu_mapping = torch.tensor([1, 2, 4, 5, 3, 0])
body_model = art.ParametricModel(paths.smpl_file)


def plot_multidimensional_trajectory(data, title=None, block=True):
    import matplotlib.pyplot as plt
    """
    Plot each dimension vs frames 

    Parameters:
    - data: A numpy array with shape (frame_num, dim) representing the input sequence data.

    """
    num_frames, num_dimensions = data.shape

    # Create an array of frame numbers
    frames = np.arange(num_frames)

    # Set up a colormap for distinguishing dimensions
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, num_dimensions)]

    # Create a line plot for each dimension with color and legend
    for dim in range(num_dimensions):
        plt.plot(frames, data[:, dim], label=f'Dimension {dim}', color=colors[dim])

    # Add labels and title
    plt.xlabel('Frames')
    plt.ylabel('Dimension Value')
    p_title = 'Change of Multidimensional Data Over Frames' if title is None else title
    plt.title(p_title)

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show(block=block)


def _compute_imu_local_offset(j_pos_glb, v_pos_glb, j_ori_glb):
    """
    Compute local offset between imu_placement and the corresponding joint position
    
    """
    r_glb = (j_pos_glb[:, ji_mask] - v_pos_glb[:, vi_mask])

    j_ori_glb = j_ori_glb[:, ji_mask]
    r_local_mean = torch.einsum('bij,bijk->bik', r_glb, j_ori_glb).mean(dim=0)
    return r_local_mean  # 6,3


def _syn_uwb(p):
    """_summary_
    Synthesize UWB value from joint positions
    """
    return torch.cdist(p, p).view(-1, 6, 6)


def _syn_acc(v, smooth_n=4):
    r"""
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def process_amass(test_split=False):
    data_pose, data_trans, data_beta, length, gender = [], [], [], [], []
    if test_split:
        ds_names = amass_test_data
    else:
        ds_names = amass_data

    for ds_name in ds_names:
        print('\rReading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths.raw_amass_dir, ds_name, '*/*_poses.npz'))):
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            poses = cdata['poses']
            trans = cdata['trans']
            if framerate == 120:
                poses = poses[::2]
                trans = trans[::2]
            elif framerate in [59, 60]:
                # Already close enough to 60Hz, no interpolation needed
                pass
            else:
                continue

            data_pose.extend(poses.astype(np.float32))
            data_trans.extend(trans.astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(len(poses))

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]  # right hand
    pose = pose[:, :24].clone()  # only use body

    # align AMASS global fame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)  # x,y,z -> x,z,-y
    pose[:, 0] = art.math.rotation_matrix_to_axis_angle(
        amass_rot.matmul(art.math.axis_angle_to_rotation_matrix(pose[:, 0])))

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    out_offset = []
    out_uwb = []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12:
            b += l
            print('\tdiscard one sequence with length', l)
            continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3  IMU is on l/r wrist
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3 IMU measures the orientation of l/r elbow

        out_uwb.append(_syn_uwb(vert[:, vi_mask]))

        offset = _compute_imu_local_offset(joint, vert, grot)
        out_offset.append(offset)
        b += l

    print('Saving')
    os.makedirs(paths.amass_dir, exist_ok=True)
    if test_split:
        test_folder = os.path.join(paths.amass_dir, f"test_split")
        os.makedirs(test_folder, exist_ok=True)
        torch.save(out_pose, os.path.join(test_folder, 'pose.pt'))
        torch.save(out_shape, os.path.join(test_folder, 'shape.pt'))
        torch.save(out_tran, os.path.join(test_folder, 'tran.pt'))
        torch.save(out_joint, os.path.join(test_folder, 'joint.pt'))
        torch.save(out_vrot, os.path.join(test_folder, 'vrot.pt'))
        torch.save(out_vacc, os.path.join(test_folder, 'vacc.pt'))
        torch.save(out_uwb, os.path.join(test_folder, 'vuwb.pt'))
        torch.save(out_offset, os.path.join(test_folder, 'offset.pt'))
        torch.save(
            {'acc': out_vacc, 'ori': out_vrot, 'pose': out_pose, 'tran': out_tran, "vuwb": out_uwb, "offset": offset, "shape": out_shape},
            os.path.join(test_folder, "test.pt"))
    else:
        torch.save(out_pose, os.path.join(paths.amass_dir, 'pose.pt'))
        torch.save(out_shape, os.path.join(paths.amass_dir, 'shape.pt'))
        torch.save(out_tran, os.path.join(paths.amass_dir, 'tran.pt'))
        torch.save(out_joint, os.path.join(paths.amass_dir, 'joint.pt'))
        torch.save(out_vrot, os.path.join(paths.amass_dir, 'vrot.pt'))
        torch.save(out_vacc, os.path.join(paths.amass_dir, 'vacc.pt'))
        torch.save(out_uwb, os.path.join(paths.amass_dir, 'vuwb.pt'))
        torch.save(out_offset, os.path.join(paths.amass_dir, 'offset.pt'))
        torch.save(
            {'acc': out_vacc, 'ori': out_vrot, 'pose': out_pose, 'tran': out_tran, "vuwb": out_uwb, "offset": offset, "shape": out_shape},
            os.path.join(paths.amass_dir, "train.pt"))
    print('Synthetic AMASS dataset is saved at', paths.amass_dir)


def process_dipimu(data_split="train", sigma=0):
    # with new data format, lw, rw, lk, rk, head, root
    imu_mask = [7, 8, 11, 12, 0, 2]

    if data_split == "train":
        split = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07']  # train sub
    elif data_split == "test":
        split = ['s_09', 's_10']  # test sub
    elif data_split == "validation":
        split = ['s_08']
    else:
        raise KeyError(f"Invalid split {data_split} for DIP IMU")
    accs, oris, poses, joints, trans, v_uwb = [], [], [], [], [], []
    out_offset = []

    for subject_name in split:
        for motion_name in os.listdir(os.path.join(paths.raw_dipimu_dir, subject_name)):
            path = os.path.join(paths.raw_dipimu_dir, subject_name, motion_name)
            data = pickle.load(open(path, 'rb'), encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()
            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]), acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]), ori[1:][torch.isnan(ori[:-1])])

            pose_rot = art.math.axis_angle_to_rotation_matrix(pose[6:-6]).view(-1, 24, 3, 3)
            grot, joint, vert = body_model.forward_kinematics(pose_rot, None, None, calc_mesh=True)
            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                accs.append(acc.clone())
                oris.append(ori.clone())
                poses.append(pose.clone())
                joints.append(joint.clone())
                trans.append(torch.zeros(pose.shape[0], 3))  # dip-imu does not contain translations
                uwb = _syn_uwb(vert[:, vi_mask])
                v_uwb.append(uwb)

                offset = _compute_imu_local_offset(joint, vert, grot)
                out_offset.append(offset)
            else:
                print('DIP-IMU: %s/%s has too much nan! Discard!' % (subject_name, motion_name))

    os.makedirs(paths.dipimu_dir, exist_ok=True)
    file_name = f"{data_split}.pt"
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'joint': joints, 'tran': trans, "vuwb": v_uwb, "offset": out_offset},
               os.path.join(paths.dipimu_dir, file_name))
    print('Preprocessed DIP-IMU dataset is saved at', os.path.join(paths.dipimu_dir, file_name))


def process_totalcapture(sigma=0):
    inches_to_meters = 0.0254
    file_name = 'gt_skel_gbl_pos.txt'
    accs, oris, poses, trans = [], [], [], []

    for file in sorted(os.listdir(paths.raw_totalcapture_dip_dir)):
        data = pickle.load(open(os.path.join(paths.raw_totalcapture_dip_dir, file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[:, torch.tensor([0, 1, 2, 3, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[:, torch.tensor([0, 1, 2, 3, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)  # N, 6, 3
        oris.append(ori)  # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    for subject_name in ['S1', 'S2', 'S3', 'S4', 'S5']:
        for motion_name in sorted(os.listdir(os.path.join(paths.raw_totalcapture_official_dir, subject_name))):
            if subject_name == 'S5' and motion_name == 'acting3':
                continue  # no SMPL poses
            f = open(os.path.join(paths.raw_totalcapture_official_dir, subject_name, motion_name, file_name))
            line = f.readline().split('\t')
            index = torch.tensor([line.index(_) for _ in ['LeftFoot', 'RightFoot', 'Spine']])
            pos = []
            while line:
                line = f.readline()
                pos.append(torch.tensor([[float(_) for _ in p.split(' ')] for p in line.split('\t')[:-1]]))
            pos = torch.stack(pos[:-1])[:, index] * inches_to_meters
            pos[:, :, 0].neg_()
            pos[:, :, 2].neg_()
            trans.append(pos[:, 2] - pos[:1, 2])  # N, 3

    # match trans with poses
    for i in range(len(accs)):
        if accs[i].shape[0] < trans[i].shape[0]:
            trans[i] = trans[i][:accs[i].shape[0]]
        assert trans[i].shape[0] == accs[i].shape[0]

    out_uwb, out_offset = [], []
    # remove acceleration bias
    for iacc, pose, tran in zip(accs, poses, trans):
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)

        uwb = _syn_uwb(vert[:, vi_mask])
        out_uwb.append(uwb)

        offset = _compute_imu_local_offset(joint, vert, grot)
        out_offset.append(offset)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.totalcapture_dir, exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses, 'tran': trans, "vuwb": out_uwb, "offset": out_offset},
               os.path.join(paths.totalcapture_dir, 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at', paths.totalcapture_dir)


def _uwb_to_matrix(uwb: torch.Tensor):
    uwb_matrix = torch.zeros(uwb.size(0), 6, 6)
    idxs = torch.triu_indices(6, 6, 1)
    uwb_matrix[:, idxs[0], idxs[1]] = uwb.squeeze(-1)
    uwb_matrix = uwb_matrix + uwb_matrix.permute(0, 2, 1)
    return uwb_matrix


def process_uwb_dataset(data_split="test", sigma=0.083):
    accs, oris, poses, joints, trans, uwbs, offsets = [], [], [], [], [], [], []
    v_oris, v_acc = [], []
    betas = []
    uwb_gt = []
    file_paths = []
    if data_split == "train":
        split = ['subject_4', 'subject_5', 'subject_6', 'subject_7', 'subject_9']  # train sub
    elif data_split == "test":
        split = ['subject_0', 'subject_2', 'subject_3']  # test sub
    elif data_split == "validation":
        split = ['subject_0']
    else:
        raise KeyError(f"Invalid split {data_split} for UWB dataset")

    def _vis_rot(iori, vori, joint_name, rot_type="euler_xyz"):
        if rot_type == "euler_xyz":
            plot_multidimensional_trajectory(torch.cat([iori[:, [0]], vori[:, [0]]], dim=1),
                                             title='Eular angle x' + joint_name)
            plot_multidimensional_trajectory(torch.cat([iori[:, [1]], vori[:, [1]]], dim=1),
                                             title='Eular angle y' + joint_name)
            plot_multidimensional_trajectory(torch.cat([iori[:, [2]], vori[:, [2]]], dim=1),
                                             title='Eular angle z' + joint_name)
        elif rot_type == "6d_rot":
            plot_multidimensional_trajectory(torch.nn.MSELoss(reduction='none')(iori, vori).mean(-1, keepdim=True),
                                             title='6D_MSE_' + joint_name)
        else:
            raise NotImplementedError

    def _manually_calibrate_mat(target_rot_m, cur_rot_m):
        # use the rot at time 0 for this
        return torch.matmul(target_rot_m, torch.transpose(cur_rot_m, 0, 1))

    def manual_calibration(pose, tran, imu):

        art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(pose_m, tran=tran, calc_mesh=True)

    file_path_count = 0
    for subject_name in split:
        for motion_name in os.listdir(os.path.join(paths.raw_uwbimu_dir, subject_name)):
            file_path_list = sorted(glob.glob(
                os.path.join(paths.raw_uwbimu_dir, subject_name, motion_name, 'processed_*.pkl')))
            for file_path in file_path_list:
                print(f"load file: {file_path}")
                data = pickle.load(open(file_path, 'rb'), encoding='latin1')
                pose = torch.from_numpy(data["pose"]).float()
                tran = torch.from_numpy(data["trans"]).float()
                imu_ori = torch.from_numpy(data["imu_ori"]).float()
                imu_acc = torch.from_numpy(data["imu_acc"]).float()
                uwb = torch.from_numpy(data["uwb"]).float()
                uwb = torch.repeat_interleave(uwb, 3, dim=0)  # upsample uwb from 20Hz to 60Hz

                if torch.isnan(imu_acc).sum() == 0:
                    print(f"{file_path_count}--load file: {file_path}")
                    file_path_count += 1
                    frame_size = min(uwb.size(0), pose.size(0))

                    pose_m = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
                    grot, joint, vert = body_model.forward_kinematics(pose_m, tran=tran, calc_mesh=True)
                    vacc = _syn_acc(vert[:, vi_mask])
                    vacc_tran = _syn_acc(tran)
                    vrot = grot[:, ji_mask]
                    vuwb = _syn_uwb(vert[:, vi_mask])
                    vuwb = _Dataset.add_syn_uwb_noise(vuwb, sigma=sigma)
                    uwb_gt.append(vuwb[:frame_size])

                    accs.append(imu_acc.clone()[:frame_size, uwb_imu_mapping])
                    oris.append(imu_ori.clone()[:frame_size, uwb_imu_mapping])
                    uwb_m = _uwb_to_matrix(uwb[:frame_size])
                    uwb_m = uwb_m[:, uwb_imu_mapping][:, :, uwb_imu_mapping]
                    uwbs.append(uwb_m.clone()[:frame_size])
                    file_paths.append(file_path)

                    tran = tran - tran[0, :]
                    offsets.append(torch.zeros(6, 3))
                    poses.append(pose.clone()[:frame_size])
                    joints.append(joint.clone()[:frame_size])
                    trans.append(tran.clone()[:frame_size])
                    betas.append(torch.from_numpy(data["beta"]).float())
                else:
                    print(f"Missing device value in {file_path}")

    for iacc, pose, tran, irot, iuwb in zip(accs, poses, trans, oris, uwbs):
        pose = art.math.axis_angle_to_rotation_matrix(pose).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(pose, tran=tran, calc_mesh=True)
        vacc = _syn_acc(vert[:, vi_mask])
        for imu_id in range(6):
            for i in range(3):
                d = -iacc[:, imu_id, i].mean() + vacc[:, imu_id, i].mean()
                iacc[:, imu_id, i] += d

    os.makedirs(paths.uwbimu_dir, exist_ok=True)
    output_f_name = f"{data_split}.pt"
    torch.save(
        {'acc': accs, 'ori': oris, 'pose': poses, 'joint': joints, 'tran': trans, "vuwb": uwbs, "uwb_gt": uwb_gt, "offset": offsets,
         "beta": betas, "fnames": file_paths}, os.path.join(paths.uwbimu_dir, output_f_name))
    print('Preprocessed UWB_IMU dataset is saved at', paths.uwbimu_dir)


def process_uwb_demo_data(data_path):
    take_paths = glob.glob(os.path.join(data_path, "*/processed_*.pkl"))
    accs, oris, uwbs = [], [], []
    file_paths = []
    print(take_paths)
    for file_path in take_paths:
        print(f"load file: {file_path}")
        data = pickle.load(open(file_path, 'rb'), encoding='latin1')
        imu_ori = torch.from_numpy(data["imu_ori"]).float()
        imu_acc = torch.from_numpy(data["imu_acc"]).float()
        uwb = torch.from_numpy(data["uwb"]).float()
        uwb = torch.repeat_interleave(uwb, 3, dim=0)  # upsample uwb from 20Hz to 60Hz

        frame_size = min(uwb.size(0), imu_acc.size(0))
        accs.append(imu_acc.clone()[:frame_size, uwb_imu_mapping])
        oris.append(imu_ori.clone()[:frame_size, uwb_imu_mapping])
        uwb_m = _uwb_to_matrix(uwb[:frame_size])
        uwb_m = uwb_m[:, uwb_imu_mapping][:, :, uwb_imu_mapping]
        uwbs.append(uwb_m.clone()[:frame_size])
        file_paths.append(file_path)

    os.makedirs(paths.uwbimu_dir, exist_ok=True)
    output_f_name = f"demo.pt"
    torch.save({'acc': accs, 'ori': oris, "vuwb": uwbs, "fnames": file_paths},
               os.path.join(paths.uwbimu_dir, output_f_name))
    print('Preprocessed UWB_IMU dataset is saved at', paths.uwbimu_dir)


def process_gip(data_split="test"):
    """Stack the two people of the GIP two-person UWB dataset into a single-person dataset.

    The raw GIP-DB ships one preprocessed file per person
    (raw_gip_dir/<split>/person{1,2}/<split>.pt) with identical keys. Each list-valued
    field is concatenated (person1 sequences followed by person2 sequences), yielding a
    dataset of twice the sequence count. Non-list fields (e.g. offset) are passed through
    unchanged.
    """
    p1_path = os.path.join(paths.raw_gip_dir, data_split, "person1", f"{data_split}.pt")
    p2_path = os.path.join(paths.raw_gip_dir, data_split, "person2", f"{data_split}.pt")
    p1 = torch.load(p1_path, map_location="cpu")
    p2 = torch.load(p2_path, map_location="cpu")

    merged = {}
    for key, value in p1.items():
        if isinstance(value, list):
            merged[key] = value + p2[key]
        else:
            merged[key] = value

    os.makedirs(paths.gip_dir, exist_ok=True)
    out_path = os.path.join(paths.gip_dir, f"{data_split}.pt")
    torch.save(merged, out_path)
    print('Stacked GIP dataset is saved at', out_path)


def process_smpl(path: Path, mocap_framerate=60):
    """processes all .npz smpl files it can find recursively in the given path and puts them as one set .pt file.
    Assumes smpl body model
    smplh and smplx will be loaded incorrectly as some indices are different.
    """
    data_pose, data_trans, data_beta, length, gender = [], [], [], [], []

    paths = list(path.glob("**/*.npz"))
    # sort paths
    paths = sorted(paths)
    for npz_fname in tqdm(paths):
        try:
            cdata = np.load(npz_fname)
        except:
            continue

        if "mocap_framerate" in cdata:
            framerate = int(cdata['mocap_framerate'])
        else:
            framerate = mocap_framerate
        poses = cdata['poses']
        if poses.shape[-1] == 72:
            poses = poses.reshape(-1, 24, 3)
        assert poses.shape[1] == 24
        trans = cdata['trans']

        if framerate == 120:
            poses = poses[::2]
            trans = trans[::2]
        elif framerate in [59, 60]:
            # Already close enough to 60Hz, no interpolation needed
            pass
        elif framerate > 0:
            # Interpolate poses and trans to 60Hz
            original_times = np.arange(len(poses)) / framerate
            target_times = np.arange(0, original_times[-1] + 1e-5, 1.0 / 60)

            interp_pose = interp1d(original_times, poses, axis=0, kind='linear', fill_value='extrapolate')
            interp_trans = interp1d(original_times, trans, axis=0, kind='linear', fill_value='extrapolate')

            poses = interp_pose(target_times)
            trans = interp_trans(target_times)
        else:
            print(f"Skipping {npz_fname}: unsupported framerate {framerate}")
            continue

        data_pose.extend(poses.astype(np.float32))
        data_trans.extend(trans.astype(np.float32))
        data_beta.append(cdata['betas'][:10])
        length.append(len(poses))

    assert len(data_pose) != 0, 'AMASS dataset not found. Check config.py or comment the function process_amass()'
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    shape = torch.zeros_like(shape)
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 24, 3)

    print('Synthesizing IMU accelerations and orientations')
    b = 0
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    out_offset = []
    out_uwb = []
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12:
            b += l
            print('\tdiscard one sequence with length', l)
            continue
        p = art.math.axis_angle_to_rotation_matrix(pose[b:b + l]).view(-1, 24, 3, 3)
        grot, joint, vert = body_model.forward_kinematics(p, shape[i], tran[b:b + l], calc_mesh=True)
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joint[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_syn_acc(vert[:, vi_mask]))  # N, 6, 3  IMU is on l/r wrist
        out_vrot.append(grot[:, ji_mask])  # N, 6, 3, 3 IMU measures the orientation of l/r elbow

        out_uwb.append(_syn_uwb(vert[:, vi_mask]))

        offset = _compute_imu_local_offset(joint, vert, grot)
        out_offset.append(offset)
        b += l

    print('Saving')

    out_path = path / "processed"
    os.makedirs(out_path, exist_ok=True)
    torch.save(out_pose, os.path.join(out_path, 'pose.pt'))
    torch.save(out_shape, os.path.join(out_path, 'shape.pt'))
    torch.save(out_tran, os.path.join(out_path, 'tran.pt'))
    torch.save(out_joint, os.path.join(out_path, 'joint.pt'))
    torch.save(out_vrot, os.path.join(out_path, 'vrot.pt'))
    torch.save(out_vacc, os.path.join(out_path, 'vacc.pt'))
    torch.save(out_uwb, os.path.join(out_path, 'vuwb.pt'))
    torch.save(out_offset, os.path.join(out_path, 'offset.pt'))
    torch.save(
        {'acc': out_vacc, 'ori': out_vrot, 'pose': out_pose, 'tran': out_tran, "vuwb": out_uwb,
         "offset": offset},
        out_path / "test.pt")
    print('Synthetic AMASS dataset is saved at', out_path)


if __name__ == '__main__':
    if input("Only evaluation? (y/n)") == "y":
        print("Only processing test split of AMASS for evaluation")
        process_amass(test_split=True)
        process_totalcapture()
        process_dipimu(data_split="test")
        process_gip(data_split="test")
    else:
        process_amass(test_split=True)
        process_amass(test_split=False)
        process_totalcapture()
        process_dipimu(data_split="train")
        process_dipimu(data_split="validation")
        process_dipimu(data_split="test")
        process_gip(data_split="train")
        process_gip(data_split="test")