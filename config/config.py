r"""
    Config for paths, joint set, and normalizing scales.
"""
import os
from typing import List

import torch
# datasets (directory names) in AMASS
# e.g., for ACCAD, the path should be `paths.raw_amass_dir/ACCAD/ACCAD/s001/*.npz`
amass_data = ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh', 'Transitions_mocap', 'SSM_synced', 'CMU',
              'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD',
              'BioMotionLab_NTroje', 'BMLhandball', 'MPI_Limits', 'DFaust_67']

amass_test_data = ["DanceDB"]

class paths:
    raw_amass_dir = os.getenv("AMASS_RAW_DIR", 'data/AMASS_SMPLH/')   # raw AMASS (raw_amass_dir/<dataset>/<subject>/*_poses.npz); may be a view of config-named symlinks
    # Processed-dataset dirs are env-overridable (default = the value below) so the
    # train/eval scripts can point a run at a specific dataset without editing this file.
    # e.g. UWBIMU_DIR=data/processed_data/UWB_IMU/SIGGRAPH_dataset for UIP-DB finetune.
    amass_dir = os.getenv("AMASS_DIR", 'data/processed_data/AMASS_syn')   # synthetic AMASS dataset

    raw_dipimu_dir = 'data/DIP_IMU'   # raw DIP-IMU dataset path (raw_dipimu_dir/s_01/*.pkl)
    dipimu_dir = os.getenv("DIPIMU_DIR", 'data/processed_data/DIP')   # preprocessed DIP-IMU dataset

    raw_uip_dir = 'data/UIP'   # raw UWB IMU dataset path
    uwbimu_dir = os.getenv("UWBIMU_DIR", 'data/processed_data/UWB_IMU/SIGGRAPH_dataset')      # output path for the preprocessed UWB IMU dataset

    # GIP (two-person UWB) dataset: per-person preprocessed splits live under
    # raw_gip_dir/<split>/person{1,2}/<split>.pt; processing stacks the two people
    # into a single-person dataset of 2x length at gip_dir.
    raw_gip_dir = os.getenv("GIP_RAW_DIR", 'data/processed_data/GIP-DB')
    gip_dir = os.getenv("GIP_DIR", 'data/processed_data/Multi-UWB-Merged')   # output path for the stacked GIP dataset

    # DIP recalculates the SMPL poses for TotalCapture dataset. You should acquire the pose data from the DIP authors.
    raw_totalcapture_dip_dir = 'data/TotalCapture_Real_60FPS'  # contain ground-truth SMPL pose (*.pkl)
    raw_totalcapture_official_dir = 'data/TotalCapture'    # contain official gt (S1/acting1/gt_skel_gbl_pos.txt)
    totalcapture_dir = os.getenv("TOTALCAPTURE_DIR", 'data/processed_data/TotalCapture')   # preprocessed TotalCapture dataset

    result_dir = 'data/result'                      # output directory for the evaluation results

    smpl_file = os.getenv("SMPL_MODEL_PATH", "data/smpl_m_lbs_10_207_0_v1.0.0.pkl")     # official SMPL model path
    physics_model_file = 'data/urdfmodels/physics.urdf'      # physics body model (used by the ZMP metric)
    weights_file = 'data/weights.pt'                # network weight file

class joint_set:
    root = [0]
    leaf = [4, 5, 12, 20, 21]
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]
    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)
    # contact dim order: 7, 10, 8, 11, left ankle, toe, right angle, toe
    foot_indices = [7, 10, 8, 11]

    # uwb_gt order is [lw, rw, lk, rk, head, root]
    # lfj_gt order is [lk, rk, head, lw, rw]


class IMU_placement:
    SENSOR_ORDER = ["lw", "rw", "lk", "rk", "head", "root"]

    vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
    ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) # lw, rw, lk, rk, head, root
      
vel_scale = 3
uwb_collision_thr = 0.3 
'''
SMPL mapping 
'''
SMPL_JOINTS = [
    "root",#        
    "lhip",
    "rhip",
    "lowerback",
    "lknee",
    "rknee",
    "upperback",
    "lankle",
    "rankle",
    "chest",
    "ltoe",
    "rtoe",
    "lowerneck",
    "lclavicle",
    "rclavicle",
    "upperneck",
    "lshoulder",
    "rshoulder",
    "lelbow",
    "relbow",
    "lwrist",
    "rwrist",
    "lhand",
    "rhand",
]
SMPL_JOINT_IDX_MAPPING = {x: i for i, x in enumerate(SMPL_JOINTS)}
SMPL_IDX_JOINT_MAPPING = {i: x for i, x in enumerate(SMPL_JOINTS)}

''' 
Definition of Link/Joint (In our character definition, one joint can only have one link)
'''
root = -1
lhip = 0
lknee = 1
lankle = 2
rhip = 3
rknee = 4
rankle = 5
lowerback = 6
upperback = 7
chest = 8
lowerneck = 9
upperneck = 10
lclavicle = 11
lshoulder = 12
lelbow = 13
lwrist = 14
rclavicle = 15
rshoulder = 16
relbow = 17
rwrist = 18

import collections
bvh_map = collections.OrderedDict()

bvh_map[root] = "root"
bvh_map[lhip] = "lhip"
bvh_map[lknee] = "lknee"
bvh_map[lankle] = "lankle"
bvh_map[rhip] = "rhip"
bvh_map[rknee] = "rknee"
bvh_map[rankle] = "rankle"
bvh_map[lowerback] = "lowerback"
bvh_map[upperback] = "upperback"
bvh_map[chest] = "chest"
bvh_map[lowerneck] = "lowerneck"
bvh_map[upperneck] = "upperneck"
bvh_map[lclavicle] = "lclavicle"
bvh_map[lshoulder] = "lshoulder"
bvh_map[lelbow] = "lelbow"
bvh_map[lwrist] = "lwrist"
bvh_map[rclavicle] = "rclavicle"
bvh_map[rshoulder] = "rshoulder"
bvh_map[relbow] = "relbow"
bvh_map[rwrist] = "rwrist"

IMU_USE = ["lw", "rw", "lk", "rk", "head", "root"]
IMU_NUM = len(IMU_USE)
INPUT_DATA_SIZE = {"vacc":IMU_NUM * 3,"vrot": IMU_NUM * 9, "vuwb" :IMU_NUM * 6,"acc_sum":IMU_NUM * 3,"f_vuwb": int(IMU_NUM * (IMU_NUM - 1) / 2)}

UIP_UWB_NOISE_LEVEL = torch.tensor([
#    lw     rw      lk      rk      head    root
    [0.000, 0.090,  0.075,  0.075,  0.050,  0.060],  # lw
    [0.090, 0.000,  0.075,  0.075,  0.050,  0.060],  # rw
    [0.075, 0.075,  0.000,  0.060,  0.085,  0.030],  # lk
    [0.075, 0.075,  0.060,  0.000,  0.085,  0.030],  # rk
    [0.050, 0.050,  0.085,  0.085,  0.000,  0.040],  # head
    [0.060, 0.060,  0.030,  0.030,  0.040,  0.000]   # root
])



def get_input_size(sensor_list: List[str] = None):
    return sum([INPUT_DATA_SIZE[k] for k in sensor_list])