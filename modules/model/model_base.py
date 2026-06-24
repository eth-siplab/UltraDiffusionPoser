'''
# --------------------------------------------
# Base model
# --------------------------------------------
# Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging (SIGGRAPH 2024)
# https://github.com/eth-siplab/UltraInertialPoser
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''
import argparse
import json

import torch.nn as nn

from modules.dataset.data_utils import D_Batch, Batch


class BaseModel(nn.Module):
    @staticmethod
    def add_args(parser):
        common_args = parser.add_argument_group("Model Base Config")
        common_args.add_argument("--set_imu_m", nargs='+', help='<Optional> change default sensor input',
                                 required=False)
        common_args.add_argument("--model_output", nargs='+', help='<Optional> change default model ouput',
                                 required=False)

    @staticmethod
    def get_config(args):
        raise NotImplementedError

    @staticmethod
    def save_config(args, args_file):
        with open(args_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    @classmethod
    def load_model_with_args(cls, args_file):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        default_args = parser.parse_args([])
        default_dict = vars(default_args)

        with open(args_file, 'r') as f:
            saved_dict = json.load(f)

        merged_dict = default_dict.copy()
        merged_dict.update(saved_dict)

        _args = D_Batch(merged_dict)
        print("Load args :", _args)
        return cls(_args)

    def __init__(self, args):
        super().__init__()
        pass

    def forward(self, batched_data: Batch, perturb=None):
        raise NotImplementedError
