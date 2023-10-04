import torch
import numpy as np
import os
import time
from torch.utils.data import Dataset, DataLoader
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.rotation_tools import aa2matrot,matrot2aa,local2global_pose
import random
from utils import utils_transform

from scipy import signal

import glob
# from IPython import embed
import time
import copy
import pickle


class AMASS_Dataset(Dataset):
    """Motion Capture dataset"""

    def __init__(self, opt):
        self.opt = opt
        self.window_size = opt['window_size']

        self.batch_size = opt['dataloader_batch_size']
        dataroot = opt['dataroot']
        filenames_train = os.path.join(dataroot, '*/train/*.pkl')
        filenames_test = os.path.join(dataroot, '*/test/*.pkl')

# CMU,BioMotionLab_NTroje,MPI_HDM05
        if self.opt['phase'] == 'train':
#            self.filename_list = glob.glob('data_fps60/*/train/*.pkl')
            self.filename_list = glob.glob(filenames_train)
        else:
#            self.filename_list = glob.glob('data_fps60/*/test/*.pkl')
            self.filename_list = glob.glob(filenames_test)

        print('Dataset lenght: {}'.format(len(self.filename_list)))

    def __len__(self):

        return len(self.filename_list)


    def __getitem__(self, idx):

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        while data['rotation_local_full_gt_list'].shape[0] < (self.window_size + 1):
            idx = random.randint(0, len(self) - 1)
            filename = self.filename_list[idx]
            with open(filename, 'rb') as f:
                data = pickle.load(f)


        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        hmd_position_global_full_gt_list = data['hmd_position_global_full_gt_list']
        body_parms_list = data['body_parms_list']
        head_global_trans_list = data['head_global_trans_list']

        frame = np.random.randint(hmd_position_global_full_gt_list.shape[0] - self.window_size)

        if self.opt['phase'] == 'train':
            input_hmd  = hmd_position_global_full_gt_list[frame:frame + self.window_size,...].reshape(self.window_size, -1).float()
            output_gt = rotation_local_full_gt_list[frame: frame + self.window_size,...].float()
            # output_gt = rotation_local_full_gt_list[frame + self.window_size - 1 : frame + self.window_size - 1 + 1,...].float()

            return {'L': input_hmd,
                    'H': output_gt,
                    'P': 1,
                    'Head_trans_global':head_global_trans_list[frame:frame + self.window_size,...],
                    'pos_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1,...],
                    'vel_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1,...]-body_parms_list['trans'][frame + self.window_size - 2:frame + self.window_size - 2+1,...]
                    }

        else:

            input_hmd  = hmd_position_global_full_gt_list[frame:frame + self.window_size,...].reshape(self.window_size, -1).float()
            output_gt = rotation_local_full_gt_list[frame: frame + self.window_size,...].float()
            # input_hmd  = hmd_position_global_full_gt_list.reshape(hmd_position_global_full_gt_list.shape[0], -1)[1:]
            # output_gt = rotation_local_full_gt_list[1:]

            return {'L': input_hmd.float(),
                    'H': output_gt.float(),
                    'P': body_parms_list,
                    'Head_trans_global':head_global_trans_list[frame:frame + self.window_size,...],
                    'pos_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1,...],
                    'vel_pelvis_gt':body_parms_list['trans'][frame + self.window_size - 1:frame + self.window_size - 1,...]-body_parms_list['trans'][frame + self.window_size - 2:frame + self.window_size - 2+1,...]
                    }



class AMASS_ALL_Dataset(AMASS_Dataset):
    """Motion Capture dataset, return all datapoints from the series"""

    def __getitem__(self, idx):

        filename = self.filename_list[idx]
        with open(filename, 'rb') as f:
            data = pickle.load(f)

        rotation_local_full_gt_list = data['rotation_local_full_gt_list']
        hmd_position_global_full_gt_list = data['hmd_position_global_full_gt_list']
        body_parms_list = data['body_parms_list']
        head_global_trans_list = data['head_global_trans_list']

        if self.opt['phase'] == 'train':
            output_gt = rotation_local_full_gt_list.float()
            # output_gt = rotation_local_full_gt_list[frame + self.window_size - 1 : frame + self.window_size - 1 + 1,...].float()

            return {'L': 1,
                    'H': output_gt,
                    'P': 1,
                    'Head_trans_global': head_global_trans_list,
                    'pos_pelvis_gt': 1,
                    'vel_pelvis_gt': 1
                    }

        else:
            output_gt = rotation_local_full_gt_list.float()
            # input_hmd  = hmd_position_global_full_gt_list.reshape(hmd_position_global_full_gt_list.shape[0], -1)[1:]
            # output_gt = rotation_local_full_gt_list[1:]

            return {'L': 1,
                    'H': output_gt.float(),
                    'P': body_parms_list,
                    'Head_trans_global':head_global_trans_list,
                    'pos_pelvis_gt': 1,
                    'vel_pelvis_gt': 1
                    }


    