from .dataset_amass import AMASS_Dataset
from .dataset_amass_repaint import Conditioned_AMASS_Dataset as True_AMASS_Dataset

import math
import random

import torch
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

from utils import utils_transform


class AMASS(Dataset):
    def __init__(
        self,
        shard=0,
        num_shards=1,
        amass_opt=None,
        class_cond=False,
    ):
        self.shard = shard
        self.num_shards = num_shards
        self.class_cond = class_cond
        self.dataset = AMASS_Dataset(amass_opt)
        self.indexes = [i for i in range(len(self.dataset))
                        if (i % self.num_shards) == self.shard]
        self.data_name = ['H', 'Head_trans_global']

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        # returns a tensor of shape CHW, where C is the feature
        # dimension, H the time dimension, and W the joint dimension

        window_size = self.dataset.opt['window_size']
        data = self.dataset[self.indexes[idx]]
        out_dict = {}
        if self.class_cond:
            out_dict = {k: v for k, v in data.items() if k not in self.data_name}

        input = data['H'].reshape(window_size, 22, -1)  # Time x Joints x Features
        return torch.permute(input, (2, 0, 1)).squeeze(), out_dict


def load_data_amass(
    *,
    opt,
    class_cond=False,
    joint_cond=False,
    joint_cond_L=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param feat_size: the size of the input features.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """

    dataset = AMASSCond(
        amass_opt=opt['datasets']['train'],
        class_cond=class_cond,
        joint_cond=joint_cond,
        joint_cond_L=joint_cond_L,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )

    loader = DataLoader(
        dataset,
        batch_size=opt['datasets']['train']['dataloader_batch_size'],
        shuffle=opt['datasets']['train']['dataloader_shuffle'],
        num_workers=opt['datasets']['train']['dataloader_num_workers'],
        drop_last=True
    )

    while True:
        yield from loader


class AMASSCond(Dataset):
    def __init__(
        self,
        shard=0,
        num_shards=1,
        amass_opt=None,
        class_cond=False,
        joint_cond=False,
        joint_cond_L=False,
    ):
        self.shard = shard
        self.num_shards = num_shards
        self.class_cond = class_cond
        self.joint_cond = joint_cond
        self.joint_cond_L = joint_cond_L
        self.dataset = True_AMASS_Dataset(amass_opt)
        self.indexes = [i for i in range(len(self.dataset))
                        if (i % self.num_shards) == self.shard]

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        # returns a tensor of shape CHW, where C is the feature
        # dimension, H the time dimension, and W the joint dimension
        window_size = self.dataset.opt['window_size']
        cond_window_size = self.dataset.opt['cond_window_size']
        max_size = max(window_size, cond_window_size)
        data = self.dataset[self.indexes[idx]]
        input = data['H'].reshape(max_size, 22, -1)[-window_size:, :]  # Time x Joints x Features --> takes the last one

        out_dict = {}
        if self.joint_cond:
            if self.joint_cond_L:
                out_dict['joints'] = data['L'].reshape(max_size, 3, -1)[-cond_window_size:, :, :]  # Time x Joints (3) x Features
            else:
                out_dict['joints'] = data['H'].reshape(max_size, 22, -1)[-cond_window_size:, [15, 20, 21], :]  # Time x Joints (3) x Features
            out_dict['joints'] = out_dict['joints'].permute((2, 0, 1))  # Feat, time, joint
        if self.class_cond:
            out_dict = {k: v for k, v in data.items() if k not in self.data_name}

        return torch.permute(input, (2, 0, 1)).squeeze(dim=1), out_dict
