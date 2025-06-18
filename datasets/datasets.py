# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import numpy as np
import torch
import torch.utils
import torch.utils.data
from tqdm import tqdm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve


DATASETS_CONFIG = {
        'PaviaC': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat', 
                     'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
            'img': 'Pavia.mat',
            'gt': 'Pavia_gt.mat'
            },
        'PaviaU': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                     'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
            'img': 'PaviaU.mat',
            'gt': 'PaviaU_gt.mat'
            },
        'KSC': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                     'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
            'img': 'KSC.mat',
            'gt': 'KSC_gt.mat'
            },
        'IndianPines': {
            'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                     'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
            'img': 'Indian_pines_corrected.mat',
            'gt': 'Indian_pines_gt.mat'
            },
        'Botswana': {
            'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                     'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
            'img': 'Botswana.mat',
            'gt': 'Botswana_gt.mat',
            },
        'Houston13': {
            'img': 'Houston13.mat',
            'gt': 'Houston13_7gt.mat',
            },
        'Houston18': {
            'img': 'Houston18.mat',
            'gt': 'Houston18_7gt.mat',
            },
        'YCZY': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY1_02D_gt.mat',
            },
        'YCGF': {
            'img': 'YC_GF5_data.mat.mat',
            'gt': 'YC_GF5_gt.mat',
            },
        'YCZY9': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY9_gt.mat',
            },
        'HHK': {
            'img': 'HHK_ZY_data.mat.mat',
            'gt': 'HHK_ZY_gt.mat',
            },
        'YCZY8': {
            'img': 'YC_ZY1_02D_data.mat.mat',
            'gt': 'YC_ZY_qj8_gt.mat',
            },
        'HHK8': {
            'img': 'HHK_ZY_data.mat.mat',
            'gt': 'HHK_ZY_qj8_gt.mat',
            },
        'Shanghai': {
            'img': 'Shanghai.mat',
            'gt': 'Shanghai_gt.mat',
            },
        'Hangzhou': {
            'img': 'Hangzhou.mat',
            'gt': 'Hangzhou_gt.mat',
            },
        'Dioni': {
            'img': 'Dioni.mat',
            'gt': 'Dioni_gt_out68.mat',
            },
        'Loukia': {
            'img': 'Loukia.mat',
            'gt': 'Loukia_gt_out68.mat',
            },
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        # self.name = hyperparams['dataset']
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        
        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]

        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        return data, label

    # def __getitem__(self, i):
    #     x, y = self.indices[i]
    #     x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
    #     x2, y2 = x1 + self.patch_size, y1 + self.patch_size

    #     data = self.data[x1:x2, y1:y2]
    #     label = self.label[x1:x2, y1:y2]
    #     if self.transform is not None:
    #         data = self.transform(data) 
    #     return data, self.labels[i]
