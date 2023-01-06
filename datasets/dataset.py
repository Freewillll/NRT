
import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys
import os

from pylib.swc_handler import parse_swc
from NRT.augmentation.augmentation import InstanceAugmentation
from NRT.datasets.swc_processing import trim_out_of_box, swc_to_forest

# To avoid the recursionlimit error
sys.setrecursionlimit(30000)


class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(32, 64, 64), seq_node_nums=8):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        print(f'Image shape of {phase}: {imgshape}')
        self.phase = phase
        self.seq_node_nums = seq_node_nums
        self.augment = InstanceAugmentation(p=0.2, imgshape=imgshape, phase=phase)

    @staticmethod
    def load_data_list(split_file, phase):

        with open(split_file, 'rb') as fp:
            data_dict = pickle.load(fp)

        if phase == 'train' or phase == 'val':
            return data_dict[phase]

        elif phase == 'test':
            dd = data_dict['test']
            return dd
        else:
            raise ValueError

    def __getitem__(self, index):
        img, gt, imgfile, swcfile = self.pull_item(index)
        return img, gt, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile = self.data_list[index]
        # parse, image should in [c,z,y,x] format

        img = np.load(imgfile)['data']

        if img.ndim == 3:
            img = img[None]

        if swcfile is not None and self.phase != 'test':
            tree = parse_swc(swcfile)
        else:
            tree = None

        # random augmentation
        img, tree = self.augment(img, tree)

        if tree is not None and self.phase != 'test':
            tree = trim_out_of_box(tree, img[0].shape, True)
            print(f'the len of tree after trim : {len(tree)}')
            seq_list = swc_to_forest(tree)
            print(seq_list)
            # pad the seq_item 
            maxlen_idx = 0
            maxlen = 0
            for idx, seq in enumerate(seq_list):
                if maxlen < len(seq):
                    maxlen = len(seq)
                    maxlen_idx = idx
                for seq_item in seq:
                    print(seq_item)
                    if len(seq_item) < self.seq_node_nums:
                        for i in range(self.seq_node_nums - len(seq_item)):
                            seq_item.append([0, 0, 0, 0])
                    else:
                        print(len(seq_item))
                        raise ValueError
                
            seq = np.array(seq_list[maxlen_idx])
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(seq.astype(np.float32)), imgfile
        else:
            lab = np.random.random(5, 5) > 0.5
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile


if __name__ == '__main__':
    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl'
    idx = 5
    imgshape = (32, 64, 64)
    dataset = GenericDataset(split_file, 'train', imgshape=imgshape)
    img, lab, *_ = dataset.pull_item(idx)
    print(torch.max(img))
    print(lab.shape)
    print(img.shape)
    # import matplotlib.pyplot as plt
    # from utils.image_util import *
    #
    # plt.imshow(unnormalize_normal(img.numpy())[0, 0])
    # from pylib.file_io import *

    # img = unnormalize_normal(img.numpy()).astype(np.uint8)
    # lab = unnormalize_normal(lab.numpy()).astype(np.uint8)

    # save_image('img.v3draw', img)
    # save_image('lab.v3draw', lab)
    #
    # plt.savefig('img.png')
    # plt.imshow(unnormalize_normal(lab[None].numpy())[0, 0])
    # plt.savefig('lab.png')
