
import numpy as np
import pickle
import torch.utils.data as tudata
import SimpleITK as sitk
import torch
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from swc_handler import parse_swc, write_swc
from augmentation.augmentation import InstanceAugmentation
from datasets.swc_processing import trim_out_of_box, swc_to_points

# To avoid the recursionlimit error
sys.setrecursionlimit(30000)


SEQ_PAD = -1
NODE_PAD = 0
EOS = 5

def collate_fn(batch):

#   batch: [[img, input_node, target, imgfile, swcfile] for data in batch]
#   return:  [img]*batch_size, [seq]*batch_size, [cls_]*batch_size, [imgfile]*batch_size, [swcfile]*batch_size
    
    output_target = []
    output_img = []
    output_imgfile = []
    output_swcfile = []
    for data in batch:
        output_target.append(data[1])
        output_img.append(data[0].tolist())
        output_imgfile.append(data[-2])
        output_swcfile.append(data[-1])

    output_img = torch.tensor(output_img, dtype=torch.float32)
    return output_img, output_target, output_imgfile, output_swcfile


def draw_lab(lab, cls_, img):
    # the shape of lab         seq_len, item_len, vec_len
    # the shape of lab_image   z, y, x 
    lab_img = np.repeat(img, 3, axis=0)
    lab_img[0, :, :, :] = 0
    lab_img[2, :, :, :] = 0
    lab = lab[:-1, ...]
    cls_ = cls_[:-1, ...]
    # filter out invalid  point
    nodes = lab[cls_ > 0]
    cls_ = cls_[cls_ > 0]
    # keep the position of nodes in the range of imgshape
    nodes = np.clip(nodes, [0,0,0], [i -1 for i in imgshape]).numpy().astype(int)
    # draw nodes
    for idx, node in enumerate(nodes):
        if cls_[idx] == 1: # root white
            lab_img[:, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 2: # branching point yellow
            lab_img[0, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 3: # tip blue
            lab_img[2, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 4: #boundary blue
            lab_img[2, node[0], node[1], node[2]] = 255
    selem = np.ones((1,2,3,3), dtype=np.uint8)
    lab_img = morphology.dilation(lab_img, selem)
    return lab_img


class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(32, 64, 64), seq_node_nums=2, node_dim=4):
        self.data_list = self.load_data_list(split_file, phase)
        self.imgshape = imgshape
        print(f'Image shape of {phase}: {imgshape}')
        self.phase = phase
        self.seq_node_nums = seq_node_nums
        self.node_dim = node_dim
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
        img, target, imgfile, swcfile = self.pull_item(index)
        return img, target, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index, num_nodes=110):
        imgfile, swcfile = self.data_list[index]
        # parse, image should in [c,z,y,x] format

        img = np.load(imgfile)['data']

        if img.ndim == 3:
            img = img[None]

        if swcfile is not None and self.phase != 'test':
            tree = parse_swc(swcfile)

        # random augmentation
        img, tree = self.augment(img, tree)

        if tree is not None and self.phase != 'test':
            tree_crop = trim_out_of_box(tree, img[0].shape, True)
            poses, labels = swc_to_points(tree_crop, img[0].shape)
            poses = poses[:num_nodes]
            labels = labels[:num_nodes]

            target = {'poses': torch.tensor(poses, dtype=torch.float32), 'labels': torch.tensor(labels, dtype=torch.int64)}
            return torch.from_numpy(img.astype(np.float32)), target, imgfile, swcfile
        else:
            lab = np.random.randn((2, self.seq_node_nums, self.node_dim)) > 0.5
            cls_ = np.random.randn((2, self.seq_node_nums)) > 0.5
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.float32)), torch.from_numpy(cls_.astype(np.int64)), imgfile, swcfile


if __name__ == '__main__':

    import skimage.morphology as morphology
    # import cv2 as cv
    from torch.utils.data.distributed import DistributedSampler
    import utils.util as util
    # import matplotlib.pyplot as plt
    from utils.image_util import *
    # from datasets.mip import *
    from train import *


    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl'
    idx = 1
    imgshape = (64, 128, 128)
    dataset = GenericDataset(split_file, 'train', imgshape=imgshape)

    loader = tudata.DataLoader(dataset, 4, 
                                num_workers=8, 
                                shuffle=False, pin_memory=True,
                                drop_last=True, 
                                collate_fn=collate_fn,
                                worker_init_fn=util.worker_init_fn)
    for i, batch in enumerate(loader):
        img, targets , imgfiles, swcfile = batch
        print(f'targets: {targets}')
        for v in targets:
            print(len(v['labels']))
        print(imgfiles)
        # print(targets.shape)
        # save_image_in_training(imgfiles, img, seq, cls_, pred=None, phase='train', epoch=1, idx=0)

        


    # img, lab, cls_, *_ = dataset.pull_item(idx)
    # img = unnormalize_normal(img.numpy()).astype(np.uint8)
    # pos = util.pos_unnormalize(lab[..., :3], img.shape[1:])
    # print(pos)
    # print(cls_)
    # lab_image = draw_lab(pos, cls_, img)

    # save_image('test.v3draw', lab_image)
    # lab_image = lab_image[1]
    # print(lab_image.shape)
    # print(lab_image.dtype)
    # plt.imshow(cv.addWeighted(convert_color_w(img_contrast(np.max(img[0], axis=0), contrast=5.0)), 0.5,
    #            convert_color_r(np.max(lab_image, axis=0)), 0.5, 0))
    # plt.savefig('test.png')

