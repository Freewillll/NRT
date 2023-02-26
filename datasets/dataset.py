
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
from datasets.swc_processing import trim_out_of_box, swc_to_forest

# To avoid the recursionlimit error
sys.setrecursionlimit(30000)


SEQ_PAD = -1
NODE_PAD = 0
EOS = 5

def collate_fn(batch):

#   batch: [[img, seq, cls, imgfile, swcfile] for data in batch]
#   return:  [img]*batch_size, [seq]*batch_size, [cls_]*batch_size, [imgfile]*batch_size, [swcfile]*batch_size

    dynamical_pad = True
    max_len = 6

    lens = [dat[1].shape[0] for dat in batch]

    # find the max len in each batch
    if dynamical_pad:
        max_len = max(lens)
    # print("collate_fn seq_len", seq_len)
    output_seq = []
    output_img = []
    output_cls = []
    output_imgfile = []
    output_swcfile = []
    for data in batch:
        seq = data[1][:max_len]
        pad_shape = [max_len-len(seq)] + list(data[1].shape[1:])
        padding = torch.zeros(pad_shape)
        padding[:] = SEQ_PAD
        seq = torch.cat((seq, padding), 0).tolist()

        cls_ = data[2][:max_len]
        pad_shape = [max_len-len(cls_)] + list(data[2].shape[1:])
        padding = torch.zeros(pad_shape, dtype=torch.int64)
        padding[:] = SEQ_PAD
        cls_= torch.cat((cls_, padding), 0).tolist()

        output_img.append(data[0].tolist())
        output_seq.append(seq)
        output_cls.append(cls_)
        output_imgfile.append(data[-2])
        output_swcfile.append(data[-1])

    output_img = torch.tensor(output_img, dtype=torch.float32)
    output_seq = torch.tensor(output_seq, dtype=torch.float32)
    output_cls = torch.tensor(output_cls, dtype=torch.int64)
    return output_img, output_seq, output_cls, output_imgfile, output_swcfile


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
        img, seq, cls_, imgfile, swcfile = self.pull_item(index)
        return img, seq, cls_, imgfile, swcfile

    def __len__(self):
        return len(self.data_list)

    def pull_item(self, index):
        imgfile, swcfile = self.data_list[index]
        # parse, image should in [c,z,y,x] format

        img = np.load(imgfile)['data']

        if img.ndim == 3:
            img = img[None]

        if swcfile is not None and self.phase == 'test':
            tree = parse_swc(swcfile)
            img, tree = self.augment(img, tree)
            _, _, x, y, z, *_ = tree[0]
            start_node = [[z, y, x, 1]]
            for i in range(self.seq_node_nums - len(start_node)):
                node_pad = self.node_dim * [0]
                node_pad[-1] = NODE_PAD
                start_node.append(node_pad)

            start_node = np.asarray(start_node)
            cls_ = start_node[..., -1].copy()

            start_node = start_node.astype(np.float32)
            for i in range(3):
                start_node[..., i] = (start_node[..., i] - 0) / (img.shape[i+1] - 0 + 1e-8)
            start_node[..., -1] = (start_node[..., -1] - NODE_PAD) / (EOS - NODE_PAD + 1e-8)
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(start_node.astype(np.float32)), torch.from_numpy(cls_.astype(np.int64)), imgfile, swcfile


        if swcfile is not None and self.phase != 'test':
            tree = parse_swc(swcfile)

        # random augmentation
        img, tree = self.augment(img, tree)

        if tree is not None and self.phase != 'test':
            tree_crop = trim_out_of_box(tree, img[0].shape, True)
            seq_list = swc_to_forest(tree_crop, img[0].shape)
            
            # if len(seq_list) == 0:
            #     os.makedirs('./debug', exist_ok=True)
            #     write_swc(tree, os.path.join('debug', os.path.split(swcfile)[-1]))
            #     write_swc(tree_crop, os.path.join('debug', f'crop_{os.path.split(swcfile)[-1]}'))
            #     print(imgfile, swcfile)

            # pad the seq_item 
            # find the seq has max len
            maxlen_idx = 0         
            maxlen = 0

            for idx, seq in enumerate(seq_list):
                if maxlen < len(seq):
                    maxlen = len(seq)
                    maxlen_idx = idx
                for seq_item in seq:
                    if len(seq_item) <= self.seq_node_nums:
                        for i in range(self.seq_node_nums - len(seq_item)):
                            node_pad = self.node_dim * [0]
                            node_pad[-1] = NODE_PAD
                            seq_item.append(node_pad)
                    else:
                        for i in range(len(seq_item) - self.seq_node_nums):
                            seq_item.pop()

            # find a seq the lenght of which is in range
            seq = np.asarray(seq_list[maxlen_idx])
            # add eos
            eos = np.zeros((1, self.seq_node_nums, self.node_dim))
            eos[..., 0, -1] = EOS
            seq = np.concatenate((seq, eos), axis=0)
            cls_ = seq[..., -1].copy()
            #normailize
            seq = seq.astype(np.float32)
            for i in range(3):
                seq[..., i] = (seq[..., i] - 0) / (img.shape[i+1] - 0 + 1e-8)
            seq[..., -1] = (seq[..., -1] - seq[..., -1].min()) / (seq[..., -1].max() - seq[..., -1].min() + 1e-8)
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(seq.astype(np.float32)), torch.from_numpy(cls_.astype(np.int64)), imgfile, swcfile
        else:
            lab = np.random.randn((5, self.seq_node_nums, self.node_dim)) > 0.5
            cls_ = np.random.randn((5, self.seq_node_nums)) > 0.5
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
    imgshape = (32, 64, 64)
    dataset = GenericDataset(split_file, 'test', imgshape=imgshape)

    loader = tudata.DataLoader(dataset, 4, 
                                num_workers=8, 
                                shuffle=False, pin_memory=True,
                                drop_last=True, 
                                collate_fn=collate_fn,
                                worker_init_fn=util.worker_init_fn)
    for i, batch in enumerate(loader):
        img, seq , cls_, imgfiles, swcfile = batch
        print(seq)
        print(cls_)
        # save_image_in_training(imgfiles, img, seq, cls_, pred=None, phase='train', epoch=1, idx=0)
        break
        


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

