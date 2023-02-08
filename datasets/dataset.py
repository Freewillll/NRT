
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


from swc_handler import parse_swc
from augmentation.augmentation import InstanceAugmentation
from datasets.swc_processing import trim_out_of_box, swc_to_forest

# To avoid the recursionlimit error
sys.setrecursionlimit(30000)


PAD = 0
EOS = 6

def collate_fn(batch):

#   batch: [[img, seq, imgfile] for data in batch]
#   return:  [img] * batch_size, [seq]*batch_szie, [imgfile]*batch_size

    dynamical_pad = True
    max_len = 6

    lens = [dat[1].shape[0] for dat in batch]

    # find the max len in each batch
    if dynamical_pad:
        seq_len = max(lens)
    else:
        # fixed length padding
        seq_len = max_len
    # print("collate_fn seq_len", seq_len)
    print(len(batch))
    output_seq = []
    output_img = []
    output_imgfile = []
    output_swcfile = []
    for data in batch:
        seq = data[1][:seq_len]
        # pad shape  (pad_len, item_len, vec_len)
        pad_shape = [max_len-len(seq)] + list(data[1].shape[1:])
        padding = torch.zeros(pad_shape)
        seq = torch.cat((seq, padding), 0).tolist()
        
        output_img.append(data[0].tolist())
        output_seq.append(seq)
        output_imgfile.append(data[-2])
        output_swcfile.append(data[-1])

    output_img = torch.tensor(output_img, dtype=torch.float32)
    output_seq = torch.tensor(output_seq, dtype=torch.float32)
    return output_img, output_seq, output_imgfile, output_swcfile


def draw_lab(lab, lab_image):
    # the shape of lab         seq_len, item_len, vec_len
    # the shape of lab_image   z, y, x 
    imgshape = lab_image.shape
    # filter out invalid  point
    nodes = lab[lab[:, :, -1] > 0]
    # keep the position of nodes in the range of imgshape
    nodes = np.clip(nodes, [0,0,0,0], [i -1 for i in imgshape] + [EOS]).astype(int)[:,:-1]
    # draw nodes
    for node in nodes:
        lab_image[node[0], node[1], node[2]] = 1
    selem = np.ones((1,3,3), dtype=np.uint8)
    lab_image = morphology.dilation(lab_image, selem)
    return lab_image


class GenericDataset(tudata.Dataset):

    def __init__(self, split_file, phase='train', imgshape=(32, 64, 64), seq_node_nums=8, node_dim=4):
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
        img, gt, imgfile = self.pull_item(index)
        return img, gt, imgfile

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
            seq_list = swc_to_forest(tree)

            # print(seq_list)
            # print(f'---------the len of seq list : {len(seq_list)}')

            # pad the seq_item 
            # find the seq has max len
            maxlen_idx = 0         
            maxlen = 0
            len_outofrange = False
            outofrange_list = []
            for idx, seq in enumerate(seq_list):
                if maxlen < len(seq):
                    maxlen = len(seq)
                    maxlen_idx = idx
                for seq_item in seq:
                    if len(seq_item) <= self.seq_node_nums:
                        for i in range(self.seq_node_nums - len(seq_item)):
                            seq_item.append(self.node_dim * [PAD])
                    else:
                        len_outofrange = True
                        if idx not in outofrange_list:
                            outofrange_list.append(idx)
                        # print(len(seq_item))
                        break

            # find a seq the lenght of which is in range
            if len_outofrange:
                for idx in range(0, len(seq_list)):
                    if idx not in outofrange_list:
                        maxlen_idx = idx
            # print(f'----------maxlen idx : {maxlen_idx}')
            seq = np.array(seq_list[maxlen_idx])
            # add eos
            eos = np.zeros((1, self.seq_node_nums, self.node_dim))
            eos[:, :, -1] = EOS
            seq = np.concatenate((seq, eos), axis=0)
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(seq.astype(np.float32)), imgfile, swcfile
        else:
            lab = np.random.random((5, self.seq_node_nums, self.node_dim)) > 0.5
            return torch.from_numpy(img.astype(np.float32)), torch.from_numpy(lab.astype(np.uint8)), imgfile, swcfile


if __name__ == '__main__':

    import skimage.morphology as morphology
    import cv2 as cv
    from torch.utils.data.distributed import DistributedSampler
    import utils.util as util


    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl'
    idx = 1
    imgshape = (32, 64, 64)
    dataset = GenericDataset(split_file, 'train', imgshape=imgshape)

    # loader = tudata.DataLoader(dataset, 2, 
    #                             num_workers=8, 
    #                             shuffle=False, pin_memory=True,
    #                             drop_last=True, 
    #                             collate_fn=collate_fn,
    #                             worker_init_fn=util.worker_init_fn)
    # for i, batch in enumerate(loader):
    #     img, seq ,*_ = batch
    #     print(img.shape)
    #     print(seq.shape)
    #     break

    img, lab, *_ = dataset.pull_item(idx)
    lab_image = np.zeros(img.shape[1:])
    print(lab_image.shape)
    print(lab)
    lab = lab.numpy()
    lab_image = draw_lab(lab, lab_image)
    

    import matplotlib.pyplot as plt
    from utils.image_util import *
    from datasets.mip import *

    img = unnormalize_normal(img.numpy()).astype(np.uint8)[0]
    lab_image = unnormalize_normal(lab_image).astype(np.uint8)[0]
    print(lab_image.shape)
    print(lab_image.dtype)
    plt.imshow(cv.addWeighted(convert_color_w(img_contrast(np.max(img, axis=0), contrast=5.0)), 0.5,
               convert_color_r(np.max(lab_image, axis=0)), 0.5, 0))
    plt.savefig('test.png')

