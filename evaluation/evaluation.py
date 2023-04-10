import numpy as np
import torch
import torch.nn.functional as F
from skimage.graph import route_through_array
import math

import utils.util as util
from datasets.dataset import *


class GenericMultiCropEvaluation(object):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='min'):
        self.patch_size = patch_size
        self.divid = divid
        self.pad_value = pad_value

    def get_divid_shape(self, dim_size, divid):
        new_size = int(math.ceil(dim_size / divid) * divid)
        return new_size

    def get_pad_value(self, img):
        if self.pad_value == 'min':
            pad_value, _ = img.reshape((img.shape[0], -1)).min(dim=1)
        elif self.pad_value == 'max':
            pad_value, _ = img.reshape((img.shape[0], -1)).max(dim=1)
        elif self.pad_value == 'mean':
            pad_value, _ = img.reshape((img.shape[0], -1)).mean(dim=1)
        elif isinstance(self.pad_value, int) or isinstance(self.pad_value, float):
            pad_value = self.pad_value
        elif isinstance(self.pad_value, tuple) or isinstance(self.pad_value, list):
            pad_value = self.pad_value
        else:
            raise ValueError("pad_value is incorrect!")
        return pad_value

    def run_evalute(self, img):
        pass

class NonOverlapCropEvaluation(GenericMultiCropEvaluation):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='min'):
        super(NonOverlapCropEvaluation, self).__init__(patch_size, divid, pad_value)

    def get_crop_sizes(self, imgshape):
        crop_sizes = []
        pads = []
        for pi, si, di in zip(self.patch_size, imgshape, self.divid):
            crop_size = self.get_divid_shape(pi, di)
            ncrop = int(math.ceil(si / crop_size))
            crop_sizes.append(crop_size)
            pads.append(crop_size * ncrop - si)
            
        return crop_sizes, pads

    def get_image_crops(self, img, lab=None):
        # image in shape [c, z, y, x]
        imgshape = img[0].shape

        pad_value = self.get_pad_value(img)
        crops = []
        lab_crops = None
        if lab is not None:
            lab_crops = []

        crop_sizes, pads = self.get_crop_sizes(imgshape)
        (size_z, size_y, size_x) = crop_sizes
        for zi in range(int(math.ceil(imgshape[0] / size_z))):
            zs, ze = zi * size_z, (zi+1) * size_z
            for yi in range(int(math.ceil(imgshape[1] / size_y))):
                ys, ye = yi * size_y, (yi+1) * size_y
                for xi in range(int(math.ceil(imgshape[2] / size_x))):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    crop = img[:, zs:ze, ys:ye, xs:xe]
                    if crop.shape != tuple(crop_sizes):
                        new_crop = torch.ones((img.shape[0], *crop_sizes), dtype=img.dtype, device=img.device) * pad_value
                        cropz, cropy, cropx = crop[0].shape
                        new_crop[:, :cropz, :cropy, :cropx] = crop
                        crops.append(new_crop)

                        # for lab
                        if lab is not None:
                            new_lab_crop = torch.zeros(crop_sizes, dtype=lab.dtype, device=lab.device)
                            new_lab_crop[:cropz, :cropy, :cropx] = lab[zs:ze, ys:ye, xs:xe]
                            lab_crops.append(new_lab_crop)
                    else:
                        crops.append(crop)
                        if lab is not None:
                            lab_crops.append(lab[zs:ze, ys:ye, xs:xe])

        return crops, crop_sizes, lab_crops


class MostFitCropEvaluation(GenericMultiCropEvaluation):
    def __init__(self, patch_size, divid=(2**4,2**5,2**5), pad_value='min'):
        super(MostFitCropEvaluation, self).__init__(patch_size, divid, pad_value)

    def get_crop_sizes(self, imgshape):
        crop_sizes = []
        pads = []
        for pi, si, di in zip(self.patch_size, imgshape, self.divid):
            if pi >= si:
                crop_size = self.get_divid_shape(si, di)
                crop_sizes.append(crop_size)
                pads.append(crop_size - si)
            else:
                crop_size = self.get_divid_shape(pi, di)
                ncrop = int(math.ceil(si / crop_size))
                crop_sizes.append(crop_size)
                pads.append(crop_size * ncrop - si)
            
        return crop_sizes, pads

    def get_image_crops(self, img, lab=None):
        # image in shape [c, z, y, x]
        imgshape = img[0].shape
        crop_sizes, pseudo_pads = self.get_crop_sizes(imgshape)
        (size_z, size_y, size_x) = crop_sizes

        # pre-padding images
        padding = []
        for i, crop_size in enumerate(crop_sizes):
            if imgshape[i] < crop_size:
                padding.append(0)
                padding.append(crop_size - imgshape[i])
            else:
                padding.append(0)
                padding.append(0)
        #print(padding)

        # NOTE: for multi-modulity image, this may problematic!! to be fixed later
        pad_value = self.get_pad_value(img)
        if isinstance(pad_value, torch.Tensor):
            pad_value = float(pad_value[0])
        
        pad_img = F.pad(img[None], padding, mode='constant', value=pad_value)[0]  # 3D padding requires 5D input
        nshape = pad_img[0].shape
        lab_crops = None
        if lab is not None:
            lab_crops = []
            pad_lab = F.pad(lab[None], padding, mode='constant', value=0)[0]
        crops = []
        index_list = []
        
        for zi in range(int(math.ceil(nshape[0] / size_z))):
            zs, ze = zi * size_z, (zi+1) * size_z
            if ze > nshape[0]:
                ze = None
                zs = -size_z
            for yi in range(int(math.ceil(nshape[1] / size_y))):
                ys, ye = yi * size_y, (yi+1) * size_y
                if ye > nshape[1]:
                    ye = None
                    ys = -size_y

                for xi in range(int(math.ceil(nshape[2] / size_x))):
                    xs, xe = xi * size_x, (xi+1) * size_x
                    if xe > nshape[2]:
                        xe = None
                        xs = -size_x
                    #print(zi,yi,xi,zs,ze,ys,ye,xs,xe,pad_img.shape)
                    crop = pad_img[:, zs:ze, ys:ye, xs:xe]
                    crops.append(crop)
                    index_list.append((zi, yi, xi))
                    if lab is not None:
                        lab_crop = pad_lab[:, zs:ze, ys:ye, xs:xe]
                        lab_crops.append(lab_crop)
        
        return crops, crop_sizes, lab_crops, index_list
    

class Tracing(object):
    def __init__(self, model, max_len):
        self.model = model
        self.max_len = max_len
        self.model.eval()

    def indices2tree(self, indices, tree, index):
        for idx, node in enumerate(indices, start=index+1):
            tree.append((idx, 2, node[2], node[1], node[0], 1, index))
            index += 1
        return tree, index

    def trace_from_soma(self, img, seq, cls_):
        # pred: b, n, nodes, dims
        # cls_: b, n, nodes
    
        for i in range(self.max_len):
            pred = self.model(img, seq)
            output = pred[:, -1, ...].clone()
            # pre_cls, b, n, nums
            pred_cls = torch.argmax(output[..., 3:], dim=-1)
            if pred_cls[0, 0, 0] == EOS:
                break
            else:
                # normalize
                pred_cls = (pred_cls - NODE_PAD) / (EOS - NODE_PAD)
                pred_seq = torch.concatenate((output[..., :3], pred_cls.unsqueeze(3)), dim=-1)
                seq = torch.cat((seq, pred_seq), dim=1)

        pos = util.pos_unnormalize(seq[..., :3], img.shape[2:])[0]
        cls_ = torch.argmax(seq[..., :3], dim=-1)[0]
        
        # pos: n, nodes, dims,  (z, y, x)
        # cls: n, nodes
        imgshape = img.shape[2:]
        tip_z_m, tip_z_p, tip_y_m, tip_y_p, tip_x_m, tip_x_p= [], [], [], [], [], []
        tree = []
        weight_img = 255 - img
        index = 1
        for i in range(len(pos) - 1):  # level i
            begin = 0
            for j in range(len(pos[i])):  # nodes in the level i
                if cls_[i, j] == 1 or cls_[i, j] == 2:
                    start = pos[i, j]
                    if i == 0:  # root to level 2
                        for k in range(len(pos[i+1])):
                            if cls_[i+1, k] != 0:
                                end = pos[i+1, k]
                                indices, _ = route_through_array(weight_img, start, end, fully_connected=True, geometric=True)
                                tree, index = self.indices2tree(indices, tree, index)
                            if cls_[i+1, k] == 4:
                                cur = tuple(pos[i+1, k])
                                if cur[0] < 0.1 * imgshape[0]:
                                    tip_z_m.append(cur)
                                elif cur[0] > 0.9 * imgshape[0]:
                                    tip_z_p.append(cur)
                                elif cur[1] < 0.1 * imgshape[1]:
                                    tip_y_m.append(cur)
                                elif cur[1] > 0.9 * imgshape[1]:
                                    tip_y_p.append(cur)
                                elif cur[2] < 0.1 * imgshape[2]:
                                    tip_x_m.append(cur)
                                elif cur[2] > 0.9 * imgshape[2]:
                                    tip_x_p.append(cur)
  
                    else:
                        if cls_[i, j] == 2:
                            for k in range(begin, begin+2):
                                begin += 1
                                if k < len(pos[i+1]):
                                    if cls_[i+1, k] !=0:
                                        end = pos[i+1, k]
                                        indices, _ = route_through_array(weight_img, start, end, fully_connected=True, geometric=True)
                                        tree, index = self.indices2tree(indices, tree, index)
                                    if cls_[i+1, k] ==4:
                                        cur = tuple(pos[i+1, k])
                                        if cur[0] < 0.1 * imgshape[0]:
                                            tip_z_m.append(cur)
                                        elif cur[0] > 0.9 * imgshape[0]:
                                            tip_z_p.append(cur)
                                        elif cur[1] < 0.1 * imgshape[1]:
                                            tip_y_m.append(cur)
                                        elif cur[1] > 0.9 * imgshape[1]:
                                            tip_y_p.append(cur)
                                        elif cur[2] < 0.1 * imgshape[2]:
                                            tip_x_m.append(cur)
                                        elif cur[2] > 0.9 * imgshape[2]:
                                            tip_x_p.append(cur)

        return tree, tip_z_m, tip_z_p, tip_y_m, tip_y_p, tip_x_m, tip_x_p

    # def trace_iter(self, img, seq, imgshape_model):
    #     imgshape = img.shape[2:]
    #     start = tuple(seq[0, 0, 0, :3])
    #     if start[0] < 0.1 * imgshape[0]:
    #         tip_z_p.append(start)
    #     elif cstartur[0] > 0.9 * imgshape[0]:
    #         tip_z_m.append(start)
    #     elif start[1] < 0.1 * imgshape[1]:
    #         tip_y_p.append(start)
    #     elif start[1] > 0.9 * imgshape[1]:
    #         tip_y_m.append(start)
    #     elif start[2] < 0.1 * imgshape[2]:
    #         tip_x_p.append(start)
    #     elif start[2] > 0.9 * imgshape[2]:
    #         tip_x_m.append(start)

    #     if len(tip_x_m) != 0:
    #         for node in tip_x_m:
    #             x_start = imgshape[2] - imgshape_model[2]
    #             x_end = imgshape[2]
    #             y_start = node[1] - imgshape_model[1] / 2
    #             y_end = node[1] + imgshape_model[1] / 2
    #             z_start = node [0] - imgshape_model[0] / 2
    #             z_end = node[0] - imgshape_model[0] / 2
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
                
    #     if len(tip_x_p) != 0:
    #         for node in tip_x_p:
    #             x_start = 0
    #             x_end = imgshape_model[2]
    #             y_start = node[1] - imgshape_model[1] / 2
    #             y_end = node[1] + imgshape_model[1] / 2
    #             z_start = node [0] - imgshape_model[0] / 2
    #             z_end = node[0] - imgshape_model[0] / 2
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
    #     if len(tip_y_m) != 0:
    #         for node in tip_y_m:
    #             x_start = node[2] - imgshape_model[2] / 2
    #             x_end = node[2] + imgshape_model[2] / 2
    #             y_start = imgshape[1] - imgshape_model[1]
    #             y_end = imgshape[1]
    #             z_start = node [0] - imgshape_model[0] / 2
    #             z_end = node[0] - imgshape_model[0] / 2
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
    #     if len(tip_y_p) != 0:
    #         for node in tip_y_p:
    #             x_start = node[2] - imgshape_model[2] / 2
    #             x_end = node[2] + imgshape_model[2] / 2
    #             y_start = 0
    #             y_end = imgshape_model[1]
    #             z_start = node [0] - imgshape_model[0] / 2
    #             z_end = node[0] - imgshape_model[0] / 2
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
    #     if len(tip_z_m) != 0:
    #         for node in tip_z_m:
    #             x_start = node[2] - imgshape_model[2] / 2
    #             x_end = node[2] + imgshape_model[2] / 2
    #             y_start = node[1] - imgshape_model[1] / 2
    #             y_end = node[1] + imgshape_model[1] / 2
    #             z_start = imgshape[0] - imgshape_model[0]
    #             z_end = imgshape[0]
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
    #     if len(tip_z_p) != 0:
    #         for node in tip_z_p:
    #             x_start = node[2] - imgshape_model[2] / 2
    #             x_end = node[2] + imgshape_model[2] / 2
    #             y_start = node[1] - imgshape_model[1] / 2
    #             y_end = node[1] + imgshape_model[1] / 2
    #             z_start = 0
    #             z_end = imgshape_model[0]
    #             crop_img = [z_start:z_end, y_start:y_end, x_start:x_end]
        
    #     tree, tip_x_m = self.trace(crop_img, start)

