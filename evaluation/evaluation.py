import numpy as np
import torch
import torch.nn.functional as F
from skimage.graph import route_through_array
import math

import utils.util as util
from datasets.dataset import *

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

    def trace(self, img, seq):
        # pred: b, n, nodes, dims
        # cls_: b, n, nodes
        cls_ = seq[..., -1]
        for i in range(self.max_len):
            pred = self.model(img, seq)
            output = pred[:, -1, ...]
            # pre_cls, b, n, nums
            pred_cls = torch.argmax(output[..., 3:], dim=-1)
            cls_ = torch.concatenate((cls_, pred_cls.clone()), dim=1)
            if pred_cls[0, 0, 0] == EOS:
                pos = util.pos_unnormalize(seq, img.shape[2:])
                return pos, cls_
            else:
                # normalize
                pred_cls = (pred_cls - NODE_PAD) / (EOS - NODE_PAD + 1e-8)
                seq = torch.concatenate((output[..., :3], pred_cls.unsqueeze(3)), axis=-1)
        pos = util.pos_unnormalize(seq, img.shape[2:])[0]
        cls_ = cls_[0]
        
        # pos: n, nodes, dims,  (z, y, x)
        # cls: n, nodes
        imgshape = img.shape[2:]
        tip_z_m, tip_z_p, tip_y_m, tip_y_p, tip_x_m, tip_x_p= [], [], [], [], [], []
        tree = []
        weight_img = 255 - img
        level = 0
        index = 1
        for i in range(len(pos) - 1):
            for j in range(len(pos[i])):
                begin = 0
                if cls_[i, j] != 0:
                    start = pos[i, j]

                    if level == 0:
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
                            for k in range(2, start=begin):
                                begin += 1
                                if k < len(pos[i+1])
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

            level += 1
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

