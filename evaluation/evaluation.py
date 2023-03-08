import numpy as np
import torch
import torch.nn.functional as F
from skimage.graph import route_through_array
import math,sys,os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import utils.util as util
from datasets.dataset import *
from train import *

class Tracer(object):
    def __init__(self, model, max_len, nodes):
        self.model = model
        self.max_len = max_len
        self.nodes = nodes
        self.model.eval()

    def indices2tree(self, indices, tree, index):
        for idx, node in enumerate(indices, start=index+1):
            tree.append((idx, 2, node[2], node[1], node[0], 1, index))
            index += 1
        return tree, index

    def pos2points(self, pos, cls_):
        points = []
        pos = rearrange(pos, 'n nodes dim -> (n nodes) dim')
        cls_ = rearrange(cls_, 'n nodes -> (n nodes)')
        p_idx = -1
        for idx, node in enumerate(pos, start=1):
            z, y, x = node[0], node[1], node[2]
            type_ = cls_[idx-1]
            if type_ != 0 and type_ != EOS:
                points.append((idx, type_, x, y, z, 1, p_idx))
        return points
    
    def test(self, img, seq, cls_, crit_ce, crit_box, model, nodes, loss_weight):
        with torch.no_grad():
            loss_ce, loss_box, loss, accuracy_cls, accuracy_pos, pred = get_forward(img, seq, cls_, crit_ce, crit_box, model, nodes, loss_weight)
            print(f'loss_ce: {loss_ce}, loss_box: {loss_box}, accuracy_cls: {accuracy_cls}, accuracy_pos: {accuracy_pos}')
            pos = util.pos_unnormalize(pred[..., :3], img.shape[2:])[0]
            pred_cls = torch.argmax(pred[..., 3:], dim=-1)
            print(pos)
            print(pred_cls)
            

    def trace_from_soma(self, img, seq, cls_):
        # pred: b, n, nodes, dims
        # cls_: b, n, nodes
        with torch.no_grad():
            for i in range(self.max_len):
                pred = self.model(img, seq)
                pred = rearrange(pred, 'b n (nodes dim) -> b n nodes dim', nodes=self.nodes)
                output = pred[:, -1, ...].clone().unsqueeze(1)
                print(output)
                # pre_cls: b, n, nums
                pred_cls = torch.argmax(output[..., 3:], dim=-1)
                if pred_cls[0, 0, 0] == EOS:
                    break
                else:
                    # normalize
                    pred_cls_normalize = (pred_cls - NODE_PAD) / (EOS - NODE_PAD)
                    pred_seq = torch.cat((output[..., :3], pred_cls_normalize.unsqueeze(3)), dim=-1)
                    cls_ = torch.cat((cls_, pred_cls), dim=1)
                    seq = torch.cat((seq, pred_seq), dim=1)
                    print(seq)

            pos = util.pos_unnormalize(seq[..., :3], img.shape[2:])[0]
            cls_ = cls_[0]
            points = self.pos2points(pos.clone().cpu().numpy(), cls_.clone().cpu().numpy())
            write_swc(points, 'debug/debug_pos.swc')
            pos = torch.clamp(pos, torch.tensor([0,0,0]).to(args.device), torch.tensor([i -1 for i in img.shape[2:]]).to(args.device)).type(torch.int64)
            
            # pos: n, nodes, dims,  (z, y, x)
            # cls: n, nodes
            imgshape = img.shape[2:]
            tip_z_m, tip_z_p, tip_y_m, tip_y_p, tip_x_m, tip_x_p= [], [], [], [], [], []
            tree = []
            weight_img = 255 - img
            weight_img = weight_img.cpu().numpy()
            index = 1
            for i in range(len(pos) - 1):  # level i
                begin = 0
                for j in range(len(pos[i])):  # nodes in the level i
                    if cls_[i, j] == 1 or cls_[i, j] == 2:
                        start = pos[i, j].clone().cpu().numpy()
                        if i == 0:  # root to level 2
                            for k in range(len(pos[i+1])):
                                if cls_[i+1, k] != 0:
                                    end = pos[i+1, k].clone().cpu().numpy()
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
                                            end = pos[i+1, k].clone().cpu().numpy()
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

if __name__ == '__main__':
    split_file = '/PBshare/SEU-ALLEN/Users/Gaoyu/Neuron_dataset/Task002_ntt_256/data_splits.pkl'
    idx = 6
    imgshape = (32, 64, 64)
    dataset = GenericDataset(split_file, 'test', imgshape=imgshape, seq_node_nums=10, node_dim=4)

    args.is_master = 1
    args.local_rank = -1
    args.device = torch.device("cuda")
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print('Network configs: ', net_configs)
        model = ntt.NTT(**net_configs)
        ddp_print('\n' + '=' * 10 + 'Network Structure' + '=' * 10)
        ddp_print(model)
        ddp_print('=' * 30 + '\n')
    model = model.to(args.device)
    args.checkpoint = './exps/exp013/final_model.pt'
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
    model.load_state_dict(checkpoint.state_dict())
    
    crit_ce = nn.CrossEntropyLoss(ignore_index=SEQ_PAD, reduction='none').to(args.device)
    crit_box = nn.MSELoss(reduction='none').to(args.device)

    tracer = Tracer(model, 8, 10)

    loader = tudata.DataLoader(dataset, 1, 
                                num_workers=8, 
                                shuffle=False, pin_memory=True,
                                drop_last=True, 
                                collate_fn=collate_fn,
                                worker_init_fn=util.worker_init_fn)
    for i, batch in enumerate(loader):
        img, seq , cls_, imgfiles, swcfile = batch
        img = img.to(args.device)
        seq = seq.to(args.device)
        cls_ = cls_.to(args.device)
        print(img.shape)
        print(seq.shape)
        print(cls_.shape)
        img_s = img[0].cpu().numpy()
        print(img_s.shape)
        print(seq)
        save_image(f'./debug/debug_{i}.v3draw', img_s)
        tracer.test(img, seq, cls_, crit_ce, crit_box, model, 10, [1, 5])
        # write_swc(tree, f'./debug/debug_{i}.swc')
        break
