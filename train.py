import os
import sys
import argparse
import numpy as np
import time
import json
import SimpleITK as sitk
from einops import rearrange
from tqdm import tqdm
from datetime import timedelta
import skimage.morphology as morphology

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler

from models.loss import SetCriterion
from models.matcher import build_matcher
from models import ntt
from utils import util
from utils.image_util import unnormalize_normal
from datasets.dataset import *

from path_util import *
from file_io import *

parser = argparse.ArgumentParser(
    description='Neuron Tracing Transformer')
# data specific
parser.add_argument('--data_file', default='/PBshare/SEU-ALLEN/Users/Gaoyu/neuronSegSR/Task501_neuron/data_splits.pkl',
                    type=str, help='dataset split file')
# training specific
parser.add_argument('--batch_size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--image_shape', default='32,64,64', type=str,
                    help='Input image shape')
parser.add_argument('--num_item_nodes', default=2, type=int,
                    help='Number of nodes of a item of the seqences')
parser.add_argument('--node_dim', default=4, type=int,
                    help='The dim of nodes in the sequences')
parser.add_argument('--cpu', action="store_true",
                    help='Whether use gpu to train model, default True')
parser.add_argument('--loss_weight', default='1,5',
                    help='The weight of loss_ce and loss_box')
parser.add_argument('--amp', action="store_true",
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='Max gradient norm.')
parser.add_argument('--num_classes', default=5, type=int,
                    help='the nums of classes')
parser.add_argument('--set_cost_class', default=1, type=int,
                    help='cost of classes in matcher')
parser.add_argument('--set_cost_pos', default=1, type=int,
                    help='cost of pos in matcher')
parser.add_argument('--pad', default=0, type=int,
                    help='the class of pad')
parser.add_argument('--weight_pad', default=0.2, type=float,
                    help='the weight of pad class')
parser.add_argument('--weight_loss_poses', default=5, type=float,
                    help='the weight of pos loss')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--step_per_epoch', default=200, type=int,
                    help='step per epoch')
parser.add_argument('--lr_drop', default=200, type=int,
                    help=' lr drop')
parser.add_argument('--warmup_steps', default=200, type=int,
                    help=' warm up steps')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--test_frequency', default=20, type=int,
                    help='frequency of testing')
parser.add_argument('--print_frequency', default=5, type=int,
                    help='frequency of information logging')
parser.add_argument('--local_rank', default=-1, type=int, metavar='N',
                    help='Local process rank')  # DDP required
parser.add_argument('--seed', default=1025, type=int,
                    help='Random seed value')
parser.add_argument('--checkpoint', default='', type=str,
                    help='Saved checkpoint')
parser.add_argument('--evaluation', action='store_true',
                    help='evaluation')
parser.add_argument('--phase', default='train')

# network specific
parser.add_argument('--net_config', default="./models/configs/default_config.json",
                    type=str,
                    help='json file defining the network configurations')

parser.add_argument('--save_folder', default='exps/temp',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


def ddp_print(content):
    if args.is_master:
        print(content)


def draw_seq(img, pos, labels):
    # img: c, z, y, x
    # pos: n, 3
    # cls: n
    img = np.repeat(img, 3, axis=0)
    img[0, :, :, :] = 0
    img[2, :, :, :] = 0
    # keep the position of nodes in the range of imgshape
    # print(seq.shape, pos.shape, cls_.shape, img.shape)
    # print(pos)
    # print(labels)
    nodes = pos.cpu().numpy().copy()
    nodes = np.clip(util.pos_unnormalize(nodes, img.shape[1:]), [0,0,0], [i -1 for i in img.shape[1:]]).astype(int)

    # draw nodes
    for idx, node in enumerate(nodes):
        if labels[idx] == 1: # soma white
            img[:, node[0], node[1], node[2]] = 255
        elif labels[idx] == 2: # branching point yellow
            img[0, node[0], node[1], node[2]] = 255
        elif labels[idx] == 3: # tip blue
            img[2, node[0], node[1], node[2]] = 255
        elif labels[idx] == 4: #boundary blue
            img[2, node[0], node[1], node[2]] = 255
            
    selem = np.ones((1,2,3,3), dtype=np.uint8)
    img = morphology.dilation(img, selem)
    return img


def save_image_in_training(imgfiles, img, targets, pred, epoch, phase, idx):  
    # the shape of image: b, c, z, y, x
    # targets: {'labels', 'poses'}   
    # pred: {'pred_logits', 'pred_poses'}  logtis: b, n, 5  poses: b, n, 3
    
    imgfile = imgfiles[idx]
    prefix = get_file_prefix(imgfile)
    with torch.no_grad():
        img = (unnormalize_normal(img[idx].numpy())).astype(np.uint8)
        # -> n, nodes, dim
        tgt_cls = targets[idx]['labels'].clone()
        tgt_pos = targets[idx]['poses'].clone()
        
        img_lab = draw_seq(img, tgt_pos, tgt_cls)
        
        if phase == 'train':
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
        else:
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
            
        save_image(os.path.join(args.save_folder, out_lab_file), img_lab)
            
        if pred != None:
            pred_cls = torch.argmax(pred['pred_logits'][idx], dim=-1)
            pred_pos = pred['pred_poses'][idx].clone()
            
            img_pred = draw_seq(img, pred_pos, pred_cls)

            if phase == 'train':
                out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'
            else:
                out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'

            save_image(os.path.join(args.save_folder, out_pred_file), img_pred)


def validate(model, criterion ,val_loader, weight_dict, epoch, debug=True, num_image_save=5, phase='val'):
    model.eval()
    num_saved = 0
    loss_all = 0
    if num_image_save == -1:
        num_image_save = 9999
        
    loss_ce = 0
    loss_pos = 0

    processed = 0
    for img, targets, imgfiles, swcfiles in val_loader:
        processed += 1

        img_d = img.to(args.device)
        targets_d = [{'labels': v['labels'].to(args.device), 'poses': v['poses'].to(args.device)} for v in targets]
        
        if args.amp:
            with autocast():
                with torch.no_grad():
                    pred = model(img_d)
        
                    loss_dict = criterion(pred, targets_d)
                    loss_ce += loss_dict['loss_ce']
                    loss_pos += loss_dict['loss_pos']
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)        
                    loss_all += losses  
        else:
            with torch.no_grad():
                pred = model(img_d)
                loss_dict = criterion(pred, targets_d)
                loss_ce += loss_dict['loss_ce']
                loss_pos += loss_dict['loss_pos']
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)        
                loss_all += losses  
                

        del img_d
        del targets_d

        if debug:
            for debug_idx in range(img.size(0)):
                num_saved += 1
                if num_saved > num_image_save:
                    break
                save_image_in_training(imgfiles, img, targets, pred, epoch, phase, debug_idx)
    
    loss_ce_mean = loss_ce / processed
    loss_pos_mean = loss_pos / processed            
    loss_mean = loss_all / processed
    return loss_ce_mean.item(), loss_pos_mean.item(), loss_mean.item()


def load_dataset(phase, imgshape):
    dset = GenericDataset(args.data_file, phase=phase, imgshape=imgshape)
    ddp_print(f'Number of {phase} samples: {len(dset)}')
    # distributedSampler
    if phase == 'train':
        sampler = RandomSampler(dset) if args.local_rank == -1 else DistributedSampler(dset, shuffle=True)
    else:
        sampler = RandomSampler(dset) if args.local_rank == -1 else DistributedSampler(dset, shuffle=False)

    loader = tudata.DataLoader(dset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False, pin_memory=True,
                               sampler=sampler,
                               drop_last=True,
                               collate_fn=collate_fn,
                               worker_init_fn=util.worker_init_fn)
    dset_iter = iter(loader)
    return loader, dset_iter


def evaluate(model, optimizer, imgshape, phase):
    val_loader, val_iter = load_dataset(phase, imgshape)
    args.curr_epoch = 0
    loss_ce, loss_dice, *_ = validate(model, val_loader, epoch=0, debug=True, num_image_save=-1,
                                        phase=phase)
    ddp_print(f'Average loss_ce and loss_dice: {loss_ce:.5f} {loss_dice:.5f}')


def train(model, optimizer, imgshape):
    # dataset preparing
    train_loader, train_iter = load_dataset('train', imgshape)
    # val_loader, val_iter = load_dataset('val', imgshape)
    args.step_per_epoch = len(train_loader) if len(train_loader) < args.step_per_epoch else args.step_per_epoch
    t_total = args.max_epochs * args.step_per_epoch
    lr_scheduler = util.WarmupLinearSchedule(optimizer, args.warmup_steps, t_total)

    # training process
    model.train()
    t0 = time.time()
    # for automatic mixed precision
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    best_accuracy = 0
    
    weight_dict = {'loss_ce': 1, 'loss_pos': args.weight_loss_poses}
    losses = ['labels', 'poses']
    matcher = build_matcher(args)
    criterion = SetCriterion(num_classes=args.num_classes, pad=args.pad, matcher=matcher, weight_dict=weight_dict, weight_pad=args.weight_pad, losses=losses)
    
    for epoch in range(args.max_epochs):
        # push the epoch information to global namespace args
        args.curr_epoch = epoch

        epoch_iterator = tqdm(train_loader,
                        desc=f'Epoch {epoch + 1}/{args.max_epochs}',
                        total=args.step_per_epoch,
                        postfix=dict,
                        dynamic_ncols=True,
                        disable=args.local_rank not in [-1, 0],
                        mininterval=5,
                        delay=5)

        for step, batch in enumerate(epoch_iterator):
            img, targets, imgfiles, swcfiles = batch

            img_d = img.to(args.device)
            targets_d = [{'labels': v['labels'].to(args.device), 'poses': v['poses'].to(args.device)} for v in targets]

            loss_tmp = {}

            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    pred = model(img_d)
                    loss_dict = criterion(pred, targets_d)
                    loss_tmp = loss_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                    del img_d
                grad_scaler.scale(losses).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                lr_scheduler.step()
            else:
                pred = model(img_d)
                loss_dict = criterion(pred, targets)
                loss_tmp = loss_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)              
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()

            # train statistics for bebug afterward
            # if step % args.print_frequency == 0:
            #     ddp_print(
            #         f'[{epoch}/{step}] loss_ce={loss_ce:.5f}, loss_box={loss_box:.5f}, accuracy_cls={accuracy_cls:.3f}, accuracy_pos={accuracy_pos:.3f}, time: {time.time() - t0:.4f}s')

            epoch_iterator.set_postfix({'loss_ce': loss_tmp['loss_ce'].item(), 'loss_pos': loss_tmp['loss_pos'].item()})

        # do validation
        if args.test_frequency != 0 and epoch !=0 and epoch % args.test_frequency == 0:
            val_loader, val_iter = load_dataset('val', imgshape)
            ddp_print('Evaluate on val set')
            loss_ce_val, loss_pos_val, loss_val = validate(model, criterion, val_loader, weight_dict, epoch, debug=debug,
                                                            phase='val')

            model.train()  # back to train phase
            ddp_print(f'[Val_{epoch}] average loss_ce loss_pos loss_all are {loss_ce_val}  {loss_pos_val}  {loss_val}')
            # save the model
            if args.is_master:
                # save current model
                torch.save(model, os.path.join(args.save_folder, 'final_model.pt'))

        # save image for subsequent analysis
        if debug and args.is_master and epoch % args.test_frequency == 0:
            save_image_in_training(imgfiles, img, targets, pred, epoch, 'train', debug_idx)


def main():
    # keep track of master, useful for IO
    args.is_master = args.local_rank in [0, -1]

    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device

    if args.deterministic:
        util.set_deterministic(deterministic=True, seed=args.seed)

    # for output folder
    if args.is_master and not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    # Network
    with open(args.net_config) as fp:
        net_configs = json.load(fp)
        print('Network configs: ', net_configs)
        model = ntt.NTT(**net_configs)
        ddp_print('\n' + '=' * 10 + 'Network Structure' + '=' * 10)
        ddp_print(model)
        ddp_print('=' * 30 + '\n')

    model = model.to(args.device)
    if args.checkpoint:
        # load checkpoint
        ddp_print(f'Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location={'cuda:0': f'cuda:{args.local_rank}'})
        model.load_state_dict(checkpoint.module.state_dict())
        del checkpoint
        # if args.is_master:
        #    torch.save(checkpoint.module.state_dict(), "exp040.state_dict")
        #    sys.exit()

    # convert to distributed data parallel model
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank],
                output_device=args.local_rank)  # , find_unused_parameters=True)

    # optimizer & loss
    if args.checkpoint:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.imgshape = tuple(map(int, args.image_shape.split(',')))
    loss_weight = list(map(float, args.loss_weight.split(',')))
    # sum_weights = sum(loss_weight)
    # loss_weight = [w / sum_weights for w in loss_weight]

    # Print out the arguments information
    ddp_print('Argument are: ')
    ddp_print(f'   {args}')

    if args.evaluation:
        evaluate(model, optimizer, args.imgshape, args.phase)
    else:
        train(model, optimizer, args.imgshape)


if __name__ == '__main__':
    main()
