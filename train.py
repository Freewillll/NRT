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

import torch
import torch.nn as nn
import torch.utils.data as tudata
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.distributed as distrib
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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
parser.add_argument('--num_item_nodes', default='8', type=int,
                    help='Number of nodes of a item of the seqences')
parser.add_argument('--node_dim', default='4', type=int,
                    help='The dim of nodes in the sequences')
parser.add_argument('--cpu', action="store_true",
                    help='Whether use gpu to train model, default True')
parser.add_argument('--amp', action="store_true",
                    help='Whether to use AMP training, default True')
parser.add_argument('--lr', '--learning-rate', default=3e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0, type=float,
                    help='Weight decay')
parser.add_argument('--decay_type', choices=['cosine', 'linear'], default='cosine', type=str,
                    help='How to decay the learning rate')
parser.add_argument('--warmup_steps', default=500, type=str,
                    help='Step of training to perform learning rate warmup for')
parser.add_argument('--max_grad_norm', default=1.0, type=float,
                    help='Max gradient norm.')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='maximal number of epochs')
parser.add_argument('--step_per_epoch', default=200, type=int,
                    help='step per epoch')
parser.add_argument('--deterministic', action='store_true',
                    help='run in deterministic mode')
parser.add_argument('--test_frequency', default=3, type=int,
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


# def translate(imgfiles, img)


def draw_seq(img, seq, cls_, pos):
    # img: c, z, y, x
    # seq: n, nodes, dim
    # pos: n, 3
    # cls: n
    img = np.repeat(img, 3, axis=0)
    # keep the position of nodes in the range of imgshape
    start = np.clip(seq.numpy()[0,0,:3], [0,0,0], [i -1 for i in img.shape[1:]]).astype(int)
    nodes = np.clip(pos.numpy(), [0,0,0], [i -1 for i in img.shape[1:]]).astype(int)
    # draw nodes
    img[1, start[0], start[1], start[2]] = 255  # start point
    for idx, node in enumerate(nodes):
        if cls_[idx] == 1: # root green
            img[1, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 2: # branching point red
            img[0, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 3: # tip yellow
            img[:2, node[0], node[1], node[2]] = 255
        elif cls_[idx] == 4:
            img[2, node[0], node[1], node[2]] = 255
    selem = np.ones((1,2,3,3), dtype=np.uint8)
    img = morphology.dilation(img, selem)
    return img


def save_image_in_training(imgfiles, img, seq, cls_, pred, epoch, phase, idx):  
    # the shape of image: b, c, z, y, x
    # cls: b, n, nodes
    imgfile = imgfiles[idx]
    prefix = get_file_prefix(imgfile)
    with torch.no_grad():
        img = (unnormalize_normal(img[idx].numpy())[0]).astype(np.uint8)
        # -> n, nodes, dim
        trg, cls_, pred = seq[0, 1:], cls_[0, 1:], pred[0]
        # add start points
        pred_cls = torch.argmax(pred[..., 3:], dim=1)
        pred = pred[torch.where(pred_cls > 0)]  # n, dim
        trg = trg[torch.where(cls_ > 0)]
        img_pred = draw_seq(img, seq[0], pred_cls, pred[..., :3])
        img_lab = draw_seq(img, seq[0], cls_, trg[..., :3])

        if phase == 'train':
            out_lab_file = f'debug_epoch{epoch}_{prefix}_{phase}_lab.v3draw'
            out_pred_file = f'debug_epoch{epoch}_{prefix}_{phase}_pred.v3draw'
        else:
            out_lab_file = f'debug_{prefix}_{phase}_lab.v3draw'
            out_pred_file = f'debug_{prefix}_{phase}_pred.v3draw'

        save_image(os.path.join(args.save_folder, out_lab_file), img_lab)
        save_image(os.path.join(args.save_folder, out_pred_file), img_pred)


def get_forward(img, seq, cls_, crit_ce, crit_box, model, nodes):
    # trg: b, n, nodes, dim (add EOS)
    # cls: b, n, nodes
    src, trg = seq[:, :-1, ...],  seq[:, 1:, ...]
    cls_ = cls_[:, 1:, :]
    #outputs  b, n, node * (pos + cls)
    pred = model(img, src)
    pred = rearrange(pred, 'b n (nodes dim) -> b n nodes dim', nodes=nodes)
    pred_pos, pred_cls = pred[..., :3], pred[..., 3:]
    # -> b, cls, nodes, n
    pred_cls_t = pred_cls.contiguous().transpose(-1, 1)
    trg_pos = trg[..., :3]
    # -> b, nodes, n
    trg_cls = cls_.contiguous().transpose(-1, -2)
    # b, n, nodes
    mask = cls_ != 0
    accuracy_cls, accuracy_pos = util.accuracy_withmask(pred_cls_t.clone(), pred_pos.clone(), trg_cls.clone(), trg_pos.clone(), mask.clone(), img.shape)
    # b, n, nodes, 3
    pos_mask = mask.unsqueeze(3).repeat(1, 1, 1, 3)
    loss_ce, loss_box = crit_ce(pred_cls_t, trg_cls), crit_box(pred_pos, trg_pos)
    loss_mask_box = (loss_box * pos_mask).sum() / pos_mask.sum()
    loss = loss_ce + loss_mask_box
    return loss_ce, loss_mask_box, loss, accuracy_cls, accuracy_pos, pred


def get_forward_eval(img, seq, cls_, crit_ce, crit_box, model, nodes):
    if args.amp:
        with autocast():
            with torch.no_grad():
                loss_ce, loss_box, loss = get_forward(img, seq, cls_, crit_ce, crit_box, model, nodes)
    else:
        with torch.no_grad():
            loss_ce, loss_box, loss = get_forward(img, seq, cls_, crit_ce, crit_box, model, nodes)
    return loss_ce, loss_box, loss


def validate(model, val_loader, crit_ce, crit_box, epoch, debug=True, num_image_save=10, phase='val'):
    model.eval()
    num_saved = 0
    if num_image_save == -1:
        num_image_save = 9999

    losses = []
    processed = -1
    for img, seq, cls_, imgfiles, swcfiles in val_loader:
        processed += 1

        img_d = img.to(args.device)
        seq_d = seq.to(args.device)
        cls_d = cls_.to(args.device)
        if phase == 'val':
            loss_ce, loss_box, loss = get_forward_eval(img_d, seq_d, cls_d, crit_ce, crit_box, model, args.num_item_nodes)

        else:
            raise ValueError

        del img_d
        del lab_d

        losses.append([loss_ce, loss_box, loss.item()])

        if debug:
            for debug_idx in range(img.size(0)):
                num_saved += 1
                if num_saved > num_image_save:
                    break
                save_image_in_training(imgfiles, img, lab, logits, epoch, phase, debug_idx)

    losses = torch.from_numpy(np.array(losses)).to(args.device)
    distrib.all_reduce(losses, op=distrib.ReduceOp.SUM)
    losses = losses.mean(dim=0) / distrib.get_world_size()

    return losses


def load_dataset(phase, imgshape):
    dset = GenericDataset(args.data_file, phase=phase, imgshape=imgshape, seq_node_nums=args.num_item_nodes, node_dim=args.node_dim)
    ddp_print(f'Number of {phase} samples: {len(dset)}')
    # distributedSampler
    if phase == 'train':
        sampler = DistributedSampler(dset, shuffle=True)
    else:
        sampler = DistributedSampler(dset, shuffle=False)

    loader = tudata.DataLoader(dset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=False, pin_memory=True,
                               sampler=sampler,
                               drop_last=True,
                               collate_fn=collate_fn,
                               worker_init_fn=util.worker_init_fn)
    dset_iter = iter(loader)
    return loader, dset_iter


def evaluate(model, optimizer, crit_ce, crit_box, imgshape, phase='test'):
    val_loader, val_iter = load_dataset(phase, imgshape)
    args.curr_epoch = 0
    loss_ce, loss_dice, loss = validate(model, val_loader, crit_ce, crit_box, epoch=0, debug=True, num_image_save=-1,
                                        phase=phase)
    ddp_print(f'Average loss_ce and loss_dice: {loss_ce:.5f} {loss_dice:.5f}')


def train(model, optimizer, crit_ce, crit_box, imgshape):
    # dataset preparing
    train_loader, train_iter = load_dataset('train', imgshape)
    val_loader, val_iter = load_dataset('val', imgshape)
    t_total = args.max_epochs * args.step_per_epoch
    if args.decay_type == "cosine":
        scheduler = util.WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = util.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    # training process
    model.train()
    t0 = time.time()
    # for automatic mixed precision
    grad_scaler = GradScaler()
    debug = True
    debug_idx = 0
    for epoch in range(args.max_epochs):
        # push the epoch information to global namespace args
        args.curr_epoch = epoch

        avg_loss_ce = 0
        avg_loss_box = 0

        epoch_iterator = tqdm(train_loader,
                        desc=f'Epoch {epoch + 1}/{args.max_epochs}',
                        total=args.step_per_epoch,
                        postfix=dict,
                        dynamic_ncols=True,
                        disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator):
            img, seq, cls_, imgfiles, swcfiles = batch

            img_d = img.to(args.device)
            seq_d = seq.to(args.device)
            cls_d = cls_.to(args.device)

            optimizer.zero_grad()
            if args.amp:
                with autocast():
                    loss_ce, loss_box, loss, accuracy_cls, accuracy_pos, pred = get_forward(img_d, seq_d, cls_d, crit_ce, crit_box, model, args.num_item_nodes)
                    del img_d
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()
            else:
                loss_ce, loss_box, loss, accuracy_cls, accuracy_pos = get_forward(img_d, seq_d, cls_d, crit_ce, crit_box, model, args.num_item_nodes)
                del img_d
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            avg_loss_ce += loss_ce
            avg_loss_box += loss_box

            # train statistics for bebug afterward
            if step % args.print_frequency == 0:
                ddp_print(
                    f'[{epoch}/{step}] loss_ce={loss_ce:.5f}, loss_box={loss_box:.5f}, accuracy_cls={accuracy_cls:.3f}, accuracy_pos={accuracy_pos:.3f}, time: {time.time() - t0:.4f}s')

            epoch_iterator.set_postfix({'loss_ce': loss_ce, 'loss_box': loss_box, 'accuracy_cls': accuracy_cls, 'accuracy_pos': accuracy_pos})

        avg_loss_ce /= args.step_per_epoch
        avg_loss_box /= args.step_per_epoch

        # do validation
        if epoch % args.test_frequency == 0:
            ddp_print('Evaluate on val set')
            val_loss_ce, val_loss_box, val_loss, val_accuracy_cls, val_accuracy_pos = validate(model, val_loader, crit_ce, crit_box, epoch, debug=debug,
                                                            phase='val')
            model.train()  # back to train phase
            ddp_print(f'[Val{epoch}] average ce loss, box loss and the sum are {val_loss_ce:.5f}, {val_loss_box:.5f}, {val_loss:.5f},\
                 cls accuracy and pos accuracy are {val_accuracy_cls:.3f}, {val_accuracy_pos:.3f}')
            # save the model
            if args.is_master:
                # save current model
                torch.save(model, os.path.join(args.save_folder, 'final_model.pt'))

        # save image for subsequent analysis
        if debug and args.is_master and epoch % args.test_frequency == 0:
            save_image_in_training(imgfiles, img, seq, cls_, pred, epoch, 'train', debug_idx)


def main():
    # keep track of master, useful for IO
    args.is_master = args.local_rank == 0

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
        args.lr /= 5
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)

    crit_ce = nn.CrossEntropyLoss(ignore_index=ntt.PAD).to(args.device)
    crit_box = nn.MSELoss(reduction='none').to(args.device)
    args.imgshape = tuple(map(int, args.image_shape.split(',')))

    # Print out the arguments information
    ddp_print('Argument are: ')
    ddp_print(f'   {args}')

    if args.evaluation:
        evaluate(model, optimizer, crit_ce, crit_box, args.imgshape, args.phase)
    else:
        train(model, optimizer, crit_ce, crit_box, args.imgshape)


if __name__ == '__main__':
    main()
