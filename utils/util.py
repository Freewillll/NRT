import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import math


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


def init_device(device_name):
    if type(device_name) == int:
        if torch.cuda.is_available():
            device = torch.device(device_name)
        else:
            raise EnvironmentError("GPU is not accessible!")
    elif type(device_name) == str:
        if device_name == 'cpu':
            device = torch.device(device_name)
        elif device_name[:4] == 'cuda':
            if torch.cuda.is_available():
                device = torch.device(device_name)
        else:
            raise ValueError("Invalid name for device")
    else:
        raise NotImplementedError
    return device


def set_deterministic(deterministic=True, seed=1024):
    if deterministic:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return True


def worker_init_fn(worker_id):
    """Function to avoid numpy.random seed duplication across multi-threads"""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def pos_unnormalize(pos, image_shape):
    # pos: b, n, nodes, 3
    #image: b, c, z, y, x
    for i in range(pos.shape[-1]):
        or1, or2 = 0, image_shape[i+2]
        m1, m2 = pos[:, :, :, i].min(), pos[:, :, :, i].max()
        pos[..., i] = (pos[..., i] - m1) * (or2 - or1) / (m2 - m1 + 1e-8) + or1
    return pos


def accuracy_withmask(pred_cls, pred_pos, lab_cls, lab_pos, mask, image_shape):
    # pred_cls: b, cls, nodes, n
    # mask: b, n, nodes
    pred_cls = torch.argmax(pred_cls, dim=1)
    cls_mask = mask.contiguous().transpose(-1, -2)
    pred_cls, lab_cls = torch.masked_select(pred_cls, cls_mask), torch.masked_select(lab_cls, cls_mask)
    accuracy_cls = (pred_cls == lab_cls).mean()
    pos_mask = mask.unsqueeze(3).repeat(1,1,1,3)
    pred_pos, lab_pos = pos_unnormalize(pred_pos, image_shape), pos_unnormalize(lab_pos, image_shape)
    pred_pos, lab_pos = torch.masked_select(pred_pos, pos_mask).view(-1, 3), torch.masked_select(lab_pos, pos_mask).view(-1, 3)
    dist = torch.linalg.norm(pred_pos - lab_pos, dim=-1)
    accuracy_pos = (dist <= 5).mean()
    return accuracy_cls, accuracy_pos


def accuracy_nomask(pred_cls, pred_pos, lab_cls, lab_pos, image_shape):
    # pred_cls: b, cls, nodes, n
    # mask: b, n, nodes
    pred_cls = torch.argmax(pred_cls, dim=1)
    accuracy_cls = (pred_cls == lab_cls).mean()
    pred_pos, lab_pos = pos_unnormalize(pred_pos, image_shape), pos_unnormalize(lab_pos, image_shape)
    dist = torch.linalg.norm(pred_pos - lab_pos, dim=-1)
    accuracy_pos = (dist <= 5).mean()
    return accuracy_cls, accuracy_pos

