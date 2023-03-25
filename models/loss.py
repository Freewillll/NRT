import torch
import torch.nn.functional as F
from torch import nn
import sys
import os 

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.matcher import HungarianMatcher, build_matcher

class SetCriterion(nn.Module):

    def __init__(self, num_classes, pad, matcher, weight_dict, weight_pad, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.pad = pad
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = weight_pad
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_nodes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [num_nodes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)     # idx
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])  
        target_classes = torch.full(src_logits.shape[:2], self.pad,     # target classes:  bs, n
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        self.empty_weight_d = self.empty_weight.to(src_logits.device)
        
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_d)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_poses(self, outputs, targets, indices, num_nodes):
        """Compute the losses related to the bounding boxes, the L1 regression loss 
           The target poses are expected in format (z,y,x), normalized by the image size.
        """
        assert 'pred_poses' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_poses = outputs['pred_poses'][idx]
        target_poses = torch.cat([t['poses'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_pos = F.l1_loss(src_poses, target_poses, reduction='none')

        losses = {}
        losses['loss_pos'] = loss_pos.sum() / num_nodes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_nodes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'poses': self.loss_poses,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_nodes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_nodes = sum(len(t["labels"]) for t in targets)
        num_nodes = torch.as_tensor([num_nodes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_nodes))

        return losses
    

if __name__ == '__main__':

    pred_poses = torch.randn(4, 2, 3)
    pred_logits = torch.randn(4, 2, 5)
    target_poses = torch.randn(2, 3)
    target_labels = torch.randint(0, 4, (2,))
    pred = {'pred_logits': pred_logits, 'pred_poses': pred_poses}
    target = [{'labels': target_labels, 'poses': target_poses} for i in range(3)]
    target.append({'labels': torch.randint(0, 4, (1,)), 'poses': torch.randn(1, 3)})
    print(pred)
    print(target)
    matcher = HungarianMatcher()
    losses = ['labels', 'poses']
    weight_dict = {'loss_ce': 1, 'loss_pos': 5}
    criterion = SetCriterion(5, 0, matcher, {1,1}, 0.1, losses)
    loss = criterion(pred, target)
    print(loss)
