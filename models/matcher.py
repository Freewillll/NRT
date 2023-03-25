import torch
from scipy.optimize import linear_sum_assignment
from torch import nn



class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_pos: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_pos: This is the relative weight of the L1 error of the node coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pos = cost_pos
        assert cost_class != 0 or cost_pos != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_nodes, num_classes] with the classification logits
                 "pred_pos": Tensor of dim [batch_size, num_nodes, 3] with the predicted node coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_nodes]
                 "poses": Tensor of dim [num_target_nodes, 3]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_pred_nodes, num_target_nodes)
        """
        bs, num_nodes = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_nodes, num_classes]
        out_pos = outputs["pred_poses"].flatten(0, 1)  # [batch_size * num_nodes, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])   # bs * num_target_nodes
        tgt_pos = torch.cat([v["poses"] for v in targets])    # bs * num_target_nodes * 3

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_pos = torch.cdist(out_pos, tgt_pos, p=1)

        # Final cost matrix
        
        sizes = [len(v["poses"]) for v in targets] # bs, 1

        C = self.cost_pos * cost_pos + self.cost_class * cost_class
        C = C.view(bs, num_nodes, -1).cpu()   # bs, num_nodes, bs * target_node_nums

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_pos=args.set_cost_pos)


if __name__ == '__main__':

    pred_poses = torch.randn(4, 2, 3)
    pred_logits = torch.randn(4, 2, 6)
    target_poses = torch.randn(2, 3)
    target_labels = torch.randint(0, 5, (2,))
    pred = {'pred_logits': pred_logits, 'pred_poses': pred_poses}
    target = [{'labels': target_labels, 'poses': target_poses} for i in range(4)]
    print(pred)
    print(target)
    matcher = HungarianMatcher()
    output = matcher(pred, target)
    print(output)
    for (_, J) in output:
        print(J)