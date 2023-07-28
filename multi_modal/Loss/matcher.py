# Refered from https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/matcher.py#L99
import torch
from torch import nn
import torch.nn.functional as F
from groundingdino.util import box_ops
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                cost_class: float = 1,
                cost_bbox: float = 1,
                cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets,positive_map=None,masks=None):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates , cxcywh

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        with torch.no_grad():
            bs, num_queries,token_nums = outputs["pred_logits"].shape
            num_gt = len(positive_map)

            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            tgt_bbox = torch.cat([v[:,1:-1] for v in targets])

            # Compute the token cost From GLIP : token_sigmoid_binary_focal_loss_v2.
            alpha = 0.25
            gamma =2.0
            out_prob = out_prob[:,None,:].expand(bs*num_queries,num_gt,token_nums)
            positive_map = positive_map[None,:,:].expand(bs*num_queries,num_gt,token_nums)

            out_prob_neg_pos = torch.stack([1 - out_prob, out_prob], dim=-1) + 1e-8
            weight = torch.pow(-out_prob_neg_pos + 1.0, gamma)
            focal_zero = - weight[:, :, :, 0] * torch.log(out_prob_neg_pos[:, :, :, 0]) * (1 - alpha)
            focal_one = - weight[:, :, :, 1] * torch.log(out_prob_neg_pos[:, :, :, 1]) * alpha 
            focal = torch.stack([focal_zero, focal_one], dim=-1)
            ce_loss = torch.gather(focal, index=positive_map.long().unsqueeze(-1), dim=-1).squeeze(-1)
            ce_loss = ce_loss.sum(-1)


            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(out_bbox),box_ops.box_cxcywh_to_xyxy(tgt_bbox))

            # 计算mask的代价函数， 随机取N个点，计算其距离

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * ce_loss + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v) for v in targets]
            indices = [ linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_bbox=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)