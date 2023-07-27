# The matcher is refered from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py#L483)
# And the multi-modal loss is from GLIP
try:
    from .matcher import build_matcher
except:
    from matcher import build_matcher

import torch
import torch.nn.functional as F
from torch import nn
from groundingdino.util import box_ops
from groundingdino.util.misc import interpolate,nested_tensor_from_tensor_list
# from torch.cuda.amp import custom_fwd

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss.mean()

def dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

def token_sigmoid_binary_focal_loss_v2(pred_logits, targets, alpha, gamma, text_mask=None):
    # Copy From From GLIP 
    assert (targets.dim() == 3)
    assert (pred_logits.dim() == 3)  # batch x from x to

    if text_mask is not None:
        assert (text_mask.dim() == 2)

    # We convert everything into binary
    out_prob = pred_logits.sigmoid()
    out_prob_neg_pos = torch.stack([1 - out_prob, out_prob], dim=-1) + 1e-8  # batch x boxes x 256 x 2
    weight = torch.pow(-out_prob_neg_pos + 1.0, gamma)

    focal_zero = - weight[:, :, :, 0] * torch.log(out_prob_neg_pos[:, :, :, 0]) * (
            1 - alpha)  # negative class
    focal_one = - weight[:, :, :, 1] * torch.log(out_prob_neg_pos[:, :, :, 1]) * alpha  # positive class
    focal = torch.stack([focal_zero, focal_one], dim=-1)
    loss_ce = torch.gather(focal, index=targets.long().unsqueeze(-1), dim=-1)
    return loss_ce

class TokenSigmoidFocalLoss(nn.Module):
    # copied from GLIP https://github.com/microsoft/GLIP
    def __init__(self, alpha, gamma):
        super(TokenSigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets, text_masks=None, version="binary", **kwargs):
        if version == "binary":
            loss_func = token_sigmoid_binary_focal_loss_v2
        # elif version == "softmax":
        #     loss_func = token_sigmoid_softmax_focal_loss
        # elif version == "binaryv2":
        #     loss_func = token_sigmoid_binary_focal_loss_v2
        # else:
        #     raise NotImplementedError
        loss = loss_func(logits, targets, self.alpha, self.gamma, text_masks, **kwargs)
        return loss.sum()

# INF = 1e8 # Only used for GLIP prepare_target
class ATSSLossComputation(torch.nn.Module):
    def __init__(self, args):
        super(ATSSLossComputation, self).__init__()
        self.args = args
        self.matcher = build_matcher(args)
        self.token_loss_func = TokenSigmoidFocalLoss(args.FUSE_TOKEN_ALPHA,
                                                         args.FUSE_TOKEN_GAMMA)

    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, outputs, targets,positive_map=None,masks=None):
        indices = self.matcher(outputs, targets,positive_map,masks)
        losses = {}
        losses.update(self.loss_boxes(outputs, targets, indices))
        losses.update(self.loss_token(outputs, positive_map, indices))
        losses.update(self.loss_masks(outputs,masks,indices))

        
        sum_loss = sum([value * self.args[f"{key}_weight"] for key,value in losses.items()])
        losses = {key:value.data for key,value in losses.items()}
        return losses,sum_loss
    
    def loss_masks(self, outputs, target_masks, indices):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs


        src_masks = outputs["pred_masks"]

        
        # upsample predictions to the target size
        
        # tg_masks = []
        # for target_mask in target_masks:
        #     target_mask = target_mask.to(src_masks)
        #     target_mask = target_mask.sum(dim=0)
        #     target_mask[target_mask!=0] = 1
        #     tg_masks.append(target_mask)
        # target_masks = torch.stack(tg_masks)
        # import pdb;pdb.set_trace()
        target_masks, valid = nested_tensor_from_tensor_list(target_masks).decompose()
        target_masks = target_masks.to(src_masks)
        # version2 or 3
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = src_masks[src_idx]
        target_masks = target_masks[tgt_idx]

        import pdb;pdb.set_trace()

        # version 4
        # src_masks = src_masks.squeeze(1)
        # target_masks = target_masks
        # import pdb;pdb.set_trace()


        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0]



        src_masks = src_masks.flatten(1) 
        target_masks = target_masks.flatten(1)

        # version 1
        # start_idx = 0
        # _target_masks = []
        # for _,i in indices:
        #     end_idx = start_idx+len(i)
        #     _target_masks.append(target_masks[start_idx:end_idx][i])
        #     start_idx = end_idx
        # target_masks = torch.cat(_target_masks, dim=0).flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks),
            "loss_dice": dice_loss(src_masks, target_masks),
        }
        return losses
        
    
    def loss_token(self,outputs, positive_map, indices,text_masks=None):

        idx = self._get_src_permutation_idx(indices)
        start_idx = 0
        _positive_map = []
        for _,i in indices:
            end_idx = start_idx+len(i)
            _positive_map.append(positive_map[start_idx:end_idx][i])
            start_idx = end_idx

        positive_map = torch.cat(_positive_map, dim=0)

        token_labels = torch.zeros_like(outputs['pred_logits'])
        token_labels[idx]= positive_map

        loss_ce = self.token_loss_func(outputs['pred_logits'],token_labels,text_masks=text_masks,version="binary")
        losses = {'loss_ce': loss_ce}
        return losses

    
    def loss_boxes(self, outputs, targets, indices):

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t[:,1:-1][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['l1_bbox'] = loss_bbox.mean() * 4

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        
        losses['loss_giou'] = loss_giou.mean()
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

    def prepare_targets(self,outputs, targets,positive_map= None):

        offset = 0
        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            bboxes_per_im = targets_per_im[:,1:-1]
            labels_per_im = targets_per_im[:,-1] # NOTE: the continues class index
            num_gt = len(bboxes_per_im)

            if positive_map is not None:
                token_per_im = positive_map[offset:offset + num_gt, :]
                offset += num_gt
            
            anchors_bbox_per_im = outputs["pred_boxes"][im_i]
            anchors_positive_map_per_im = outputs["pred_logits"][im_i].sigmoid()

            gt_cx = (bboxes_per_im[:, 2] + bboxes_per_im[:, 0]) / 2.0
            gt_cy = (bboxes_per_im[:, 3] + bboxes_per_im[:, 1]) / 2.0
            gt_points = torch.stack((gt_cx, gt_cy), dim=1)

            ious,union = box_ops.box_iou(anchors_bbox_per_im, bboxes_per_im)

            anchors_cx_per_im = (anchors_bbox_per_im[:, 2] + anchors_bbox_per_im[:, 0]) / 2.0
            anchors_cy_per_im = (anchors_bbox_per_im[:, 3] + anchors_bbox_per_im[:, 1]) / 2.0
            anchor_points = torch.stack((anchors_cx_per_im, anchors_cy_per_im), dim=1)
            distances = (anchor_points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()

            # candiate_idx for predn based on distance
            topk = min(self.args.atss_topk,len(anchors_bbox_per_im))
            _, topk_idxs = distances.topk(topk, dim=0, largest=False)

            candidate_ious = ious[topk_idxs, torch.arange(num_gt)]
            iou_mean_per_gt = candidate_ious.mean(0)
            iou_std_per_gt = candidate_ious.std(0)
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt

            is_pos = candidate_ious >= iou_thresh_per_gt[None, :]

            # predn'center should be within the target object
            anchor_num = anchors_cx_per_im.shape[0]
            for ng in range(num_gt):
                topk_idxs[:, ng] += ng * anchor_num
            e_anchors_cx = anchors_cx_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            e_anchors_cy = anchors_cy_per_im.view(1, -1).expand(num_gt, anchor_num).contiguous().view(-1)
            topk_idxs = topk_idxs.view(-1)
            l = e_anchors_cx[topk_idxs].view(-1, num_gt) - bboxes_per_im[:, 0] 
            t = e_anchors_cy[topk_idxs].view(-1, num_gt) - bboxes_per_im[:, 1]
            r = bboxes_per_im[:, 2] - e_anchors_cx[topk_idxs].view(-1, num_gt)
            b = bboxes_per_im[:, 3] - e_anchors_cy[topk_idxs].view(-1, num_gt)
            is_in_gts = torch.stack([l, t, r, b], dim=1).min(dim=1)[0] > 0.01

            is_pos = is_pos & is_in_gts

            ious_inf = torch.full_like(ious, -INF).t().contiguous().view(-1)
            index = topk_idxs.view(-1)[is_pos.view(-1)]
            ious_inf[index] = ious.t().contiguous().view(-1)[index] 
            ious_inf = ious_inf.view(num_gt, -1).t()

            anchors_to_gt_values, anchors_to_gt_indexs = ious_inf.max(dim=1) #  确保 一个 anchor 只会 分配给一个 target bbox

            # cls_labels_per_im = labels_per_im[anchors_to_gt_indexs]
            # cls_labels_per_im[anchors_to_gt_values == -INF] = 0 # For the negative background classes

            if positive_map is not None:
                token_labels_per_im = token_per_im[anchors_to_gt_indexs]
                unmatched_labels = torch.zeros(token_labels_per_im.shape[1], device=token_labels_per_im.device)
                unmatched_labels[-1] = 1
                token_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels # 不匹配的 分配为 unmatched
            
            matched_gts = bboxes_per_im[anchors_to_gt_indexs]


            import pdb;pdb.set_trace()
#Test the loss function

if __name__ == '__main__':
    from groundingdino.util.slconfig import SLConfig
    args = SLConfig.fromfile("../config/GroundingDINO_SwinT_OGC.py")
    args.device = "cuda"
    tmp = ATSSLossComputation(args)