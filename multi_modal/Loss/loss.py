# The matcher is refered from Deformable-DETR (https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/deformable_detr.py#L483)
# And the multi-modal loss is from GLIP
try:
    from .matcher import build_matcher
except:
    from matcher import build_matcher

import torch
from groundingdino.util import box_ops
# from torch.cuda.amp import custom_fwd

INF = 1e8
class ATSSLossComputation(torch.nn.Module):
    def __init__(self, args):
        super(ATSSLossComputation, self).__init__()
        self.args = args
        self.matcher = build_matcher(args)

    # @custom_fwd(cast_inputs=torch.float32)
    def forward(self, outputs, targets,positive_map=None):
        indices = self.matcher(outputs, targets,positive_map)
        # with torch.no_grad():
        #     self.prepare_targets(outputs,targets,positive_map)

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