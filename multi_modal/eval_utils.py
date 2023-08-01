import torch
import numpy as np

class Evaluation:
    def __init__(self, predictions, targets, threshold):
        super(Evaluation, self).__init__()
        self.predictions = predictions
        self.targets = targets
        self.threshold = threshold

    @staticmethod
    def compute_ap(recall, precision):
        # average precision calculation
        recall = np.concatenate(([0.], recall, [1.]))
        precision = np.concatenate(([0.], precision, [0.]))

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])

        ap = 0.0  # average precision (AUC of the precision-recall curve).
        for i in range(precision.size - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        return ap

    def evaluate(self): #transfer the class_id to class_name previouly 不用担心class_id 在pred 跟label起始索引不一样
        aps = []
        print('CLASS'.ljust(25, ' '), 'AP')
        for class_name in CAR_CLASSES:
            class_preds = self.predictions[class_name]  # [[image_id,confidence,x1,y1,x2,y2],...]
            if len(class_preds) == 0:
                ap = 0
                print(f'{class_name}'.ljust(25, ' '), f'{ap:.2f}')
                aps.append(ap)
                continue
            image_ids = [x[0] for x in class_preds]
            confidence = np.array([float(x[1]) for x in class_preds])
            BB = np.array([x[2:] for x in class_preds])
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            npos = 0.
            for (key1, key2) in self.targets:
                if key2 == class_name:
                    npos += len(self.targets[(key1, key2)])
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            for d, image_id in enumerate(image_ids):
                bb = BB[d]
                if (image_id, class_name) in self.targets:
                    BBGT = self.targets[(image_id, class_name)]
                    for x1y1_x2y2 in BBGT:
                        # compute overlaps
                        # intersection
                        x_min = np.maximum(x1y1_x2y2[0], bb[0])
                        y_min = np.maximum(x1y1_x2y2[1], bb[1])
                        x_max = np.minimum(x1y1_x2y2[2], bb[2])
                        y_max = np.minimum(x1y1_x2y2[3], bb[3])
                        w = np.maximum(x_max - x_min + 1., 0.)
                        h = np.maximum(y_max - y_min + 1., 0.)
                        intersection = w * h

                        union = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (x1y1_x2y2[2] - x1y1_x2y2[0] + 1.) * (
                                x1y1_x2y2[3] - x1y1_x2y2[1] + 1.) - intersection
                        if union == 0:
                            print(bb, x1y1_x2y2)

                        overlaps = intersection / union
                        if overlaps > self.threshold:
                            tp[d] = 1
                            BBGT.remove(x1y1_x2y2)
                            if len(BBGT) == 0:
                                del self.targets[(image_id, class_name)]
                            break
                    fp[d] = 1 - tp[d]
                else:
                    fp[d] = 1
            ###################################################################
            # TODO: Please fill the codes to compute recall and precision
            ##################################################################
            fp = fp.cumsum(0)
            tp = tp.cumsum(0)
            recall = tp / (npos + 1e-16) 
            px, py = np.linspace(0, 1, 1000), [] 
            # r_c = np.interp(-px, sorted_scores, recall, left=0) # use to plot
            precision = tp / (tp + fp) 
            # p_c = np.interp(-px, sorted_scores, precision, left=1) # use to plot

            ##################################################################
            ap = self.compute_ap(recall, precision)

            print(f'{class_name}'.ljust(25, ' '), f'{ap*100:.2f}')
            aps.append(ap)

        return aps

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc)), np.zeros((nc, 1000)), np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            # recall[:,0]取了第一个iou=0.5时候的recall，
            r[ci] = np.interp(-px, -conf[i], recall, left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision, left=1)  # p at pr_score

            # AP from recall-precision curve
            ap[ci] = Evaluation.compute_ap(recall, precision)
    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)


    return {"ap":ap,"f1":f1,"recall":r,"precision":p,"classes":unique_classes.astype('int32')}



def box_iou(box1, box2):
    """ compute IOU between boxes
        - box1 (bs, 4)  4: [x1, y1, x2, y2]  left top and right bottom
        - box2 (bs, 4)  4: [x1, y1, x2, y2]  left top and right bottom
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh += 1.
    wh[wh < 0] = 0  # clip at 0
    inter = (wh[:, :, 0]) * (wh[:, :, 1])  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]+1.) * (box1[:, 3] - box1[:, 1]+1.)  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]+1.) * (box2[:, 3] - box2[:, 1]+1.)  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


def processs_batch(predn,targets,match_threshold=0.3):
    '''
    inputs:
    predn list[torch.Tensor(N*6),x1y1x2y2,conf,class_index]
    targets tuple((N*6), bs_id, x1y1x2y2, class_index )
    '''
    device = predn[0].device
    stats = []
    for si, pred in enumerate(predn):
        labels = targets[si]
        labels = labels.to(device)
        nl = len(labels)
        tcls = labels[:, -1].tolist() if nl else [] 
        
        if len(pred) == 0:
            if nl:
                stats.append((torch.zeros(0, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)) # only calculate mAP10
            continue
        
        if nl:
            correct = torch.zeros(pred.shape[0], dtype=torch.bool, device=device)
            iou = box_iou(labels[:, 1:5], pred[:, :4])
            x = torch.where((iou > match_threshold) & (labels[:, 5:6] == pred[:, 5]))  # IoU above threshold and classes match 
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
                for pred_id in range(len(pred)):
                    tmp_matches = matches[matches[:,1]==pred_id]
                    if len(tmp_matches):
                        target_remove_id = tmp_matches[0][0]
                        correct[pred_id] = True
                        matches = matches[matches[:,0]!=target_remove_id] 
                # matches = matches[np.unique(matches[:, 1], return_index=True)[1]] #再满足检测框的匹配
                # matches = matches[np.unique(matches[:, 0], return_index=True)[1]] #优先满足标签匹配
                # matches = torch.Tensor(matches).to(device)
                # correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
                # correct[matches[:, 1].long()] = True
        else:
            correct = torch.zeros(pred.shape[0], dtype=torch.bool)
        stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)
    return stats