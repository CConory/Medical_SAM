from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple


ROOT_PATH = './datasets/'
DATANAME = "bowl_2018"

class Prompt_plut_decoder:
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
    
    @torch.no_grad()
    def predict(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:


        outputs = []
        for image_record in batched_input:
            if "point_coords" in image_record:
                points = (image_record["point_coords"], image_record["point_labels"])
            else:
                points = None
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=image_record["feature"],
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.model.postprocess_masks(
                low_res_masks,
                input_size=(self.model.image_encoder.img_size,self.model.image_encoder.img_size), # encode 前输入网络的大小
                original_size=image_record["original_size"],
            )
            masks = masks > self.model.mask_threshold
            outputs.append(
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs


def calculate_iou(y_true, y_pred):
    '''Calculate Intersection over Union (IOU) for a given class'''
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
#     show_mask(intersection, plt.gca(), random_color=True)
#     plt.show()
#     show_mask(union, plt.gca(), random_color=True)
#     plt.show()
    return iou_score

def dice_score(y_true, y_pred):
    '''Calculate Dice score for a given class'''
    intersection = np.sum(y_true * y_pred)
    dice_score = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice_score

def mean_iou_and_dice(y_true,y_pred): #[category_nums,1,w,h]
    num_classes = y_true.shape[0]
    iou_scores = []
    dice_scores = []
    for class_idx in range(num_classes):
        y_true_class = y_true[class_idx]
        y_pred_class = y_pred[class_idx]
        if np.sum(y_true_class) == 0:
            if np.sum(y_pred_class) == 0:
                iou_scores.append(1.0)
                dice_scores.append(1.0)
            else:
                iou_scores.append(0.0)
                dice_scores.append(0.0)
        else:
            iou_scores.append(calculate_iou(y_true_class, y_pred_class))
            dice_scores.append(dice_score(y_true_class, y_pred_class))
    mean_iou_score = np.mean(iou_scores)
    ean_dice_score = np.mean(dice_scores)
    return mean_iou_score,ean_dice_score


if __name__ == '__main__':

    sam_checkpoint = "/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 一个网络结构
    sam=sam.to(device=device)
    predictor = Prompt_plut_decoder(sam) 

    # Get Features & labels

    TRAIN_PATH = os.path.join(ROOT_PATH,DATANAME)
    train_ids = next(os.walk(TRAIN_PATH))[1]

    mIoU = []
    dice = []

    for file_name in tqdm(train_ids):
        path = os.path.join(TRAIN_PATH,file_name)
        data = torch.load(path + '/features/' + file_name + '.pt')
        feature = data["fm"].to(device=device)
        target = None
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file,-1)
            if target is None:
                target = mask_
            else:
                target = np.maximum(target, mask_)  

        # 预留了batch 借口， 目前 一个batch 一张图片
        batched_input = [
            {
                "feature":feature,
                'original_size':data["origin_shape"][:2]
            }
        ]

        batched_output = predictor.predict(batched_input, multimask_output=False)

        # calculate Metrices
        mask_output = batched_output[0]['masks'].cpu().numpy()
        _mIoU,_dice = mean_iou_and_dice(target[None][None].astype(bool),~mask_output) # bowl-2018 no points should do ~ operation
        mIoU.append(_mIoU)
        dice.append(_dice)

        # Check 
        # mask_output = (mask_output*255).astype(int)
        # cv2.imwrite("./tmp.jpg",mask_output[0][0])
        # import pdb;pdb.set_trace()

    print("mIoU: ",round(sum(mIoU)/len(mIoU),3))
    print("Dice: ",round(sum(dice)/len(dice),3))
        

