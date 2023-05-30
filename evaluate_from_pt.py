from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from dataset import PanNuke_Dataset
import wandb

ROOT_PATH = './datasets/'
DATANAME = "PanNuke"

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

def evaluation(model,dataloader,device,max_vis=10):
    '''
        based on the features embedding;
        inference include the prompt encoder and masks decoder.
        return:
            mIoU,Dice
            visualization result
                Blue for prediction
                Green for target
                Yellow for intersection
    '''

    mIoU = []
    dice = []
    vis_results = []
    vis_count = 0
    for batched_input,masks,image_paths in tqdm(dataloader):
        for batch_id in range(len(batched_input)):
            batched_input[batch_id]['feature'] = batched_input[batch_id]['feature'].to(device)
        # batched_input = [
        #     {
        #         "feature":feature,
        #         'original_size':data["origin_shape"][:2]
        #     }
        # ]
        # masks.shape : (BS, category_num, 1, width, height)
        batched_output = predictor.predict(batched_input, multimask_output=False)

        for batch_id in range(len(batched_output)):
            mask_output = batched_output[batch_id]['masks'].cpu().numpy()
            target = masks[batch_id]
            _mIoU,_dice = mean_iou_and_dice(target,mask_output) # bowl-2018 no points should do ~ operation

            mIoU.append(_mIoU)
            dice.append(_dice)

            # visualization
            if vis_count< max_vis:
                img = cv2.imread(image_paths[batch_id])
                pred_fg = mask_output[0][0]!=0
                target_fg = target[0][0]!=0
                inter_fg = (mask_output[0][0]!=0) & (target[0][0]!=0)
                target_fg = (~inter_fg) & target_fg
                pred_fg = (~inter_fg) & pred_fg
                img[pred_fg] = (img[pred_fg]*0.6 + tuple(tmp*0.4 for tmp in (255,0,0))).astype(np.uint8) # Blue for prediction
                img[target_fg] = (img[target_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,0))).astype(np.uint8) # Green for target
                img[inter_fg] = (img[inter_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,255))).astype(np.uint8) # Yellow for intersection
                vis_results.append(wandb.Image(img[...,::-1]))
            vis_count += 1

        # _mIoU,_dice = mean_iou_and_dice(target[None][None].astype(bool),~mask_output) # bowl-2018 no points should do ~ operation
        # mIoU.append(_mIoU)
        # dice.append(_dice)

        # Check 
        # mask_output = (mask_output*255).astype(int)
        # cv2.imwrite("./tmp.jpg",mask_output[0][0])
        # import pdb;pdb.set_trace()
    
    return  mIoU, dice, vis_results

if __name__ == '__main__':

    dataset_name = "PanNuke"
    prompt_type = "N"

    wandb.init(project="Medical_SAM",config={
        "dataset": dataset_name,
        "prompt": prompt_type
    })
    wandb.run.name = wandb.run.id
    wandb.run.save()

    device = "cuda"

    # Load dataset
    valid_dataset = PanNuke_Dataset(f"./datasets/{dataset_name}/data_split.json","valid",device)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = PanNuke_Dataset.collate_fn,
        drop_last=False
    )

    sam_checkpoint = "/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 一个网络结构
    sam=sam.to(device=device)
    predictor = Prompt_plut_decoder(sam) 

    mIoU,dice, valid_vis_results = evaluation(predictor,valid_loader,device)
    
    valid_mIoU = round(sum(mIoU)/len(mIoU),3)
    valid_dice = round(sum(dice)/len(dice),3)
    print("valid_mIoU: ",valid_mIoU)
    print("valid_Dice: ",valid_dice)
    
    test_dataset = PanNuke_Dataset(f"./datasets/{dataset_name}/data_split.json","test",device)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = PanNuke_Dataset.collate_fn,
        drop_last=False
    )

    mIoU,dice, test_vis_results = evaluation(predictor,test_loader,device)

    test_mIoU = round(sum(mIoU)/len(mIoU),3)
    test_dice = round(sum(dice)/len(dice),3)
    print("test_mIoU: ",test_mIoU)
    print("test_Dice: ",test_dice)


    wandb.log({
        "valid_results": valid_vis_results,
        "test_results": test_vis_results,
        "valid/mIoU":valid_mIoU,
        "valid/dice":valid_dice,
        "test/mIoU":test_mIoU,
        "test/dice":test_dice
        })