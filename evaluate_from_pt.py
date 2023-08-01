from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
from dataset import Dataset,One_point_Dataset,Two_point_Dataset,Five_point_Dataset,Twenty_point_Dataset,All_point_Dataset,All_boxes_Dataset
import matplotlib.pyplot as plt
import wandb
import argparse
from torch.nn import functional as F


def calculate_before_padding_size(after_padding_size,original_size):
    """
        original_size 长边等比缩放到 after_padding_size 的长度；
        计算等比缩放后的,padding前的 size
    """
    rr = min(after_padding_size[0]/original_size[0],after_padding_size[1]/original_size[1])
    return_size = (int(original_size[0]*rr),int(original_size[1]*rr))
    return return_size
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
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.device = self.model.device

    @torch.no_grad()
    def predict(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:


        outputs = []
        for image_record in batched_input:

            points = image_record.get("point_coords", None)
            point_labels =  image_record.get("point_labels", None) 
            box = image_record.get("boxes", None)
            if box is not None and not len(box):
                box = None
            masks = image_record.get("mask_inputs", None)

            coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None

            input_size = calculate_before_padding_size((self.model.image_encoder.img_size,self.model.image_encoder.img_size),image_record["original_size"])

            # Preprocessing for point-prompt:
            if points is not None:
                assert (
                    point_labels is not None
                ), "point_labels must be supplied if point_coords is supplied."

                point_coords = self.transform.apply_coords(points, image_record["original_size"])
                coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
                labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
                coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
                if box is not None:
                    bs = max(box.shape[0],1)
                    if bs > 1:
                        coords_torch = coords_torch.repeat(bs,1,1)
                        labels_torch = labels_torch.repeat(bs,1)

            if box is not None:
                box = self.transform.apply_boxes(box, image_record["original_size"])
                box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
                # box_torch = box_torch[None, :]
                # tmp_low_res_masks = None
                tmp_iou_predictions = None
                max_box_inference = 16
                final_masks = torch.zeros((1,*image_record["original_size"]),device=self.device)
                for box_id in range(0,len(box_torch),max_box_inference): #受显存限制
                    if coords_torch is not None:
                        tmp_coords_torch = coords_torch[box_id:box_id+max_box_inference]
                        tmp_labels_torch = labels_torch[box_id:box_id+max_box_inference]
                    points = (tmp_coords_torch,tmp_labels_torch) if coords_torch is not None else None
                    tmp_box_torch = box_torch[box_id:box_id+max_box_inference]
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                        points=points,
                        boxes=tmp_box_torch,
                        masks=masks,
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
                        input_size=input_size, # 输入网络前未padding的大小
                        original_size=image_record["original_size"],
                    )
                    masks = masks > self.model.mask_threshold
                    for mask_id in range(len(masks)):
                        final_masks[masks[mask_id]] = mask_id+box_id+1
                    
                    if tmp_iou_predictions is None:
                        tmp_iou_predictions = iou_predictions
                    else:
                        tmp_iou_predictions = torch.cat((tmp_iou_predictions,iou_predictions))
                    masks = None

                iou_predictions = tmp_iou_predictions
                masks = final_masks[None]
                low_res_masks = None

            else:
                points = (coords_torch,labels_torch) if coords_torch is not None else None
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=points,
                    boxes=box_torch,
                    masks=masks,
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
                    input_size=input_size, # 输入网络前未padding的大小
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


def show_mask(mask, ax, color):
    # if random_color:
    #     color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    # else:
    #     color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=50):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax,color='green'):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0,0,0,0), lw=2))  


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
            instance_bboxes = batched_input[batch_id].get("boxes", None)
            mask_output = batched_output[batch_id]['masks']
            mask_output = mask_output!=0
            mask_output = mask_output.cpu().numpy()

            if instance_bboxes is not None and not len(instance_bboxes): 
                mask_output = np.zeros_like(mask_output)


            target = masks[batch_id]
            _mIoU,_dice = mean_iou_and_dice(target,mask_output) # bowl-2018 no points should do ~ operation

            mIoU.append(_mIoU)
            dice.append(_dice)
        

            # visualization
            if vis_count< max_vis:
                img = cv2.imread(image_paths[batch_id])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                pred_fg = mask_output[0][0]!=0
                target_fg = target[0][0]!=0
                inter_fg = (mask_output[0][0]!=0) & (target[0][0]!=0)
                target_fg = (~inter_fg) & target_fg
                pred_fg = (~inter_fg) & pred_fg

                plt.figure(figsize=(10,10))
                plt.imshow(img)
                show_mask(pred_fg, plt.gca(),np.array([0/255, 0/255, 255/255, 0.4]))
                show_mask(target_fg, plt.gca(),np.array([0/255, 255/255, 0/255, 0.4]))
                show_mask(inter_fg, plt.gca(),np.array([255/255, 255/255, 0/255, 0.4]))
                if batched_input[batch_id].get("point_coords", None) is not None:
                    show_points(batched_input[batch_id]['point_coords'], batched_input[batch_id]['point_labels'], plt.gca())
                if instance_bboxes is not None:
                    for box in instance_bboxes:
                        show_box(box, plt.gca())
                plt.axis('off')
                # show_points(input_point, input_label, plt.gca())

                # img[pred_fg] = (img[pred_fg]*0.6 + tuple(tmp*0.4 for tmp in (255,0,0))).astype(np.uint8) # Blue for prediction
                # img[target_fg] = (img[target_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,0))).astype(np.uint8) # Green for target
                # img[inter_fg] = (img[inter_fg]*0.6 + tuple(tmp*0.4 for tmp in (0,255,255))).astype(np.uint8) # Yellow for intersection
                vis_results.append(wandb.Image(plt))
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MoNuSeg', help='the path of the dataset')
    parser.add_argument('--weight_path', type=str, default='/userhome/cs2/kuangww/segment-anything/notebooks/models/sam_vit_h_4b8939.pth', help='the path of the pre_train weight')
    parser.add_argument('--model_type', type=str, default='vit_h', help='the type of the model')
    parser.add_argument('--prompt_type', type=str, default='N', help='the type of the prompt')
    parser.add_argument('--wandb_log', action='store_true', help='save the result to wandb or not')
    args = parser.parse_args()

    wandb_flag = args.wandb_log

    prompt_dict = {
        "N" : Dataset,
        "One_Point" : One_point_Dataset,
        "Two_Point" : Two_point_Dataset,
        "Five_Point" : Five_point_Dataset,
        "Twenty_Point" : Twenty_point_Dataset,
        "All_Point": All_point_Dataset,
        "All_boxes": All_boxes_Dataset
    }

    dataset_root_dir = "./datasets"
    dataset_name = args.dataset
    dataset_dir = os.path.join(dataset_root_dir,dataset_name)
    prompt_type = args.prompt_type

    if wandb_flag:
        wandb.init(project="Medical_SAM",config={
            "dataset": dataset_name,
            "prompt": prompt_type
        })
        wandb.run.name = wandb.run.id
        wandb.run.save()

    device = "cuda"

    # Load dataset
    valid_dataset = prompt_dict[prompt_type](os.path.join(dataset_dir,"data_split.json"),"valid",device)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = prompt_dict[prompt_type].collate_fn,
        drop_last=False
    )

    sam_checkpoint = args.weight_path
    model_type = args.model_type
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # 一个网络结构
    sam=sam.to(device=device)
    predictor = Prompt_plut_decoder(sam) 

    mIoU,dice, valid_vis_results = evaluation(predictor,valid_loader,device)
    
    valid_mIoU = round(sum(mIoU)/len(mIoU),3)
    valid_dice = round(sum(dice)/len(dice),3)
    print("valid_mIoU: ",valid_mIoU)
    print("valid_Dice: ",valid_dice)
    
    test_dataset = prompt_dict[prompt_type](os.path.join(dataset_dir,"data_split.json"),"test",device)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=2,
        num_workers=4,
        shuffle=False,
        pin_memory=False,
        collate_fn = prompt_dict[prompt_type].collate_fn,
        drop_last=False
    )

    mIoU,dice, test_vis_results = evaluation(predictor,test_loader,device)

    test_mIoU = round(sum(mIoU)/len(mIoU),3)
    test_dice = round(sum(dice)/len(dice),3)
    print("test_mIoU: ",test_mIoU)
    print("test_Dice: ",test_dice)

    if wandb_flag:
        wandb.log({
            "valid_results": valid_vis_results,
            "test_results": test_vis_results,
            "valid/mIoU":valid_mIoU,
            "valid/dice":valid_dice,
            "test/mIoU":test_mIoU,
            "test/dice":test_dice
            })