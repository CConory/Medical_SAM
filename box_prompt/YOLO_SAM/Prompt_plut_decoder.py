from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.transforms import ResizeLongestSide
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
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
