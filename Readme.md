# Medical_SAM

## Step0: Install  
Follows the installation of  [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [Segment-anything](https://github.com/facebookresearch/segment-anything)
## Step1: Conver data format
Please follow the instruction in dir ```./data```
## The baseline experiment:
    ### Step2: Extract the image embedding
    ```
    # e.g.
    python extract_fm \
        --dataset PanNuke \
        --weight_path WEIGHT_DIR/sam_vit_h_4b8939.pth \
        --model_type vit_h

    Argument:
    dataset: the folder name in ./dataset
    weight_path: the path of pretrained model weight, which is download from SAM
    model_type: should correspone to the weight_path, which could be checked at SAM'github
    Output:
    The image embedding would be save as a torch.tensor in ./datasets/DATASET_NAME/features
    ```
    ### Step3: Evaluate the Dataset with different prompt
    ```
    e.g.
    python evaluate_from_pt.py \
        --dataset MoNuSeg \
        --weight_path WEIGHT_DIR/sam_vit_h_4b8939.pth \
        --model_type vit_h \
        --prompt_type One_Point \
        --wandb_log
    Argument:
        such arguments are as same in Step2
        prompt_type: which type of prompt you want to use
        wandb_log : save the score and visualization to wandb 
    ```
## MaskGrounding DINO
```
cd ./multi_modal
```
### Finetuned 
```
python finetune_GDINO.py --dataset DASTASET_NAME --output_dir SAVE_PATH
```
### Evaluate the pretrained or finetuend model, e.g. SegPC-2021's valid set
```
python Grounding_dino_infer.py --dataset SegPC-2021 --dataset_type valid
```
### Generate the segmentation tasks e.g. SegPC-2021's valid set
```
python generate_mask.py --dataset SegPC-2021 --dataset_type valid --wandb_log
```

## Model ZOO
### Pretrain-model
1. SAM : pleased download from [Segment-anything](https://github.com/facebookresearch/segment-anything)
2. The multi-modal pretrained model pleased downloaded from [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

### Fintuned-model
1. MaskGroundingDINO : [weights](https://drive.google.com/file/d/1gMQe8RywGqzQfAQaQzYGUe--XRzAtaQT/view?usp=drive_link) , [Config](https://drive.google.com/file/d/1l5h4lxqNhSS1hwurFx6fL9x4R4iIKcag/view?usp=drive_link)

