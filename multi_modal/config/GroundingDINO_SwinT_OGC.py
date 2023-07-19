batch_size = 1
modelname = "groundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
use_checkpoint = True
use_transformer_ckpt = True
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

# For Finetune
is_train = True
language_backbone_freeze = True
image_backbone_freeze = True
transformer_freeze = True
box_cls_embed_freeze = False

# For finetune loss

## For macher between predn and target
atss_topk = 100  # 用于统计 距离匹配前 100 个 predn 的 iou 的均值跟方差
set_cost_class = 2.0
set_cost_bbox = 5.0
set_cost_giou = 2.0
FUSE_TOKEN_ALPHA = 0.25
FUSE_TOKEN_GAMMA = 2.0

# Optimizer
optimizer = "ADAMW"
base_lr = 1.0e-5
weight_decay = 0.0001
weight_decay_bias = 0.0
bias_lr_factor = 2
weight_decay_norm_factor = 1.0
gridient_clip_type = "full_model"
gridient_clip_value = 1.0
gradient_clip_enabled = True
gradient_clip_norm_type = 2.0

# Learning_rate_schedule
lr_use_cosine = False
lr_use_autostep = False
max_epoch = 10
max_iter = None
schedule_gamma = 0.1
warmup_factor =  0.001
warmup_iters = None
warmup_method = "linear"
warmup_min_lr = 1.0e-6
warmup_step_patience = 5 # for autostep
lr_steps = [0.67,0.89]

