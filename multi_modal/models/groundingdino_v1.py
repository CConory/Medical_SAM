# ------------------------------------------------------------------------
# XXXXX
# url: XXXXX
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
# Modified from Grounding DINO (https://github.com/IDEA-Research/GroundingDINO)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import List

from groundingdino.models.GroundingDINO.transformer import build_transformer
from groundingdino.models.GroundingDINO.backbone import build_backbone
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import (
    NestedTensor,
    accuracy,
    get_world_size,
    interpolate,
    inverse_sigmoid,
    is_dist_avail_and_initialized,
    nested_tensor_from_tensor_list,
)

from groundingdino.models.GroundingDINO.utils import ContrastiveEmbed,MLP

from groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from .detr import MaskHeadSmallConv,MHAttentionMap,MaskHeadAsSam,Mask_head_v1
from segment_anything.modeling import TwoWayTransformer
from segment_anything.modeling.common import LayerNorm2d

import copy
from torch import nn
import torch
import torch.nn.functional as F


class GroundingDINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=2,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        training_args= None
    ):
        """Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.training_args = training_args

        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)
        # freeze

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        # prepare input projection layers
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.box_pred_damping = box_pred_damping = None

        self.iter_update = iter_update
        assert iter_update, "Why not iter_update?"

        # prepare pred layers
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share

        # NOTE: prepare class & box embed
        _class_embed = ContrastiveEmbed()

        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for i in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(transformer.num_decoder_layers)
            ]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
        
        # self.bbox_attention = MHAttentionMap(self.hidden_dim, self.hidden_dim, self.nheads, dropout=0)
        # self.mask_head = MaskHeadAsSam(self.hidden_dim + self.nheads, self.hidden_dim)
        self.mask_embed = MLP(self.hidden_dim, self.hidden_dim, self.hidden_dim, 3)
        # self.mask_head = Mask_head_v1(self.hidden_dim, self.hidden_dim, self.nheads, dropout=0)

        # 
        # self.mask_transformer=TwoWayTransformer(depth=2, embedding_dim=256,mlp_dim=2048,num_heads=4,)
        # self.output_upscaling = nn.Sequential(
        #     nn.ConvTranspose2d(self.hidden_dim , self.hidden_dim  // 4, kernel_size=2, stride=2),
        #     LayerNorm2d(self.hidden_dim  // 4),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(self.hidden_dim  // 4, 1, kernel_size=2, stride=2),
        #     nn.GELU(),
        # )
        self._reset_parameters()

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
    
    def forward(self, samples: NestedTensor, targets: List = None, **kw):
        """The forward expects a NestedTensor, which consists of:
           - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
           - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """

        if targets is None:
            captions = kw["captions"]
        else:
            captions = [t["caption"] for t in targets]
        

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(
            samples.device
        )
        (
            text_self_attention_masks,
            position_ids,
            cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            tokenized, self.specical_tokens, self.tokenizer
        )

        if text_self_attention_masks.shape[1] > self.max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]
            position_ids = position_ids[:, : self.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:, : self.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:, : self.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:, : self.max_text_len]

        # extract text embeddings
        if self.sub_sentence_present:
            tokenized_for_encoder = {k: v for k, v in tokenized.items() if k != "attention_mask"}
            tokenized_for_encoder["attention_mask"] = text_self_attention_masks
            tokenized_for_encoder["position_ids"] = position_ids
        else:
            # import ipdb; ipdb.set_trace()
            tokenized_for_encoder = tokenized
        
        if self.training_args is not None and self.training_args['language_backbone_freeze']:
            with torch.no_grad():
                bert_output = self.bert(**tokenized_for_encoder) 
        else:
            bert_output = self.bert(**tokenized_for_encoder) 
        

        encoded_text = self.feat_map(bert_output["last_hidden_state"])  # bs, 195, d_model
        text_token_mask = tokenized.attention_mask.bool()  # bs, 195
        # text_token_mask: True for nomask, False for mask
        # text_self_attention_masks: True for nomask, False for mask

        if encoded_text.shape[1] > self.max_text_len:
            encoded_text = encoded_text[:, : self.max_text_len, :]
            text_token_mask = text_token_mask[:, : self.max_text_len]
            position_ids = position_ids[:, : self.max_text_len]
            text_self_attention_masks = text_self_attention_masks[
                :, : self.max_text_len, : self.max_text_len
            ]

        # NOTE: 
        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # TODO: # 探究 poss 怎么来, possition_encoding; features 包含 src 跟 mask
        if self.training_args is not None and self.training_args['image_backbone_freeze']:
            with torch.no_grad():   
                features, poss = self.backbone(samples) 
        else:
            features, poss = self.backbone(samples) #mask 就是 nested_tensor_from_tensor_list生成
        # {'tensors.shape': torch.Size([4, 192, 100, 100]), 'mask.shape': torch.Size([4, 100, 100])}
        # poss[0].shape = [4,256,100,100]
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                if self.training_args is not None and self.training_args['image_backbone_freeze']:
                    with torch.no_grad():
                        pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                else:
                    pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        input_query_bbox = input_query_label = attn_mask = dn_meta = None

        if self.training_args is not None and self.training_args['transformer_freeze']:
            with torch.no_grad():
                hs, reference, hs_enc, ref_enc, init_box_proposal,memory = self.transformer(
                    srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
                )
        else:
            hs, reference, hs_enc, ref_enc, init_box_proposal,memory = self.transformer(
                srcs, masks, input_query_bbox, poss, input_query_label, attn_mask, text_dict
            )        
        
        if self.training_args is not None and self.training_args['box_cls_embed_freeze']:
            with torch.no_grad():
                # deformable-detr-like anchor update
                outputs_coord_list = []
                for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                    zip(reference[:-1], self.bbox_embed, hs)
                ):
                    layer_delta_unsig = layer_bbox_embed(layer_hs)
                    layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
                    layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                    outputs_coord_list.append(layer_outputs_unsig)
                outputs_coord_list = torch.stack(outputs_coord_list)

                # output
                outputs_class = torch.stack(
                    [
                        layer_cls_embed(layer_hs, text_dict)
                        for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                    ]
                )
        else:
            # deformable-detr-like anchor update
            outputs_coord_list = []
            for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)
            ): 
                layer_delta_unsig = layer_bbox_embed(layer_hs) # [bs,num_queries,256] ->(fc)-> [bs,num_queries,4]
                layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig) # 残差融合 [bs,num_queries,4]
                layer_outputs_unsig = layer_outputs_unsig.sigmoid()
                outputs_coord_list.append(layer_outputs_unsig)
            outputs_coord_list = torch.stack(outputs_coord_list)

            # output # [6,bs,num_queryes,hidden_dim]
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, text_dict)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )

        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        
        # Mask Decoder
        # Version1
        # hs_enc the proposal bboxes focus on the embedding areas features
        # hs 如何跟 原图关联起来，原特征每一个pixel[bs,256,h,w] 与这对应区域的特征 [bs, 900, 256] -> [bs,900,h,w] 
        # bbox_mask = self.bbox_attention(hs[-1], memory, mask=masks[1])
        # seg_masks = self.mask_head(srcs[1], bbox_mask)
        # outputs_mask = seg_masks.view(bbox_mask.shape[0], bbox_mask.shape[1], seg_masks.shape[-2], seg_masks.shape[-1])

        # outputs_mask = self.mask_head(hs[-1],memory,srcs[:-1],masks)

        # version2
        mask_embed = self.mask_embed( hs[-1])
        mask_features = srcs[0]
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        out["pred_masks"] = outputs_mask

        # version 3
        # outputs_mask = torch.einsum("bqc,bchw->bqhw", hs[-1], memory)

        # Version 4 based on SAM decoder
        # hs, src = self.mask_transformer(srcs[0],poss[0],hs[-1])
        # b,c,h,w = srcs[0].shape
        # src = src.transpose(1, 2).view(b, c, h, w) #[1,h/8*w/8,256] -> [1,256,h/8,w/8]
        # upscaled_embedding = self.output_upscaling(src) #[1,32,h/2,w/2]
        # out["pred_masks"] = upscaled_embedding
        return out


def build_groundingdino(args):

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    dn_labelbook_size = args.dn_labelbook_size
    dec_pred_bbox_embed_share = args.dec_pred_bbox_embed_share
    sub_sentence_present = args.sub_sentence_present

    training_args = {
        "language_backbone_freeze":args.language_backbone_freeze,
        "image_backbone_freeze": args.image_backbone_freeze,
        "transformer_freeze":args.transformer_freeze,
        "box_cls_embed_freeze":args.box_cls_embed_freeze
        }

    model = GroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=True,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        dn_number=0,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_labelbook_size=dn_labelbook_size,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=sub_sentence_present,
        max_text_len=args.max_text_len,
        training_args=training_args if args.is_train else None
    )

    return model


if __name__ == '__main__':
    from groundingdino.util.slconfig import SLConfig
    args = SLConfig.fromfile("../config/GroundingDINO_SwinT_OGC.py")
    args.device = "cuda"
    model = build_groundingdino(args)