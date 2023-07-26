import torch
from torch import nn
from torch.nn import functional as F
import copy

from typing import Any, Dict, List, Tuple
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder

from groundingdino.util import get_tokenlizer
from groundingdino.models.GroundingDINO.bertwarper import (
    BertModelWarper,
    generate_masks_with_special_tokens,
    generate_masks_with_special_tokens_and_transfer_map,
)
from groundingdino.util.misc import inverse_sigmoid,NestedTensor
from groundingdino.models.GroundingDINO.utils import ContrastiveEmbed,MLP
from .detr import MaskHeadSmallConv,MHAttentionMap,MaskHeadAsSam

class IT_decoder(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
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
        training_args= None,
        poss_encoding=None,
    ):
        super().__init__()
        self.poss_encoding = poss_encoding
        self.training_args = training_args
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = 256
        self.sub_sentence_present = sub_sentence_present

        # setting query dim
        self.query_dim = query_dim
        assert query_dim == 4
    
        # bert
        self.tokenizer = get_tokenlizer.get_tokenlizer(text_encoder_type)
        self.bert = get_tokenlizer.get_pretrained_language_model(text_encoder_type)
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)
        self.bert = BertModelWarper(bert_model=self.bert)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, self.hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        # special tokens
        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])


        # prepare input projection layers
        # assert two_stage_type == "no", "two_stage_type should be no if num_feature_levels=1 !!!"
        self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone_out_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        
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


        self.bbox_attention = MHAttentionMap(self.hidden_dim, self.hidden_dim, self.nheads, dropout=0)
        self.mask_head = MaskHeadAsSam(self.hidden_dim + self.nheads, self.hidden_dim)

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def forward(self,img_encoding,captions):
        # img_encoding.shape : [bs, 256, h/16, w/16]
        device = img_encoding.device

        # encoder texts
        tokenized = self.tokenizer(captions, padding="longest", return_tensors="pt").to(device)
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
        
        text_dict = {
            "encoded_text": encoded_text,  # bs, 195, d_model
            "text_token_mask": text_token_mask,  # bs, 195
            "position_ids": position_ids,  # bs, 195
            "text_self_attention_masks": text_self_attention_masks,  # bs, 195,195
        }

        srcs = []
        masks = []
        # poss = [image_pe]
        B,C,H,W = img_encoding.shape
        mask = torch.zeros((B,H,W),dtype=torch.bool,device=device)
        poss = [self.poss_encoding(NestedTensor(img_encoding, mask))]
        masks.append(mask)
        srcs.append(self.input_proj[0](img_encoding)) # bs, 256, h/16, w/16

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
        
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        # bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        # seg_masks = self.mask_head(srcs[0], bbox_mask)
        # outputs_seg_masks = seg_masks.view(B, bbox_mask.shape[1], seg_masks.shape[-2], seg_masks.shape[-1])
        # out["pred_masks"] = outputs_seg_masks
        return out




class Sam(nn.Module):
    mask_threshold: float = 0.0 #sigmoid 前0 == sigmoid 后0.5
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        new_decoder : IT_decoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        # self.mask_decoder = mask_decoder
        self.new_decoder = new_decoder
        # self.text_mask_decoder = 

    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    
    def forward(self,imgs,captions=None):
        # with torch.no_grad():
        image_embeddings = self.image_encoder(imgs)
        outputs = self.new_decoder(image_embeddings,captions)
        # if captions is not None:
        #     poss = self.prompt_encoder.get_dense_pe()
        #     poss = torch.repeat_interleave(poss, image_embeddings.shape[0], dim=0) #[1,256,h/16,w/16]
        #     outputs = self.new_decoder(image_embeddings,captions,poss)
        return outputs
        # # torch.Size([2, 256, 64, 64])
        # # poss = self.prompt_encoder.get_dense_pe()
        # import pdb;pdb.set_trace()

        # for curr_embedding in image_embeddings:
        #     sparse_embeddings, dense_embeddings = self.prompt_encoder(
        #         points=None,
        #         boxes=None,
        #         masks=None,
        #     )
        #     low_res_masks, iou_predictions = self.mask_decoder(
        #         image_embeddings=curr_embedding.unsqueeze(0),
        #         image_pe=self.prompt_encoder.get_dense_pe(),
        #         sparse_prompt_embeddings=sparse_embeddings,
        #         dense_prompt_embeddings=dense_embeddings,
        #         multimask_output=False,
        #     )
        #     import pdb;pdb.set_trace()