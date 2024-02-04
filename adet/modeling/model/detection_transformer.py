import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from adet.layers.deformable_transformer import DeformableTransformer
from adet.utils.misc import (
    NestedTensor,
    inverse_sigmoid_offset,
    nested_tensor_from_tensor_list,
    sigmoid_offset
)
from adet.modeling.model.utils import MLP, cls_head
from adet.layers.pos_encoding import PositionalEncoding2D
from transformers import RobertaModel, RobertaTokenizerFast
# from transformers import BertTokenizer, BertModel
# from transformers import GPT2Tokenizer, GPT2Model



class FeatureResizerEnc(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc_dim = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, encoder_features):
        encoder_features = encoder_features.transpose(0, 1)
        x = self.fc_dim(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


class FeatureResizerDec(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc_dim = nn.Linear(input_feat_size, 50 * output_feat_size, bias=True)
        # self.fc_dim = nn.Linear(input_feat_size, 25 * output_feat_size, bias=True)
        # self.conv_length = nn.Conv1d(512, 200, 1, 1, bias=False)
        self.conv_length = nn.Conv2d(512, 100, 1, 1, bias=False)
        self.layer_norm = nn.LayerNorm(50 * output_feat_size, eps=1e-12)
        # self.layer_norm = nn.LayerNorm(25 * output_feat_size, eps=1e-12)
        # self.bn = nn.BatchNorm1d(200)
        self.bn = nn.BatchNorm2d(100)
        self.dropout = nn.Dropout(dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, encoder_features):
        encoder_features = encoder_features.transpose(0, 1)
        x = self.fc_dim(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        x = self.dropout(x)
        bs, num, _ = x.shape
        x = x.reshape(bs, num, 50, 256)
        # x = x.reshape(bs, num, 25, 256)
        output = self.conv_length(x)
        output = self.bn(output)
        # bs, num, _ = output.shape
        # output = output.reshape(bs, num, 50, 256)

        return output


class DETECTION_TRANSFORMER(nn.Module):
    def __init__(self, cfg, backbone,
                 ###
                 text_encoder_type="/media/WD_2T/chill_research/deepsolo/roberta-base",
                 # text_encoder_type="/media/WD_2T/chill_research/deepsolo/bert-base-uncased",
                 # text_encoder_type="/media/WD_2T/chill_research/deepsolo/gpt2",
                freeze_text_encoder = False,
                 ###
    ):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = backbone

        self.d_model = cfg.MODEL.TRANSFORMER.HIDDEN_DIM
        self.nhead = cfg.MODEL.TRANSFORMER.NHEADS
        self.num_encoder_layers = cfg.MODEL.TRANSFORMER.ENC_LAYERS
        self.num_decoder_layers = cfg.MODEL.TRANSFORMER.DEC_LAYERS
        self.dim_feedforward = cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.TRANSFORMER.DROPOUT
        self.activation = "relu"
        self.return_intermediate_dec = True
        self.num_feature_levels = cfg.MODEL.TRANSFORMER.NUM_FEATURE_LEVELS
        self.dec_n_points = cfg.MODEL.TRANSFORMER.ENC_N_POINTS
        self.enc_n_points = cfg.MODEL.TRANSFORMER.DEC_N_POINTS
        self.pos_embed_scale = cfg.MODEL.TRANSFORMER.POSITION_EMBEDDING_SCALE
        self.num_classes = 1 # text or not text
        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.sigmoid_offset = False

        self.num_proposals = cfg.MODEL.TRANSFORMER.NUM_QUERIES
        self.num_points = cfg.MODEL.TRANSFORMER.NUM_POINTS
        self.point_embed = nn.Embedding(self.num_proposals * self.num_points, self.d_model)

        self.transformer = DeformableTransformer(
            temp=cfg.MODEL.TRANSFORMER.TEMPERATURE,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            return_intermediate_dec=self.return_intermediate_dec,
            num_feature_levels=self.num_feature_levels,
            dec_n_points=self.dec_n_points,
            enc_n_points=self.enc_n_points,
            num_proposals=self.num_proposals,
            num_points=self.num_points
        )

        if self.num_feature_levels > 1:
            strides = [8, 16, 32]
            if cfg.MODEL.BACKBONE.NAME == 'build_swin_backbone':
                if cfg.MODEL.SWIN.TYPE == 'tiny' or 'small':
                    num_channels = [192, 384, 768]
                else:
                    raise NotImplementedError
            elif cfg.MODEL.BACKBONE.NAME == 'build_vitaev2_backbone':
                if cfg.MODEL.ViTAEv2.TYPE == 'vitaev2_s':
                    num_channels = [128, 256, 512]
                else:
                    raise NotImplementedError
            else:
                num_channels = [512, 1024, 2048]

            num_backbone_outs = len(strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(self.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            strides = [32]
            num_channels = [2048]
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            num_channels[0], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                ]
            )
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        self.aux_loss = cfg.MODEL.TRANSFORMER.AUX_LOSS

        # bezier center line proposal after the encoder
        # x_0, y_0, ... , x_3, y_3
        self.bezier_proposal_coord = MLP(self.d_model, self.d_model, 8, 3)
        self.bezier_proposal_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        # task specific heads after the decoder
        self.ctrl_point_coord = MLP(self.d_model, self.d_model, 2, 3)
        self.ctrl_point_class = nn.Linear(self.d_model, self.num_classes)  # text or non-text
        # self.ctrl_point_class = cls_head(self.num_proposals, self.d_model, 1)  # text or non-text
        self.ctrl_point_text = nn.Linear(self.d_model, self.voc_size + 1)  # specific character class for each point
        self.boundary_head_on = cfg.MODEL.TRANSFORMER.BOUNDARY_HEAD
        if self.boundary_head_on:
            self.boundary_offset = MLP(self.d_model, self.d_model, 4, 3)  # to rebuild the text boundary from queries

        prior_prob = 0.01
        bias_value = -np.log((1 - prior_prob) / prior_prob)
        self.bezier_proposal_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_class.bias.data = torch.ones(self.num_classes) * bias_value
        self.ctrl_point_text.bias.data = torch.ones(self.voc_size + 1) * bias_value

        nn.init.constant_(self.bezier_proposal_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.bezier_proposal_coord.layers[-1].bias.data, 0)
        self.transformer.bezier_coord_embed = self.bezier_proposal_coord
        self.transformer.bezier_class_embed = self.bezier_proposal_class

        nn.init.constant_(self.ctrl_point_coord.layers[-1].weight.data, 0)
        nn.init.constant_(self.ctrl_point_coord.layers[-1].bias.data, 0)
        if self.boundary_head_on:
            nn.init.constant_(self.boundary_offset.layers[-1].weight.data, 0)
            nn.init.constant_(self.boundary_offset.layers[-1].bias.data, 0)

        ######################################################################
        # shared prediction heads
        ######################################################################
        num_pred = self.num_decoder_layers
        self.ctrl_point_coord = nn.ModuleList(
            [self.ctrl_point_coord for _ in range(num_pred)]
        )
        self.ctrl_point_class = nn.ModuleList(
            [self.ctrl_point_class for _ in range(num_pred)]
        )
        self.ctrl_point_text = nn.ModuleList(
            [self.ctrl_point_text for _ in range(num_pred)]
        )
        if self.boundary_head_on:
            self.boundary_offset = nn.ModuleList(
                [self.boundary_offset for _ in range(num_pred)]
            )

        self.transformer.decoder.ctrl_point_coord = self.ctrl_point_coord

        ###
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)
        # self.tokenizer = BertTokenizer.from_pretrained(text_encoder_type)
        # self.text_encoder = BertModel.from_pretrained(text_encoder_type)
        # self.tokenizer = GPT2Tokenizer.from_pretrained(text_encoder_type)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # self.text_encoder = GPT2Model.from_pretrained(text_encoder_type)
        # self.text_encoder.resize_token_embeddings(len(self.tokenizer))
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        self.dec_resizer = FeatureResizerDec(
            input_feat_size=768,
            output_feat_size=self.d_model,
            dropout=0.1,
        )

        # self.enc_resizer = FeatureResizerEnc(
        #     input_feat_size=768,
        #     output_feat_size=self.d_model,
        #     dropout=0.1,
        # )
        # N_steps = cfg.MODEL.TRANSFORMER.HIDDEN_DIM // 2
        # self.pos_encoding = PositionalEncoding2D(N_steps, cfg.MODEL.TRANSFORMER.TEMPERATURE, normalize=True)
        ###

        self.to(self.device)

    def forward(self, samples: NestedTensor, captions):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        ###
        tokenized = self.tokenizer.batch_encode_plus(captions, max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        # tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(self.device)
        encoded_text = self.text_encoder(**tokenized)
        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()
        # # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized_dec = self.dec_resizer(text_memory)
        # text_memory_resized_enc = self.enc_resizer(text_memory)
        # merge_enc = NestedTensor(text_memory_resized_enc.transpose(1,2).unsqueeze(3), text_attention_mask.unsqueeze(2))
        # text_pos_enc = self.pos_encoding(merge_enc)

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
                m = masks[0]
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        # (n_proposal x n_pts, d_model) -> (n_proposal, n_pts, d_model)
        point_embed = self.point_embed.weight.reshape((self.num_proposals, self.num_points, self.d_model)) # not shared

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            # enc_outputs_coord_unact, s, w
            enc_outputs_coord_unact
        ) = self.transformer(srcs, masks, pos, point_embed, text_memory_resized_dec)

        outputs_texts = []
        outputs_coords = []
        outputs_classes = []
        if self.boundary_head_on:
            outputs_bd_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            # hs shape: (bs, n_proposal, n_pts, d_model)
            reference = inverse_sigmoid_offset(reference, offset=self.sigmoid_offset)
            outputs_class = self.ctrl_point_class[lvl](hs[lvl])
            outputs_text = self.ctrl_point_text[lvl](hs[lvl])  # bs, n_proposal, n_pts, voc_size
            tmp = self.ctrl_point_coord[lvl](hs[lvl])
            if self.boundary_head_on:
                tmp_bd = self.boundary_offset[lvl](hs[lvl])

            if reference.shape[-1] == 2:
                tmp += reference
                if self.boundary_head_on:
                    tmp_bd += reference.repeat(1, 1, 1, 2)
            else:
                raise NotImplementedError

            outputs_coord = sigmoid_offset(tmp, offset=self.sigmoid_offset)
            if self.boundary_head_on:
                outputs_bd_coord = sigmoid_offset(tmp_bd, offset=self.sigmoid_offset)
                outputs_bd_coords.append(outputs_bd_coord)

            outputs_classes.append(outputs_class)
            outputs_texts.append(outputs_text)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_text = torch.stack(outputs_texts)
        outputs_coord = torch.stack(outputs_coords)
        if self.boundary_head_on:
            outputs_bd_coord = torch.stack(outputs_bd_coords)

        out = {
            'pred_logits': outputs_class[-1],
            'pred_text_logits': outputs_text[-1],
            'pred_ctrl_points': outputs_coord[-1],
            'pred_bd_points': outputs_bd_coord[-1] if self.boundary_head_on else None
        }

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class,
                outputs_text,
                outputs_coord,
                outputs_bd_coord if self.boundary_head_on else None
            )

        enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_beziers': enc_outputs_coord
        }

        # return out, s, w
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_text, outputs_coord, outputs_bd_coord):
        if outputs_bd_coord is not None:
            return [
                {'pred_logits': a, 'pred_text_logits': b, 'pred_ctrl_points': c, 'pred_bd_points': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_text[:-1], outputs_coord[:-1], outputs_bd_coord[:-1])
            ]
        else:
            return [
                {'pred_logits': a, 'pred_text_logits': b, 'pred_ctrl_points': c}
                for a, b, c in zip(outputs_class[:-1], outputs_text[:-1], outputs_coord[:-1])
            ]