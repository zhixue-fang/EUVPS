from math import ceil
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from vita.modeling.transformer_decoder.vita import VITA
from euvps.modeling.proposal_generative_head.cffm_head import CFFMHead_clips_resize1_8
from vita.vita_model import Vita
import numpy as np


class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, nIn, nOut, add=False):
        super(DilatedParallelConvBlockD2, self).__init__()
        n = int(np.ceil(nOut / 2.))
        n2 = nOut - n

        self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, bias=False)
        self.conv1 = nn.Conv2d(n, n, 3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = nn.Conv2d(n2, n2, 3, stride=1, padding=2, dilation=2, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.add = add

    def forward(self, input):
        in0 = self.conv0(input)
        in1, in2 = torch.chunk(in0, 2, dim=1)
        b1 = self.conv1(in1)
        b2 = self.conv2(in2)
        output = torch.cat([b1, b2], dim=1)

        if self.add:
            output = input + output
        output = self.bn(output)

        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@SEM_SEG_HEADS_REGISTRY.register()
class euvps_sem_seg_head(VITA):
    """"""
    def __init__(self, cfg, in_channel):
        super().__init__(
            cfg=cfg,
            in_channels=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            aux_loss=cfg.MODEL.VITA.DEEP_SUPERVISION,
        )

        self.num_frames = cfg.MODEL.LEN_CLIP_WINDOW
        self.proc_clip_len = cfg.MODEL.PER_LEN_CLIP_WINDOW

        hidden_dim = cfg.MODEL.VITA.HIDDEN_DIM
        num_queries = cfg.MODEL.VITA.NUM_OBJECT_QUERIES

        self.in_dims = cfg.MODEL.IN_CHANNEL_DIMS
        self.in_channels = cfg.MODEL.IN_CHANNELS
        self.query_in_channels = cfg.MODEL.QUERY_PROC_CHANNELS
        self.shape_dim = cfg.MODEL.SHAPE_DIM
        self.shape_layers = cfg.MODEL.SHAPE_LAYER
        self.region_embed = [MLP(in_dim[0] * in_dim[1], hidden_dim, self.shape_dim, self.shape_layers).cuda() for in_dim in self.in_dims]
        self.gen_region_block = [DilatedParallelConvBlockD2(in_dim, num_queries // 4).cuda() for in_dim in self.query_in_channels]

        self.pre_memory_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_memory_embed_v = nn.Linear(hidden_dim, hidden_dim)

        self.pre_query_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_query_embed_v = nn.Linear(hidden_dim, hidden_dim)

        self.query_feat = nn.Embedding(num_queries, hidden_dim)

        norm_cfg = dict(type='SyncBN', requires_grad=True)
        
        self.proposal_generative_head = CFFMHead_clips_resize1_8(
            in_channels=cfg.MODEL.IN_CHANNELS,
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=hidden_dim,
            dropout_ratio=0.1,
            num_classes=1,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=hidden_dim, depths=2),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            num_clips=cfg.MODEL.LEN_CLIP_WINDOW,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            short_proc=cfg.MODEL.SHORT_PROC
        )

    def transformer_enc_dec(self, frame_query, pre_memory, output, clip_len):
        if not self.training:
            frame_query = frame_query[[-1]]

        pre_memory_k = pre_memory["k"]
        pre_memory_v = pre_memory["v"]

        L, BT, fQ, C = frame_query.shape

        T = clip_len
        B = BT // T if self.training else 1

        frame_query = frame_query.reshape(L*B, T, fQ, C)
        frame_query = frame_query.permute(1, 2, 0, 3).contiguous()
        frame_query = self.input_proj_dec(frame_query)  # T, fQ, LB, C

        if self.window_size > 0:
            pad = int(ceil(T / self.window_size)) * self.window_size - T
            _T = pad + T
            frame_query = F.pad(frame_query, (0, 0, 0, 0, 0, 0, 0, pad))   # _T, fQ, LB, C
            enc_mask = frame_query.new_ones(L*B, _T).bool()         # LB, _T
            enc_mask[:, :T] = False
        else:
            enc_mask = None

        frame_query = self.encode_frame_query(frame_query, enc_mask)
        frame_query = frame_query[:T].flatten(0, 1)               # TfQ, LB, C

        if self.use_sim:
            pred_fq_embed = self.sim_embed_frame(frame_query)   # TfQ, LB, C
            pred_fq_embed = pred_fq_embed.transpose(0, 1).reshape(L, B, T, fQ, C)
        else:
            pred_fq_embed = None

        src = self.src_embed(frame_query)   # TfQ, LB, C
        dec_pos = self.fq_pos.weight[None, :, None, :].repeat(T, 1, L*B, 1).flatten(0, 1)  # TfQ, LB, C

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, T*B, 1)  # cQ, LB, C

        cQ, LB, C = output.shape

        # pre query embed
        pre_query_k = self.pre_query_embed_k(output)  # cQ, LB, C
        pre_query_v = self.pre_query_embed_v(output)  # cQ, LB, C

        # pre memory read
        if pre_memory_k and pre_memory_v:
            pre_memory_k = torch.cat(pre_memory_k).flatten(1, 2)  # M, LB, cQ, C
            pre_memory_v = torch.cat(pre_memory_v).flatten(1, 2)  # M, LB, cQ, C
        else:
            pre_memory_k = torch.empty((0, LB, cQ, C), device=output.device)
            pre_memory_v = torch.empty((0, LB, cQ, C), device=output.device)

        qk_mk = torch.einsum("qbc, mbpc -> bqmp", pre_query_k, pre_memory_k)  # LB, cQ, M, cQ
        qk_mk = torch.einsum("bqmq -> bqm", qk_mk)  # LB, cQ, M
        qk_mk = F.softmax(qk_mk, dim=2)
        qk_mk_mv = torch.einsum("bqm, mbqc-> qbc", qk_mk, pre_memory_v)  # cQ, B, C

        pre_query_v = pre_query_v + qk_mk_mv  # cQ, LB, C
        output = output + pre_query_v        # cQ, LB, C

        decoder_outputs = []

        for i in range(self.num_layers):
            attention: cross-attention
            output = self.transformer_cross_attention_layers[i](
                output,
                src,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=dec_pos,
                query_pos=query_embed
            )
            # self-attention
            output = self.transformer_self_attention_layers[i](
                output,
                tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            if (self.training and self.aux_loss) or (i == self.num_layers - 1):
                dec_out = self.decoder_norm(output)  # cQ, LB, C
                dec_out = dec_out.transpose(0, 1)   # LB, cQ, C
                decoder_outputs.append(dec_out.view(B, T, self.num_queries, C))

        decoder_outputs = torch.stack(decoder_outputs, dim=0)   # D, L, B, cQ, C

        pred_mask_embed = self.mask_embed(decoder_outputs)

        memory_input = decoder_outputs[-1]

        pre_memory_k = self.pre_memory_embed_k(memory_input)[None]  # 1, L, B, cQ, C
        pre_memory_v = self.pre_memory_embed_v(memory_input)[None]  # 1, L, B, cQ, C

        out = {
            'pred_mask_embed': pred_mask_embed[-1],
            'pred_fq_embed': pred_fq_embed,
            'pre_memory': {"k": pre_memory_k, "v": pre_memory_v},
        }

        return out, output

    def forward(self, features):
        """"""
        stage_features, mask_proposals = self.proposal_generative_head(features)

        region_features = []
        for ori_feat, stage_feat, gen_region in zip(features, stage_features, self.gen_region_block):
            region_features.append(gen_region(ori_feat + stage_feat))

        region_feat_embed = []
        for feat, reg_embed in zip(region_features, self.region_embed):
            region_feat_embed.append(reg_embed(feat.view(feat.shape[0], feat.shape[1], -1)))

        shape_query = torch.cat(region_feat_embed, dim=1).squeeze(2)

        l, cq, c = shape_query.shape

        clip_len = self.proc_clip_len

        pre_memory = {"k": [], "v": []}
        output_q = self.query_feat.weight.unsqueeze(1).repeat(1, clip_len, 1)  # cQ, LB, C

        output = []

        for i in range(l // clip_len):
            outputs, output_q = self.transformer_enc_dec(
                shape_query[i*clip_len:(i+1)*clip_len].unsqueeze(0),
                pre_memory,
                output_q,
                clip_len
            )

            out = torch.einsum(
                "blqc, blqhw -> blhw",
                outputs["pred_mask_embed"],
                mask_proposals[:, i*clip_len:(i+1)*clip_len, :, :, :]
            )

            output.append(out)
            pre_memory["k"].append(outputs["pre_memory"]["k"])
            pre_memory["v"].append(outputs["pre_memory"]["v"])

        return torch.cat(output, dim=1)

