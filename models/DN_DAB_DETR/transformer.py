# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
from typing import Optional, List
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention

eps = 1e-40
class channel_selection(nn.Module):
    def __init__(self, D_in, layer_num=-1):
        super(channel_selection, self).__init__()

        '''
        [(a_i + gamma_i)(1 + u_i * gamma) + w_i * gamma] * neuron
        '''

        self.prune_a = nn.Parameter(1. / D_in * torch.ones(1, 1, D_in), requires_grad=False)
        self.prune_gamma = nn.Parameter(0. * torch.ones(1, 1, D_in), requires_grad=False)
        # self.prune_u = nn.Parameter(0. * torch.ones(1, D_in, 1, 1), requires_grad=False)
        self.prune_w = nn.Parameter(0. * torch.ones(1, 1, D_in), requires_grad=False)
        self.prune_lsearch = nn.Parameter(0. * torch.tensor(1.), requires_grad=False)
        self.scale = D_in

        self.layer_num = layer_num
        self.D_in = D_in # 256
        self.device = 'cuda'
        self.mode = 'train'
        self.zeros = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.ones = nn.Parameter(torch.ones(1, 1, self.D_in), requires_grad=False)

    def forward(self, x):
        res = torch.mul(self.scale * x, (
                (self.prune_a + self.prune_gamma) * (1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w))
        return res

    def pforward(self, x, chosen_layer):
        if self.layer_num == chosen_layer:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                    1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), x
        else:
            return torch.mul(self.scale * x, ((self.prune_a + self.prune_gamma) * (
                    1. - self.prune_lsearch) + self.prune_lsearch * self.prune_w)), self.zeros

    def turn_off(self, src_param, is_lsearch=False):
        if not is_lsearch:
            tar_param = nn.Parameter(torch.zeros(1, 1, self.D_in), requires_grad=False)
        else:
            tar_param = nn.Parameter(torch.tensor(1.), requires_grad=False)
        tar_param.data = src_param.data.clone()

        return tar_param

    def switch_mode(self, mode='train'):
        self.mode = mode
        if mode == 'train':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'prune':
            self.prune_gamma.requires_grad = True
            self.prune_lsearch.requires_grad = True
            self.prune_a = self.turn_off(self.prune_a)

        elif mode == 'adjust_a':
            self.prune_gamma = self.turn_off(self.prune_gamma)
            self.prune_lsearch = self.turn_off(self.prune_lsearch, True)
            self.prune_a.requires_grad = True
        else:
            raise NotImplementedError

    def empty_all_eps(self):
        self.prune_a.data = -eps * self.prune_a.data

    def init_lsearch(self, neuron_index):
        self.prune_gamma.data = 0. * self.prune_gamma.data
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        if neuron_index >= 0:
            self.prune_w[:, :, neuron_index] += 1.

    def update_alpha(self, neuron_index, lsearch):
        self.prune_a.data *= (1. - lsearch)
        self.prune_a[:, :, neuron_index] += lsearch
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    def update_alpha_back(self, neuron_index, lsearch):
        # self.prune_a.data *= (1. - lsearch)
        self.prune_a[:, :, neuron_index] *= 0
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    # def set_alpha_to_init(self):
    #    self.prune_a.data = 0. * self.prune_a.data
    #    self.prune_a.data += 1./self.D_in * self.ones#* torch.ones(1, self.D_in, 1, 1).to(self.device)
    #    self.prune_w.data = 0. * self.prune_w.data
    #    self.prune_lsearch.data = 0. * self.prune_lsearch.data
    #    self.prune_gamma.data = 0. * self.prune_gamma.data

    def set_alpha_to_init(self, prunable_neuron):
        if len(prunable_neuron) != self.prune_a.shape[2]:
            print('dim of prunable_neuron error!')
            raise ValueError

        self.prune_a.data = 0. * self.prune_a.data

        num_prunable_neuron = prunable_neuron.sum()
        for _ in range(len(prunable_neuron)):
            if prunable_neuron[_] > 0:
                self.prune_a.data[0, 0, _] += 1. / num_prunable_neuron

        # self.prune_a.data += 1./self.D_in * torch.ones(1, self.D_in, 1, 1).to(self.device)
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

    def assign_alpha(self, alpha):
        self.prune_a.data = 0. * self.prune_a.data
        self.prune_a.data += alpha
        self.prune_w.data = 0. * self.prune_w.data
        self.prune_lsearch.data = 0. * self.prune_lsearch.data
        self.prune_gamma.data = 0. * self.prune_gamma.data

# 比例剪枝
# class channel_selection(nn.Module):
#     def __init__(self, num_channels):
#         """
#         Initialize the `indexes` with all one vector with the length same as the number of channels.
#         During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
#         """
#         super(channel_selection, self).__init__()
#         self.indexes = nn.Parameter(torch.ones(num_channels))
#
#     def forward(self, input_tensor):
#         """
#         Parameter
#         ---------
#         input_tensor: (B, num_patches + 1, dim).
#         """
#         output = input_tensor.mul(self.indexes)
#         return output

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

def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4, # xywh 四个值
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        # self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed, tgt, attn_mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)        
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed) #(1292,4,256)

        if self.num_patterns > 0:
            l = tgt.shape[0]
            tgt[l - self.num_queries * self.num_patterns:] += \
                self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)

        hs, references = self.decoder(tgt, memory, tgt_mask=attn_mask, memory_key_padding_mask=mask, # tgt(300,4,256)
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed)
        return hs, references



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 比DETR的Encoder多了一个这个MLP
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
                    d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        # 每个坐标是 256/2, 四个坐标，一共是512，output的channel依然是256
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer


        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        # decoder 是否使用query_pos
        # 第一层的留下，剩下的都是None
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        # 限制0-1 [300,bs,4]
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            # [300,bs,4]
            obj_center = reference_points[..., :self.query_dim]     # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            # 三角函数高频编码 [300,bs,512]
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # 经过一个MLP [300,bs,256]
            query_pos = self.ref_point_head(query_sine_embed) 

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            # 给cross attention使用的，结构图中第二行的MLP的左边那个MLP
            query_sine_embed = query_sine_embed[...,:self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                # 结构图中第二行的MLP的右边那个MLP
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)


            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2), 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # 重新实现self_attn
        self.in_proj_weight = nn.Parameter(torch.empty((3 * d_model, d_model)))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * d_model))

        self.out_proj = nn.Linear(d_model, d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        # 新加入
        self.nhead = nhead
        self.dropout_p = dropout
        self.select1 = channel_selection(d_model)
        self.select2 = channel_selection(dim_feedforward)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 原代码
        # q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                       key_padding_mask=src_key_padding_mask)[0]
        # 重写self_attn==============================================================
        tgt_len, bsz, _ = src.shape  # [950, 4, 256]
        # 进行channel_selection
        src = self.select1(src)
        # q、k加入pos, v就是输入的src
        q = k = self.with_pos_embed(src, pos) # 1.[950, 4, 256]  2.[3, 256, 256]
        # 切分self.in_proj_weight和self.in_proj_bias
        w_q, w_k, w_v = self.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.in_proj_bias.chunk(3)
        # q、k、v先进行线性映射
        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)
        v = F.linear(src, w_v, b_v)
        # reshape为多头
        q = q.contiguous().view(tgt_len, bsz * self.nhead, -1).transpose(0, 1)  # [4*8, 950, 32]
        k = k.contiguous().view(tgt_len, bsz * self.nhead, -1).transpose(0, 1)
        v = v.contiguous().view(tgt_len, bsz * self.nhead, -1).transpose(0, 1)
        # 计算mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.view(bsz, 1, 1, tgt_len). \
                expand(-1, self.nhead, -1, -1).reshape(bsz * self.nhead, 1, tgt_len)  # [4, 950]->[4*8, 1, 950]
            if src_mask is None:
                src_mask = src_key_padding_mask
        # convert mask to float
        if src_mask is not None and src_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(src_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(src_mask, float("-inf"))
            src_mask = new_attn_mask
        # 注意力计算
        B, Nt, E = q.shape # [4*8, 950, 32]
        q = q / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        if src_mask is not None:
            attn = torch.baddbmm(src_mask, q, k.transpose(-2, -1)) # [4*8,950,950]
        else:
            attn = torch.bmm(q, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        if not self.training:
            self.dropout_p = 0.0
        attn = F.dropout(attn, p=self.dropout_p)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        attn_output = torch.bmm(attn, v) # [4*8,950,32]
        # 过输出的Linear层
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, -1)  # [950*4,256]
        attn_output = self.out_proj(attn_output) # [950*4,256]
        src2 = attn_output.view(tgt_len, bsz, attn_output.size(1)) # [950,4,256]
        # 重写self_attn end==============================================================
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.linear2(self.select2(self.dropout(self.activation(self.linear1(src)))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model, is_cross=False)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model*2, nhead, dropout=dropout, vdim=d_model, is_cross=True)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

        # 新加入
        self.dropout_p = dropout
        self.select1 = channel_selection(d_model)
        self.select2 = channel_selection(d_model)
        self.select3 = channel_selection(dim_feedforward)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     query_sine_embed = None,
                     is_first = False):
                     
        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # 加入选择
            tgt = self.select1(tgt) # (300,4,256)
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            # num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask, is_cross=False)[0] # (300,4,256)

            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # 加入选择
        tgt = self.select2(tgt) # (300,4,256)
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model//self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q, # (300,4,512)
                                   key=k, # (1444,4,512)
                                   value=v, attn_mask=memory_mask, # (1444,4,256)
                                   key_padding_mask=memory_key_padding_mask, is_cross=True)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2) #(300,4,256)
        tgt = self.norm2(tgt)
        # 原代码
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.linear2(self.select3(self.dropout(self.activation(self.linear1(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
