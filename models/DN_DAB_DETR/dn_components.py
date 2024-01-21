# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes

def prepare_for_dn(dn_args, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    prepare for dn components in forward function
    Args:
        dn_args: (targets, args.scalar, args.label_noise_scale,
                                                             args.box_noise_scale, args.num_patterns) from engine input
        dn_args:
        1. targets
        2. args.scalar, dn group的数量 5
        3. args.label_noise_scale,  noise ratio to flip 0.2
        4. args.box_noise_scale, box noise scale to shift and scale 0.4
        5. args.num_patterns anchor detr的pattern 默认是0

        embedweight: positional queries as anchor (300,4)
        training: whether it is training or inference
        num_queries: number of queries 300
        num_classes: number of classes 91
        hidden_dim: transformer hidden dimenstion 256
        label_enc: label encoding embedding (92,255)

    Returns: input_query_label, input_query_bbox, attn_mask, mask_dict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if training:
        targets, scalar, label_noise_scale, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = 0

    if num_patterns == 0:
        num_patterns = 1
    indicator0 = torch.zeros([num_queries * num_patterns, 1]).to(device) # (300,1)
    tgt = label_enc(torch.tensor(num_classes).to(device)).repeat(num_queries * num_patterns, 1) # (300,255)
    tgt = torch.cat([tgt, indicator0], dim=1) # (300,256)
    refpoint_emb = embedweight.repeat(num_patterns, 1) # (300,4)
    if training:
        # 都是1的tensor, 是一个list，数量是bs的大小，里面的tensor的大小是每个image中的target的数量(每张图片有几个gt)
        known = [(torch.ones_like(t['labels'])).to(device) for t in targets]
        # 返回一个二维张量，其中每一行都是非零值的索引(gt的索引)
        know_idx = [torch.nonzero(t) for t in known]
        # 是一个list，每个item是每个image包含的gt的数量
        known_num = [sum(k) for k in known]
        # you can uncomment this to use fix number of dn queries
        # if int(max(known_num))>0:
        #     scalar=scalar//int(max(known_num))

        # can be modified to selectively denosie some label or boxes; also known label prediction
        unmask_bbox = unmask_label = torch.cat(known) # 所有的1 cat到一起
        labels = torch.cat([t['labels'] for t in targets]) # 取出所有gt的label
        boxes = torch.cat([t['boxes'] for t in targets]) # 取出所有gt的box
        # 标识属于哪个图片 like tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2], device='cuda:0')
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        # 返回一个二维张量，其中每一行都是非零值的索引,这里用不用他俩相加都没有什么区别
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # add noise 重复5组，拉平 (5N,)-(5*12,)
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        known_labels_expaned = known_labels.clone() # (60)
        known_bbox_expand = known_bboxs.clone() # (60)

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float()) # 生成0-1内的随机数
            # 选择低于阈值的id
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
            # 随意更改这些gt
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            # 把加噪的值加入到原始label上来
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_expand) # (5N,4)
            # 宽高的一半放到中心坐标位置
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            # 加上随机的偏移量
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0),
                                           diff).to(device) * box_noise_scale
            # 裁剪，防止溢出
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        # add dn part indicator
        # (5N, 1) 全是1的tensor
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).to(device)
        # (5N,256) tgt相关的在最后拼的是0，tgt是给正常的匹配部分使用的，这里拼的是1
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1)
        # 对坐标取反函数 (N,4), 对应于特征图上的坐标
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        # bs中最多的target的数量
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        # (5*max_gt_num,256)
        padding_label = torch.zeros(pad_size, hidden_dim).to(device)
        # (5*max_gt_num,4)
        padding_bbox = torch.zeros(pad_size, 4).to(device)

        # 拼在前面的是去噪部分的，在后面的是正常匹配部分的
        # 与正常的，需要经过匈牙利匹配的那一部分的query拼接在一起
        input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
        # refpoint_emb是正常的，给到300个预测使用的部分
        input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)

        # map in order
        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            # 各个image的合并在一起了, 如tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 1])
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            # 对各个group加上了偏移，这个偏移是这些batch中最大的gt的数量
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid): # known_bid是各个gt属于某个image的那个id
            # known_bid (5N,) map_known_indice (5N,) 替换对应的embed
            # input_query_label为[bs,300+5N,256] 其实也就是替换了各个image的前5N个
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries * num_patterns
        # 初始值都是False,0表示可以看见，1表示被Mask，看不见 [300+5N,300+5N]
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # 第一个 pad_size: 表示是这些group之后的，就是正常需要进行匹配的那300个，在那300个里面，不能看见前面的那些group
        # 因此第二个取值是 :pad_size
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            # 第一组
            if i == 0:
                # single_pad 是bs中拥有最多gt的gt的数量
                # 看不到他后面的所有group
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            # 最后一组
            if i == scalar - 1:
                # 看不到他前面的所有group
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            # 中间组
            else:
                # 看不到他后面的group
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                # 也看不到他前面的group
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'know_idx': know_idx,
            'pad_size': pad_size
        }
    else:  # no dn for inference
        input_query_label = tgt.repeat(batch_size, 1, 1)
        input_query_bbox = refpoint_emb.repeat(batch_size, 1, 1)
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1)
    input_query_bbox = input_query_bbox.transpose(0, 1)

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes']=(output_known_class,output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to(device),
            'tgt_loss_giou': torch.as_tensor(0.).to(device),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt

    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
        box_ops.box_cxcywh_to_xyxy(src_boxes),
        box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to(device),
            'tgt_class_error': torch.as_tensor(0.).to(device),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :-1]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, training, aux_num, focal_alpha):
    """
    compute dn loss in criterion
    Args:
        mask_dict: a dict for dn information
        training: training or inference flag
        aux_num: aux loss number
        focal_alpha:  for focal loss
    """
    losses = {}
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, output_known_class, output_known_coord, \
        num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to(device)
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to(device)
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to(device)
        losses['tgt_class_error'] = torch.as_tensor(0.).to(device)

    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to(device)
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to(device)
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to(device)
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to(device)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses