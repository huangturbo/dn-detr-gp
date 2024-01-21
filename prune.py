# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import collections
import datetime
import json
import random
import time
from pathlib import Path
from os import path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_DABDETR, build_dab_deformable_detr, \
    build_dab_deformable_detr_deformable_encoder_only, build_dab_dino_deformable_detr
from models.DN_DAB_DETR.transformer import channel_selection
from util import logger
from util.utils import load_partial_weight_dab, transplant_weight_dab



def get_args_parser():
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)

    # about dn args
    parser.add_argument('--use_dn', default=True, action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true",
                        help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true",
                        help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true",
                        help="use look forward twice from DINO.")

    # about lr
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float,
                        help='learning rate for backbone')

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--override_resumed_lr_drop', default=False, action='store_true')
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--modelname', '-m', type=str, default='dn_dab_detr',
                        choices=['dn_dab_detr', 'dn_dab_deformable_detr',
                        'dn_dab_deformable_detr_deformable_encoder_only', 'dn_dab_dino_deformable_detr'])
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--pe_temperatureH', default=20, type=int,
                        help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int,
                        help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str,
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")

    # * Transformer
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str,
                        help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.0, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_results', default=300, type=int,
                        help="Number of detection results")
    parser.add_argument('--pre_norm', action='store_true',
                        help="Using pre-norm in the Transformer blocks.")
    parser.add_argument('--num_select', default=300, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,
                        help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true',
                        help="Random init the x,y of anchor boxes and freeze them.")

    # for DAB-Deformable-DETR
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help="Using two stage variant for DAB-Deofrmable-DETR")
    parser.add_argument('--num_feature_levels', default=4, type=int,
                        help='number of feature levels')
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help="number of deformable attention sampling points in decoder layers")
    parser.add_argument('--enc_n_points', default=4, type=int,
                        help="number of deformable attention sampling points in encoder layers")


    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=1, type=float,
                        help="loss coefficient for cls")
    parser.add_argument('--mask_loss_coef', default=1, type=float,
                        help="loss coefficient for mask")
    parser.add_argument('--dice_loss_coef', default=1, type=float,
                        help="loss coefficient for dice")
    parser.add_argument('--bbox_loss_coef', default=5, type=float,
                        help="loss coefficient for bbox L1 loss")
    parser.add_argument('--giou_loss_coef', default=2, type=float,
                        help="loss coefficient for bbox GIOU loss")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help="alpha for focal loss")


    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/home/users/xjs/DataSets/coco') #/home/users/xjs/DataSets/coco /mnt/public/datasets/coco
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true',
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='output_ratio/pruned_train', help='path where to save, empty for no saving') # TODO
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda:1', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='output_ratio/ts_train/checkpoint.pth', help='resume from checkpoint') # TODO # resume/checkpoint_optimized_44.7ap.pth
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+',
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True, action='store_true', help="eval only. w/o Training.")
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--debug', action='store_true',
                        help="For debug only. It will perform only a few steps during trainig and val.")
    parser.add_argument('--find_unused_params', default=False, action='store_true')

    parser.add_argument('--save_results', action='store_true',
                        help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true',
                        help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    # prune
    parser.add_argument('--cfgs', type=list, default=None, help='cfg after prune')
    return parser

def build_model_main(args):
    if args.modelname.lower() == 'dn_dab_detr':
        model, criterion, postprocessors = build_DABDETR(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr':
        model, criterion, postprocessors = build_dab_deformable_detr(args)
    elif args.modelname.lower() == 'dn_dab_deformable_detr_deformable_encoder_only':
        model, criterion, postprocessors = build_dab_deformable_detr_deformable_encoder_only(args)
    elif args.modelname.lower() == 'dn_dab_dino_deformable_detr':
        model, criterion, postprocessors = build_dab_dino_deformable_detr(args)
    else:
        raise NotImplementedError

    return model, criterion, postprocessors

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model_main(args)
    # model_dict = model.state_dict()
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume and (args.resume.startswith('https') or path.exists(args.resume)):
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        # load_partial_weight_dab(model_without_ddp, checkpoint['model'])
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1

            if args.drop_lr_now:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.1

    # if args.eval:
    #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                           data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
    #     if args.output_dir:
    #         utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    #     return
    # print("Test model before prune ...")
    test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                          data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)

    # 剪枝
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, channel_selection):
    #         print("{}:{}".format(k, m))

    # 计算总通道数
    encoder_self_attn_total = 0
    encoder_ffn_total = 0

    decoder_self_attn_total = 0
    decoder_mh_attn_total = 0
    decoder_ffn_total = 0

    for k, m in enumerate(model_without_ddp.modules()):
        if isinstance(m, channel_selection):
            if k in [14, 26, 38, 50, 62, 74]:  # encoder6
                encoder_self_attn_total += m.indexes.data.shape[0]
            elif k in [15, 27, 39, 51, 63, 75]:
                encoder_ffn_total += m.indexes.data.shape[0]
            elif k in [108, 136, 164, 192, 220, 248]:  # decoder6
                decoder_self_attn_total += m.indexes.data.shape[0]
            elif k in [109, 137, 165, 193, 221, 249]:
                decoder_mh_attn_total += m.indexes.data.shape[0]
            elif k in [110, 138, 166, 194, 222, 250]:
                decoder_ffn_total += m.indexes.data.shape[0]
    print("encoder_self_attn_total: ", encoder_self_attn_total)
    print("encoder_ffn_total: ", encoder_ffn_total)
    print("decoder_self_attn_total: ", decoder_self_attn_total)
    print("decoder_mh_attn_total: ", decoder_mh_attn_total)
    print("decoder_ffn_total: ", decoder_ffn_total)

    # 取出评测参数
    encoder_self_attn_bn = torch.zeros(encoder_self_attn_total)
    encoder_ffn_bn = torch.zeros(encoder_ffn_total)

    decoder_self_attn_bn = torch.zeros(decoder_self_attn_total)
    decoder_mh_attn_bn = torch.zeros(decoder_mh_attn_total)
    decoder_ffn_bn = torch.zeros(decoder_ffn_total)

    encoder_self_attn_index = 0
    encoder_ffn_index = 0
    decoder_self_attn_index = 0
    decoder_mh_attn_index = 0
    decoder_ffn_index = 0
    for k, m in enumerate(model_without_ddp.modules()):
        if isinstance(m, channel_selection):
            if k in [14, 26, 38, 50, 62, 74]:  # encoder6
                size = m.indexes.data.shape[0]
                encoder_self_attn_bn[
                encoder_self_attn_index:(encoder_self_attn_index + size)] = m.indexes.data.abs().clone()
                encoder_self_attn_index += size
            elif k in [15, 27, 39, 51, 63, 75]:
                size = m.indexes.data.shape[0]
                encoder_ffn_bn[encoder_ffn_index:(encoder_ffn_index + size)] = m.indexes.data.abs().clone()
                encoder_ffn_index += size
            elif k in [108, 136, 164, 192, 220, 248]:  # decoder6
                size = m.indexes.data.shape[0]
                decoder_self_attn_bn[
                decoder_self_attn_index:(decoder_self_attn_index + size)] = m.indexes.data.abs().clone()
                decoder_self_attn_index += size
            elif k in [109, 137, 165, 193, 221, 249]:
                size = m.indexes.data.shape[0]
                decoder_mh_attn_bn[decoder_mh_attn_index:(decoder_mh_attn_index + size)] = m.indexes.data.abs().clone()
                decoder_mh_attn_index += size
            elif k in [110, 138, 166, 194, 222, 250]:
                size = m.indexes.data.shape[0]
                decoder_ffn_bn[decoder_ffn_index:(decoder_ffn_index + size)] = m.indexes.data.abs().clone()
                decoder_ffn_index += size

    # 根据剪枝比例计算阈值
    encoder_self_attn_ratio = 0.1
    encoder_ffn_ratio = 0.5

    decoder_self_attn_ratio = 0.1
    decoder_mh_attn_ratio = 0.1
    decoder_ffn_ratio = 0.5


    encoder_self_attn_y, _ = torch.sort(encoder_self_attn_bn)
    encoder_self_attn_thre_index = int(encoder_self_attn_total * encoder_self_attn_ratio)  # 阈值索引
    encoder_self_attn_thre = encoder_self_attn_y[encoder_self_attn_thre_index]  # 阈值
    encoder_ffn_y, _ = torch.sort(encoder_ffn_bn)
    encoder_ffn_thre_index = int(encoder_ffn_total * encoder_ffn_ratio)  # 阈值索引
    encoder_ffn_thre = encoder_ffn_y[encoder_ffn_thre_index]  # 阈值

    decoder_self_attn_y, _ = torch.sort(decoder_self_attn_bn)
    decoder_self_attn_thre_index = int(decoder_self_attn_total * decoder_self_attn_ratio)  # 阈值索引
    decoder_self_attn_thre = decoder_self_attn_y[decoder_self_attn_thre_index]  # 阈值
    decoder_mh_attn_y, _ = torch.sort(decoder_mh_attn_bn)
    decoder_mh_attn_thre_index = int(decoder_mh_attn_total * decoder_mh_attn_ratio)  # 阈值索引
    decoder_mh_attn_thre = decoder_mh_attn_y[decoder_mh_attn_thre_index]  # 阈值
    decoder_ffn_y, _ = torch.sort(decoder_ffn_bn)
    decoder_ffn_thre_index = int(decoder_ffn_total * decoder_ffn_ratio)  # 阈值索引
    decoder_ffn_thre = decoder_ffn_y[decoder_ffn_thre_index]  # 阈值

    # 计算剩余通道、mask列表
    encoder_cfg = []
    encoder_cfg_mask = []
    decoder_cfg = []
    decoder_cfg_mask = []

    for k, m in enumerate(model_without_ddp.modules()):
        if isinstance(m, channel_selection):
            if k in [14, 26, 38, 50, 62, 74]:  # encoder的SA
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(encoder_self_attn_thre).float().to(device)
                weight_copy_y, weight_copy_i = torch.sort(weight_copy, descending=True)
                aux_index = (weight_copy_y <= encoder_self_attn_thre).nonzero()
                i = 0
                while (torch.sum(mask) % args.nheads != 0 or torch.sum(mask) == 0):  # 必须满足heads整除
                    # print(torch.sum(mask))
                    mask[weight_copy_i[aux_index[i]]] = 1
                    i += 1
                encoder_cfg.append(int(torch.sum(mask)))  # 剩余通道
                encoder_cfg_mask.append(mask.clone())  # mask列表
            elif k in [15, 27, 39, 51, 63, 75]: # encoder的FFN
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(encoder_ffn_thre).float().to(device)
                if int(torch.sum(mask)) < 2: # 至少保留2个通道
                    mask[0] = 1
                    mask[1] = 1
                encoder_cfg.append(int(torch.sum(mask)))  # 剩余通道
                encoder_cfg_mask.append(mask.clone())  # mask列表
            elif k in [108, 136, 164, 192, 220, 248]: # decoder的SA
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(decoder_self_attn_thre).float().to(device)
                weight_copy_y, weight_copy_i = torch.sort(weight_copy, descending=True)
                aux_index = (weight_copy_y <= decoder_self_attn_thre).nonzero()
                i = 0
                while (torch.sum(mask) % args.nheads != 0 or torch.sum(mask) == 0):  # 必须满足heads整除
                    # print(torch.sum(mask))
                    mask[weight_copy_i[aux_index[i]]] = 1
                    i += 1
                decoder_cfg.append(int(torch.sum(mask)))  # 剩余通道
                decoder_cfg_mask.append(mask.clone())  # mask列表
            elif k in [109, 137, 165, 193, 221, 249]: # decoder的MHSA
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(decoder_mh_attn_thre).float().to(device)
                weight_copy_y, weight_copy_i = torch.sort(weight_copy, descending=True)
                aux_index = (weight_copy_y <= decoder_mh_attn_thre).nonzero()
                i = 0
                while (torch.sum(mask) % args.nheads != 0 or torch.sum(mask) == 0):  # 必须满足heads整除
                    # print(torch.sum(mask))
                    mask[weight_copy_i[aux_index[i]]] = 1
                    i += 1
                decoder_cfg.append(int(torch.sum(mask)))  # 剩余通道
                decoder_cfg_mask.append(mask.clone())  # mask列表
            else: #k in [110, 138, 166, 194, 222, 250]:  # decoder的FFN
                weight_copy = m.indexes.data.abs().clone()
                mask = weight_copy.gt(decoder_ffn_thre).float().to(device)
                if int(torch.sum(mask)) < 2: # 至少保留两个通道
                    mask[0] = 1
                    mask[1] = 1
                decoder_cfg.append(int(torch.sum(mask)))  # 剩余通道
                decoder_cfg_mask.append(mask.clone())  # mask列表

            m.indexes.data.mul_(mask)  # 掩膜操作
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                  format(k, mask.shape[0], int(torch.sum(mask))))
    # 传入新结构的参数
    cfgs = []
    encoder_prune = []
    decoder_prune = []
    for i in range(len(encoder_cfg)):
        if i % 2 != 0:
            encoder_prune.append([encoder_cfg[i - 1], encoder_cfg[i]])
    cnt = 0
    for i in range(len(decoder_cfg)):
        cnt += 1
        if cnt == 3:
            decoder_prune.append([decoder_cfg[i - 2], decoder_cfg[i - 1], decoder_cfg[i]])
            cnt = 0
    cfgs.append(encoder_prune)
    cfgs.append(decoder_prune)
    args.cfgs = cfgs
    # 定义新结构
    new_model, criterion, postprocessors = build_model_main(args)
    new_model.to(device)
    new_model_without_ddp = new_model

    # 参数移植
    new_model_dict = new_model.state_dict().copy()
    new_dict = transplant_weight_dab(model_without_ddp, new_model, encoder_cfg_mask, decoder_cfg_mask)
    new_model_dict.update(new_dict)
    new_model_without_ddp.load_state_dict(new_model_dict)

    n_parameters = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in new_model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in new_model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    args.start_epoch = 0

    print("Test new_model after load new weight ...")
    test_stats, coco_evaluator = evaluate(new_model_without_ddp, criterion, postprocessors,
                                          data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            new_model_without_ddp, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args,
            logger=(logger if args.save_log else None))
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}_beforedrop.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': new_model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': new_model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            new_model_without_ddp, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
            wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        epoch_time = time.time() - epoch_start_time
        epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
        log_stats['epoch_time'] = epoch_time_str

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    print("Now time: {}".format(str(datetime.datetime.now())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
