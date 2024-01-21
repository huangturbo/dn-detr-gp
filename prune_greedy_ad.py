# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import collections
import datetime
import json
import os
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
from engine import evaluate, train_one_epoch, accuracy
from models import build_DABDETR, build_dab_deformable_detr, \
    build_dab_deformable_detr_deformable_encoder_only, build_dab_dino_deformable_detr
from models.DN_DAB_DETR.transformer import channel_selection, eps
from util.utils import load_partial_weight_dab


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

    parser.add_argument('--batch_size', default=6, type=int)
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
    parser.add_argument('--coco_path', type=str, default='/mnt/public/datasets/coco') # TODO
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true',
                        help="Using for debug only. It will fix the size of input images to the maximum.")


    # Traing utils
    parser.add_argument('--output_dir', default='output_greedy/test', help='path where to save, empty for no saving') # TODO
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='resume/checkpoint_optimized_44.7ap.pth', help='resume from checkpoint') # TODO
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+',
                        help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, action='store_true', help="eval only. w/o Training.")
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
    # parser.add_argument('--cfgs_mask', type=list, default=None, help='cfg mask after prune')

    # adding neuron
    parser.add_argument('--num_evaluate', default=100, type=int,
                        help='num of neuron to evaluate for every evaluation. (Randomly pickup num_evaluate number of neuron if there are more potential neuron that can be add)')
    # skip for convergence criterion
    parser.add_argument('--map_tol', default=0.10, type=float,
                        help='tol to stop pruning a layer. Larger tol means more neurons to prune')
    parser.add_argument('--skip_eval_converge', default=0.02, type=float,
                        help='when bacth_top1 < (1 - skip_eval_convergence) * init_top, we skip eval the convergence using the full training dataset')
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

def decide_candidate_set(m, prunable_neuron, num_evaluate=50, isfirst=False):
    # only randomly pickup num_evaluate number of neurons to form the candidate set
    candidate_plus = []

    tem_a = m.prune_a.data.squeeze().cpu().numpy()  # 256

    if isfirst:
        eps_ = eps
    else:
        eps_ = 0.

    tem_a = np.where(tem_a <= eps_)[0]  # randomly pick up outside neuron to add
    np.random.shuffle(tem_a)
    tem_a = set(tem_a)
    prunable_neuron = set(np.where(prunable_neuron.astype(float) > 0)[0])
    tem_a = list(tem_a & prunable_neuron)

    candidate_plus = tem_a[:num_evaluate]
    return candidate_plus

def find_top_k_numbers(nums, k):
    sorted_data = sorted(nums, key=lambda x: list(x.values())[0])
    top_k = sorted_data[:k]
    return top_k


def decide_candidate_attn(net, eval_train_loader, m, candidate_plus, criterion, isfirst=False):
    # decide the candidate to perform update by 1/n stepsize

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    net.eval()
    criterion.eval()

    # opt_index = -1
    # opt_loss = float('inf')  # 正无穷大
    # opt_stepsize = 0.

    current_num_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))  # 0

    # judge the selected neurons one by one
    nums = []
    for candidate in candidate_plus:
        m.init_lsearch(candidate)
        m.prune_lsearch.data += 1. / (current_num_neuron + 1)

        loss_sum = 0
        with torch.no_grad():
            for samples, targets in eval_train_loader:
                samples = samples.to(args.device)
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    if need_tgt_for_training:
                        outputs, _ = net(samples, dn_args=(args.num_patterns, args.device))
                    else:
                        outputs = net(samples)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss_sum += sum(loss_dict_reduced_scaled.values())
                break

        loss = loss_sum

        nums.append({candidate: loss})

        # if loss < opt_loss:
        #     opt_index = candidate
        #     opt_loss = loss
        #     opt_stepsize = 1. / (current_num_neuron + 1)
    topk = find_top_k_numbers(nums, args.nheads) # TODO
    if isfirst:
        for i, item in enumerate(topk):
            opt_index = next(iter(item))
            if i == 0:
                m.prune_a *= 0.
                m.prune_a[:, :, opt_index] += 1.
                m.prune_w.data = 0. * m.prune_w.data
                m.prune_lsearch.data = 0. * m.prune_lsearch.data
                m.prune_gamma.data = 0. * m.prune_gamma.data
            else:
                opt_stepsize = 1. / (current_num_neuron + i + 1)
                m.update_alpha(opt_index, opt_stepsize)

    else:
        for i, item in enumerate(topk):
            opt_index = next(iter(item))
            opt_stepsize = 1. / (current_num_neuron + i + 1)
            m.update_alpha(opt_index, opt_stepsize)


def decide_candidate_ffn(net, eval_train_loader, m, candidate_plus, criterion, isfirst=False):
    # decide the candidate to perform update by 1/n stepsize
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    net.eval()
    criterion.eval()

    current_num_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))  # 0

    # judge the selected neurons one by one
    nums = []
    for candidate in candidate_plus:
        m.init_lsearch(candidate)
        m.prune_lsearch.data += 1. / (current_num_neuron + 1)

        with torch.no_grad():
            for samples, targets in eval_train_loader:
                samples = samples.to(args.device)
                targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast(enabled=args.amp):
                    if need_tgt_for_training:
                        outputs, _ = net(samples, dn_args=(args.num_patterns, args.device))
                    else:
                        outputs = net(samples)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                            for k, v in loss_dict_reduced.items() if k in weight_dict}
                loss = sum(loss_dict_reduced_scaled.values())
                break

        nums.append({candidate: loss})

        # if loss < opt_loss:
        #     opt_index = candidate
        #     opt_loss = loss
        #     opt_stepsize = 1. / (current_num_neuron + 1)
    topk = find_top_k_numbers(nums, args.nheads * 4) # TODO
    if isfirst:
        for i, item in enumerate(topk):
            opt_index = next(iter(item))
            if i == 0:
                m.prune_a *= 0.
                m.prune_a[:, :, opt_index] += 1.
                m.prune_w.data = 0. * m.prune_w.data
                m.prune_lsearch.data = 0. * m.prune_lsearch.data
                m.prune_gamma.data = 0. * m.prune_gamma.data
            else:
                opt_stepsize = 1. / (current_num_neuron + i + 1)
                m.update_alpha(opt_index, opt_stepsize)

    else:
        for i, item in enumerate(topk):
            opt_index = next(iter(item))
            opt_stepsize = 1. / (current_num_neuron + i + 1)
            m.update_alpha(opt_index, opt_stepsize)


def prune_a_layer(m, k, net, eval_train_loader, criterion, postprocessors, base_ds, layer_type, cnt_layer):
    isalladd = 0
    num_layer = k
    # test_stats, _ = evaluate(net, criterion, postprocessors,
    #                                       eval_train_loader, base_ds, args.device, args.output_dir, wo_class_error=False, args=args)
    init_map = 44.6 * ((1-args.skip_eval_converge) ** cnt_layer) # TODO
    print("The layer init_map:{}".format(init_map))

    m.switch_mode('prune')

    # prunable neuron list; only consider the neuron that is inside at initial
    prunable_neuron = (m.prune_a.cpu().data.squeeze().numpy() > 0)  # boolean列表
    all_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))  # 数量

    m.empty_all_eps()
    print('-' * 90)

    is_first_neuron = True
    # iteration = 0
    cur_neuron = 0
    verbose = True

    while True:

        args.num_evaluate = 100 if layer_type == "attn" else 200
        candidate_plus = decide_candidate_set(m, prunable_neuron, num_evaluate=args.num_evaluate,
                                              isfirst=is_first_neuron)
        if layer_type == "attn":
            decide_candidate_attn(net, eval_train_loader, m, candidate_plus, criterion, is_first_neuron)
        else:
            decide_candidate_ffn(net, eval_train_loader, m, candidate_plus, criterion, is_first_neuron)
        if is_first_neuron:
            is_first_neuron = False
        batch = 40 if layer_type == "attn" else 20
        test_batch_stats, _ = accuracy(net, criterion, postprocessors,
                                       eval_train_loader, base_ds, args.device, batch,
                                       wo_class_error=False, args=args, logger=None)
        batch_map = test_batch_stats['coco_eval_bbox'][0] * 100

        if batch_map >= (1. - args.skip_eval_converge) * init_map :
            if np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int)) == 0:
                continue
            elif layer_type == "attn" and np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int)) % args.nheads != 0:
                continue
            else:
                break
            # evaluate whether converged
            # test_stats, _ = evaluate(net, criterion, postprocessors,
            #                          eval_train_loader, base_ds, args.device, args.output_dir)
            # cur_loss = test_stats['loss']
            # cur_map = test_stats['coco_eval_bbox'][0] * 100
            # # cur_loss, cur_top1, cur_top5 = eval_train(net, eval_train_loader, criterion, args.device)
            # cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            # if verbose:
            #     print('Converge Eval------', args.map_tol)
            #     print(
            #         'Layer: ({:d}); Cur Loss: {:.4f}; Init Loss: {:.4f}; Cur map: ({:.4f}%); Init map: {:.4f}'.format(
            #             num_layer, cur_loss, init_loss, cur_map, init_map))
            #     all_neuron = 256 if layer_type == "attn" else 2048
            #     print('Cur_neuron/ All neuron', cur_neuron, all_neuron)
            #
            # if cur_map >= (1. - args.map_tol) * (init_map): break  # reach convergence

        else:
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            if verbose:
                print('Layer: ({:d}); Batch top1: {:.4f}'.format(num_layer, batch_map))
                print('Cur_neuron/ All neuron', cur_neuron, all_neuron)

        if cur_neuron >= all_neuron:
            print('all the neurons are added')
            m.set_alpha_to_init(prunable_neuron)
            isalladd = 1
            break

    print("This layer's Neuron", cur_neuron)
    test_stats, _ = evaluate(net, criterion, postprocessors,
                                          eval_train_loader, base_ds, args.device, args.output_dir, wo_class_error=False, args=args)
    cur_loss = test_stats['loss']
    cur_map = test_stats['coco_eval_bbox'][0] * 100
    print('Layer (before finetune): ({:d}); Cur Loss: {:.4f}; Cur top1: ({:.4f}%);'.format(
        num_layer, cur_loss, cur_map))
    print('=' * 90)

    a_para = m.prune_a.data
    a_num = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
    m.set_alpha_to_init(prunable_neuron)

    # return a_para, a_num, cur_map, isalladd
    return a_para, a_num, isalladd

def net_prune(net, eval_train_loader, criterion, postprocessors, base_ds):
    net.eval()

    cur_cfg = []
    cnt_layer = -1

    total_start = time.time()
    order = [14, 26, 38, 50, 62, 74, 108, 136, 164, 192, 220, 248, 109, 137, 165, 193, 221, 249,
             15, 27, 39, 51, 63, 75, 110, 138, 166, 194, 222, 250]
    attn = [14, 26, 38, 50, 62, 74, 108, 136, 164, 192, 220, 248, 109, 137, 165, 193, 221, 249]  # attn
    ffn = [15, 27, 39, 51, 63, 75, 110, 138, 166, 194, 222, 250] # ffn

    net_list = list(net.modules())
    for k in order:
        m = net_list[k]
        if isinstance(m, channel_selection):
            # ====================Skip the parts that have been pruned
            cur_neuron = np.sum((m.prune_a.cpu().data.numpy() > 0).astype(int))
            if k in attn:
                if cur_neuron != 256:
                    cnt_layer += 1
                    cur_cfg.append(cur_neuron)
                    continue
            else:
                if cur_neuron != 2048:
                    cnt_layer += 1
                    cur_cfg.append(cur_neuron)
                    continue
            # ====================Skip the parts that have been pruned
            cnt_layer += 1
            layer_type = "attn" if k in attn else "ffn"
            print("Start prune layer:{},layer type:{}".format(k, layer_type))
            a_para, a_num, isalladd = prune_a_layer(m, k, net, eval_train_loader, criterion,
                                                                     postprocessors, base_ds, layer_type, cnt_layer)  # 对该层剪枝
            m.prune_a.data = a_para
            cur_neuron = a_num

            print("This layer's Neuron", cur_neuron)

            cur_cfg.append(cur_neuron)

            # layer finetune
            m.switch_mode('train')
            # if not isalladd:
            #     train(train_loader, args.n_epoch)

            print('=' * 90)
            all_neuron = 256 if layer_type == "attn" else 2048
            # print("current cfg", cur_neuron)
            print('neuron pruned', all_neuron - cur_neuron)
            print('=' * 38, ' All Finish ', '=' * 38)
            torch.save({'model': net.state_dict()},
                       os.path.join(args.output_dir, 'dn_detr_prune.pth')) # TODO

    print('total time', time.time() - total_start)
    print('Finish Prune')
    m.switch_mode('train')
    torch.save({'model': net.state_dict()},
               os.path.join(args.output_dir, 'dn_detr_prune.pth')) # TODO


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
    wo_class_error = False
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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
        # sampler_val = torch.utils.data.RandomSampler(dataset_val)

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
        load_partial_weight_dab(model_without_ddp, checkpoint['model']) # TODO
        # model_without_ddp.load_state_dict(checkpoint['model'])
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

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                                  data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    net_prune(model_without_ddp, data_loader_val, criterion, postprocessors, base_ds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
