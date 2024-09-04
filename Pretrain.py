'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import pdb
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer as MyBertTokenizer
from transformers import BertTokenizer
from models.replace_model import Replace_Model

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

def reinit_scheduler_properties_mysched(optimizer, scheduler, cfg) -> None:
    """
    with ApexDDP, do re-init to avoid lr_scheduler warning.
    issue: https://github.com/pytorch/pytorch/issues/27595
    issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/841
    """
    args = cfg
    if scheduler.optimizer == optimizer:
        scheduler.__init__(optimizer, t_initial=args.epochs, lr_min=args.min_lr, decay_rate=args.decay_rate,
                           warmup_lr_init=args.warmup_lr, warmup_t=args.warmup_epochs)

stopwords = json.load(open('data/sw_spacy_keep_preposition.json', 'r'))

def train(model, rep_model, data_loader, optimizer, tokenizer, epoch,
          warmup_steps, device, scheduler, config, args, std_tokenizer, accelerator=None):
    # train
    model.train()

    metric_logger = utils.MetricLogger(args.output_dir, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    inplace_rep = config['rep_model']['inplace'] if 'inplace' in config['rep_model'].keys() else False

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)
    
    optimizer.zero_grad()
    for i, (image, raw_text, text, rep_ids, image_paths) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        image = image.to(device, non_blocking=True)
        wrong_token_ids = construct_wrong_captions(rep_model, std_tokenizer, raw_text, rep_ids, device, inplace=inplace_rep)
        text_input = tokenizer(text, max_length=30, padding='max_length', truncation=True, return_tensors="pt").to(
            device)
        text_input['rep_token_ids'] = wrong_token_ids
        text_input['rep_ids'] = recompute_replace_pos_idx(text_input['input_ids'], wrong_token_ids)
        if args.add_sw_mask:
            text_input['sw_mask'] = get_sw_mask(text_input.input_ids, tokenizer)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_output = model(image, text_input, alpha=alpha, beta=args.beta)
        metric_logger.add_meter_from_list(list(loss_output.keys()), [utils.SmoothedValue(window_size=50, fmt='{value:.4f}') for i in range(len(loss_output.keys()))])

        loss = sum(loss_output.values())
        loss = loss / args.accum_iter
        
        if accelerator is not None:
            accelerator.backward_step(loss, optimizer)
            accelerator_clip_grad_norm = float(config['accelerator']['CLIP_GRAD_NORM'])
            if accelerator_clip_grad_norm > 0:
                accelerator.optimizer_step(optimizer, model, accelerator_clip_grad_norm)
        else:
            loss.backward()
        
        if ((i + 1) % args.accum_iter == 0) or (i + 1 == len(data_loader)):
            optimizer.step()
            optimizer.zero_grad()
        
        metric_logger.update_from_dict(loss_output)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        args.global_step += 1

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
    return total_norm

@torch.no_grad()
def recompute_replace_pos_idx(right_text_ids, wrong_text_ids):
    return right_text_ids != wrong_text_ids

@torch.no_grad()
def construct_wrong_captions(generator, std_tokenizer, text, rep_pos_ids, device, inplace=False):
    text_input = std_tokenizer(text, padding=True, return_tensors="pt").to(device)
    # generate replace captions
    num_replace = rep_pos_ids.shape[1]
    rep_token_ids = text_input.input_ids.clone()
    with torch.no_grad():
        for rep_index in range(num_replace):
            rep_ids_i = rep_pos_ids[:, rep_index]
            if not inplace:
                rep_token_ids = generator(text_input.input_ids, text_input.attention_mask, rep_ids_i, rep_token_ids) # .input_ids, text_input.attention_mask
            # 如果要考虑i-1次的replace，则使用下面的语句
            else:
                rep_token_ids = generator(rep_token_ids, text_input.attention_mask, rep_ids_i)
            rep_token_ids = rep_token_ids.detach()
        # rep_token_ids的CLS可能会被换掉，因此这里要换上CLS
        rep_token_ids[:, 0] = std_tokenizer.cls_token_id
        # 去掉SEP
        rep_token_ids[rep_token_ids==std_tokenizer.sep_token_id] = std_tokenizer.pad_token_id
        # 去掉逗号和句号
        rep_token_ids = remove_comma_period(rep_token_ids, std_tokenizer)
    # wrong_text = batch_convert_ids_to_strings(rep_token_ids, std_tokenizer)
    # return wrong_text
    return rep_token_ids

def remove_comma_period(text_ids, tokenizer):
    result = torch.zeros((text_ids.shape[0], 30), device=text_ids.device).long()
    for i, ids in enumerate(text_ids):
        if tokenizer.convert_tokens_to_ids(',') in ids:
            ids = ids[ids!=tokenizer.convert_tokens_to_ids(',')]
        if tokenizer.convert_tokens_to_ids('.') in ids:
            ids = ids[ids != tokenizer.convert_tokens_to_ids('.')]
        ids = ids[:30]
        result[i, :len(ids)] = ids
    return result

def get_sw_mask(input_ids, tokenizer):
    sw_mask = torch.ones_like(input_ids)
    for i, ids in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for j, token in enumerate(tokens):
            if token in stopwords:
                sw_mask[i, j] = 0
    return sw_mask

@torch.no_grad()
def batch_convert_ids_to_strings(text_ids, tokenizer):
    strings = []
    for ids in text_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        strings.append(tokens_to_sentence(tokens))
    return strings

@torch.no_grad()
def tokens_to_sentence(tokens):
    clean_tokens = []
    for token in tokens:
        # 去掉逗号和句号
        if token in ['[CLS]', '[SEP]', '[PAD]', ',', '.']:
            continue

        if len(token) >= 2 and token[:2] == '##' and len(clean_tokens) > 0:
            clean_tokens[-1] += token[2:]
        else:
            clean_tokens.append(token)

    return ' '.join(clean_tokens)

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    if 'grad_checkpointing' in config.keys():
        model.visual_encoder.set_grad_checkpointing(False)
    metric_logger = utils.MetricLogger(args.output_dir, delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    text_embeds = []
    text_embeds_m = []

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(
            device)
        text_embed, text_embed_m = model.encode_text(text_input)
        text_embeds.append(text_embed)
        text_embeds_m.append(text_embed_m)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_embeds_m = torch.cat(text_embeds_m, dim=0)

    image_embeds = []
    image_embeds_m = []
    
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed, image_embed_m = model.encode_image(image)
        image_embeds.append(image_embed)
        image_embeds_m.append(image_embed_m)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_embeds_m = torch.cat(image_embeds_m, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    score_matrix_i2t = image_embeds @ text_embeds_m.t()
    score_matrix_t2i = text_embeds @ image_embeds_m.t()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    if 'grad_checkpointing' in config.keys():
        model.visual_encoder.set_grad_checkpointing(config['grad_checkpointing'])

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': round(tr1, 3),
                   'txt_r5': round(tr5, 3),
                   'txt_r10': round(tr10, 3),
                   'txt_r_mean': round(tr_mean, 3),
                   'img_r1': round(ir1, 3),
                   'img_r5': round(ir5, 3),
                   'img_r10': round(ir10, 3),
                   'img_r_mean': round(ir_mean, 3),
                   'r_mean': round(r_mean, 3)}
    return eval_result

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    args.global_step = 0
    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating dataset")
    datasets, test_dataset = create_dataset('pretrain', config, debug=args.debug)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        samplers = create_sampler([datasets], [True], num_tasks, global_rank) + [None, ]
    else:
        samplers = [None, None]
    
    data_loader, test_dataloader = create_loader([datasets, test_dataset], samplers,
                                                 batch_size=[config['batch_size'], config['batch_size']],
                                                 num_workers=[4, 4], is_trains=[True, False], collate_fns=[None, None])

    tokenizer = MyBertTokenizer.from_pretrained(args.text_encoder)
    std_tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    VILEM = utils.import_class(args.model)
    model = VILEM(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)
    model = model.to(device)

    rep_model = Replace_Model(tokenizer, config['rep_model'])
    rep_model = rep_model.to(device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
    
    if 'accelerator' in config.keys():
        arg_acc = utils.AttrDict(config['accelerator'])
        accelerator = ApexDDPAccelerator(arg_acc, logger=None)

        model, optimizer, lr_scheduler = accelerator.set_up(model, optimizer, lr_scheduler, local_rank, num_tasks, global_rank)
        reinit_scheduler_properties_mysched(optimizer, lr_scheduler, arg_sche)
    else:
        accelerator = None
    
    if 'grad_checkpointing' in config.keys():
        model.visual_encoder.set_grad_checkpointing(config['grad_checkpointing'])
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if accelerator:
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict['module.'+k] = v
            state_dict = new_state_dict
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
    else:
        temp_checkpoint = os.path.join(args.output_dir, 'checkpoint_temp.pth')
        if os.path.exists(temp_checkpoint):
            checkpoint = torch.load(temp_checkpoint, map_location='cpu')
            state_dict = checkpoint['model']
            if accelerator:
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_state_dict['module.'+k] = v
                state_dict = new_state_dict
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(state_dict)
            print('load checkpoint from %s' % args.checkpoint)


    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #, find_unused_parameters=True
        model_without_ddp = model.module
        if accelerator:
            model_without_ddp = model_without_ddp.module

    
    ##===========================test evaluation=============================##
    if args.inference:
        print("Start inference")
        start_time = time.time()
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_dataloader, tokenizer, device, config)
        test_result = itm_eval(score_test_i2t, score_test_t2i, test_dataloader.dataset.txt2img,
                               test_dataloader.dataset.img2txt)
        print('test_result:', test_result)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Inference time {}'.format(total_time_str))

    ##=======================================================================##
    else:
        print("Start training")
        start_time = time.time()
        for epoch in range(start_epoch, max_epoch):

            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)

            train_stats = train(model, rep_model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device,
                                lr_scheduler, config, args, std_tokenizer, accelerator=accelerator)
            score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_dataloader, tokenizer, device, config)
            if utils.is_main_process():
                test_result = itm_eval(score_test_i2t, score_test_t2i, test_dataloader.dataset.txt2img,
                                       test_dataloader.dataset.img2txt)
                print('test_result:', test_result)
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'test_{k}': round(v, 3) for k, v in test_result.items()},
                             'epoch': epoch,
                             }
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }

                # save template checkpoint
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_temp.pth'))
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))

                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

            dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        # delete temp checkpoint
        if utils.is_main_process():
            os.remove(os.path.join(args.output_dir, 'checkpoint_temp.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Pretrain.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='Pretrain/')
    parser.add_argument('--text_encoder', default='weights/bert-base-uncased')
    parser.add_argument('--model', type=str, default='models.model_pretrain.ALBEF')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--uniform_replace', type=bool, default=False)
    parser.add_argument('--add_sw_mask', type=bool, default=False)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--tecd_step', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--inference', type=bool, default=False)
    parser.add_argument('--bert_size', type=str, default=None)
    parser.add_argument('--accum_iter', type=int, default=1)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    # yaml = YAML(typ='rt')
    # config = yaml.load(open(args.config, 'r'))


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.save_model_file(args.model, args.output_dir)
    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n\n")
        yaml.dump(config, f)

    main(args, config)