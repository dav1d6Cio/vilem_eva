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
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer as MyBertTokenizer
from models.replace_model import Replace_Model
from eva_clip.utils import resize_vilem_pos_embed
from transformers import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

stopwords = json.load(open('data/sw_spacy_keep_preposition.json', 'r'))
def train(model, rep_model, data_loader, optimizer, tokenizer, epoch, warmup_steps,
          device, scheduler, config, args, std_tokenizer):
    # train
    model.train()

    metric_logger = utils.MetricLogger(args.output_dir, delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 1
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    loss_ita, loss_tec_l, loss_det_l, loss_tec_g, loss_det_g = [], [], [], [], []
    for i, (image, raw_text, text, rep_ids, idx) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device, non_blocking=True)
        idx = idx.to(device, non_blocking=True)
        wrong_token_ids = construct_wrong_captions(rep_model, std_tokenizer, raw_text, rep_ids, device, config['rep_model']['in_place'])
        text_input = tokenizer(text, max_length=30, padding='max_length', truncation=True, return_tensors="pt").to(
            device)
        text_input['rep_token_ids'] = wrong_token_ids
        text_input['rep_ids'] = recompute_replace_pos_idx(text_input['input_ids'], wrong_token_ids)
        if args.add_sw_mask:
            text_input['sw_mask'] = get_sw_mask(text_input.input_ids, tokenizer)
        
        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1, i / len(data_loader))

        loss_output = model(image, text_input, alpha=alpha, beta=args.beta, idx=idx)
        metric_logger.add_meter_from_list(list(loss_output.keys()),
                                          [utils.SmoothedValue(window_size=50, fmt='{value:.4f}') for i in
                                           range(len(loss_output.keys()))])

        loss = sum(loss_output.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update_from_dict(loss_output)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        
        if i < 1000:
            loss_ita.append(loss_output["loss_ita"].item())
            loss_tec_l.append(loss_output["loss_tec_l"].item())
            loss_det_l.append(loss_output['loss_det_l'].item())
            loss_tec_g.append(loss_output['loss_tec_g'].item())
            loss_det_g.append(loss_output['loss_det_g'].item())
        else:
            with open("coco_retrieval_base_log.json", "w") as f:
                json.dump({"loss_ita": loss_ita, "loss_tec_l": loss_tec_l, "loss_det_l": loss_det_l, "loss_tec_g": loss_tec_g, "loss_det_g": loss_det_g}, f)


            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

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

        if len(token) >= 2 and token[:2] == '##':
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

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset('re', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                          batch_size=[config['batch_size_train']] + [
                                                              config['batch_size_test']] * 2,
                                                          num_workers=[4, 4, 4],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    tokenizer = MyBertTokenizer.from_pretrained(args.text_encoder)
    std_tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model with {}".format(args.model))
    VILEM = utils.import_class(args.model)
    model = VILEM(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    rep_model = Replace_Model(tokenizer, config['rep_model'])
    rep_model = rep_model.to(device)

    if 'grad_checkpointing' in config.keys():
        model.visual_encoder.set_grad_checkpointing(config['grad_checkpointing'])

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = resize_vilem_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        if 'visual_encoder_m.pos_embed' in state_dict.keys():
            m_pos_embed_reshaped = resize_vilem_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        state_dict = model.resize_queue(state_dict)
        state_dict['queue_ptr'] = torch.zeros(1, dtype=torch.long)
        # remove rope param
        new_state_dict = {}
        for name, param in state_dict.items():
            if "rope." not in name:
                new_state_dict[name] = param
        state_dict = new_state_dict

        msg = model.load_state_dict(state_dict, strict=False)

        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # , find_unused_parameters=True
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0  

    print("Start training")
    start_time = time.time()
    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            train_stats = train(model, rep_model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, args, std_tokenizer)

        score_val_i2t, score_val_t2i = evaluation(model_without_ddp, val_loader, tokenizer,
                                                  device, config)
        score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader,
                                                    tokenizer, device, config)

        if utils.is_main_process():

            val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)
            print('val_result:', val_result)
            test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt)
            print('test_result:', test_result)

            if args.evaluate:
                log_stats = {
                    **{f'val_{k}': v for k, v in val_result.items()},
                    **{f'test_{k}': v for k, v in test_result.items()},
                    'epoch': epoch,
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if args.plot:
                    posi_nega_stat(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img,
                                   test_loader.dataset.img2txt)
            else:
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             **{f'val_{k}': round(v, 3) for k, v in val_result.items()},
                             **{f'test_{k}': round(v, 3) for k, v in test_result.items()},
                             'epoch': epoch,
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if val_result['r_mean'] > best:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = val_result['r_mean']
                    best_epoch = epoch

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write("best epoch: %d" % best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_flickr')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--model', type=str, default='models.model_retrieval.ALBEF')
    parser.add_argument('--text_encoder', default='weights/bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--uniform_replace', type=bool, default=False)
    parser.add_argument('--add_sw_mask', type=bool, default=False)
    parser.add_argument('--beta', type=float, default=1.)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    utils.save_model_file(args.model, args.output_dir)

    with open(os.path.join(args.output_dir, 'config.yaml'), 'w') as f:
        f.write(f"# command line: {' '.join(sys.argv)}\n\n")
        yaml.dump(config, f)

    main(args, config)
