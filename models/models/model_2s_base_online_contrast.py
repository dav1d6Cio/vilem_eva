'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert_blip1 import BertConfig, BertModel

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import random
import pdb


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability'] if 'mlm_probability' in config.keys() else 0.15
        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if init_deit:
            checkpoint = torch.load('weights/deit_base_patch16_224-b5f2ef4d.pth')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("text_ids_queue", torch.zeros((30, self.queue_size), dtype=torch.long))
        self.register_buffer("text_mask_queue", torch.zeros((30, self.queue_size), dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, image, text, alpha=0, idx=None, contrast_online=True):
        if idx is None:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        # search related text in the queue
        if contrast_online:
            with torch.no_grad():
                temp_sim_i2t = image_feat @ self.text_queue.clone().detach()
                ret_index = temp_sim_i2t.max(dim=-1)[1]  # B
                ret_text_ids = self.text_ids_queue[:, ret_index].T
                ret_text_mask = self.text_mask_queue[:, ret_index].T
            
            text.input_ids = torch.cat([text.input_ids, ret_text_ids], dim=0)
            text.attention_mask = torch.cat([text.attention_mask, ret_text_mask], dim=0)
            
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                             mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        if idx is not None:
            idx = idx.view(-1, 1)
            if contrast_online:
                img_idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
                img_pos_idx = torch.eq(idx, img_idx_all).float()
                sim_t2i_targets = img_pos_idx / img_pos_idx.sum(1, keepdim=True)

                text_idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()[:, ret_index], self.idx_queue.clone().detach()], dim=1)
                text_pos_idx = torch.eq(idx, text_idx_all).float()
                sim_i2t_targets = text_pos_idx / text_pos_idx.sum(1, keepdim=True)
            else:
                idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
                pos_idx = torch.eq(idx, idx_all).float()
                sim_i2t_targets = pos_idx / pos_idx.sum(1, keepdim=True)
                sim_t2i_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            if contrast_online:
                batch_size = image_feat.shape[0]
                text_feat_m = text_feat_m[:batch_size]  # retrieval得到的text不计算loss_t2i
            sim_i2t_m = image_feat_m @ text_feat_all / self.temp  # B, B+B+queue_size
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp  # B, B+queue_size

            if idx is None:
                if contrast_online:
                    sim_t2i_targets = torch.zeros(sim_t2i_m.size()).to(image.device)
                    sim_t2i_targets.fill_diagonal_(1)

                    sim_i2t_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                    sim_i2t_targets.fill_diagonal_(0.5)
                    sim_i2t_targets[torch.arange(batch_size), batch_size+torch.arange(batch_size)] = 0.5
                else:
                    sim_t2i_targets = torch.zeros(sim_t2i_m.size()).to(image.device)
                    sim_t2i_targets.fill_diagonal_(1)
                    sim_i2t_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                    sim_i2t_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_i2t_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_t2i_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        if contrast_online:
            text_feat = text_feat[:batch_size]
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        
        if contrast_online:
            text.input_ids = text.input_ids[:batch_size]
            text.attention_mask = text.attention_mask[:batch_size]
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx, text.input_ids, text.attention_mask)


        return loss_ita

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    def encode_text(self, text):
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                        mode='text', output_hidden_states=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        return text_feat

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx, text_ids, text_mask):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        text_ids_gather = concat_all_gather(text_ids)
        text_mask_gather = concat_all_gather(text_mask)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.text_ids_queue[:, ptr:ptr+batch_size] = text_ids_gather.T
        self.text_mask_queue[:, ptr:ptr+batch_size] = text_mask_gather.T

        if idx is not None:
            idxs = concat_all_gather(idx)
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class GatherLayer(torch.autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = torch.distributed.get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor):
        output = [torch.empty_like(tensor)
                  for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, tensor)
        output = torch.cat(output, dim=0)

        ctx.rank = torch.distributed.get_rank()
        ctx.batch_size = tensor.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM)
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )