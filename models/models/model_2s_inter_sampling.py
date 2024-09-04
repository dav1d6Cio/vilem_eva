'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert12_wo_ca import BertConfig, BertModel

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

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.temp_neg = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
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
        self.register_buffer("text_neg_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.text_neg_queue = nn.functional.normalize(self.text_neg_queue, dim=0)

    def forward(self, image, text, alpha=0, idx=None, beta=0.):
        if idx is None:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
        batch_size = image.shape[0]
        image_embeds_list = self.visual_encoder(image, return_all_hidden_states=True)
        image_embeds = image_embeds_list[-1]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                        mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)  # B,C
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)  # B,C
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)
            
#             # negative captions
            rep_token_ids = text.rep_token_ids
            _, num_neg_text, text_len = rep_token_ids.shape
            rep_token_ids = rep_token_ids.view(-1, text_len)
            
            rep_text_attn_mask = text.attention_mask.unsqueeze(1).repeat(1, num_neg_text, 1).view(-1, text_len)
            text_neg_output_m = self.text_encoder_m(rep_token_ids, attention_mask=rep_text_attn_mask,
                                                    return_dict=True, mode='text')
            text_neg_feat_m = F.normalize(self.text_proj_m(text_neg_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_neg_feat_m = text_neg_feat_m.view(batch_size, num_neg_text, -1)  # B,num_neg,C
            text_neg_feat_m_all = torch.cat([text_feat_m.unsqueeze(1), text_neg_feat_m], dim=1)  # B,1+num_neg,C

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            sim_neg_mm = torch.matmul(image_feat_m.unsqueeze(1), text_neg_feat_m_all.transpose(1,2)).squeeze(1) / self.temp_neg # B,1+num_neg

            if idx is None:
                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)
            
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_neg_targets = torch.zeros((batch_size, 1+num_neg_text), device=image.device)
            sim_neg_targets[:, 0] = 1
            sim_neg_targets = alpha * F.softmax(sim_neg_mm, dim=1) + (1 - alpha) * sim_neg_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        
#         text_neg_feat_all_online = torch.cat([text_feat.unsqueeze(1), text_neg_feat_m], dim=1)  # B,1+num_neg,C
        # negative captions
#         rep_text_attn_mask = text.attention_mask.unsqueeze(1).repeat(1, num_neg_text, 1).view(-1, text_len)
        text_neg_output = self.text_encoder(rep_token_ids, attention_mask=rep_text_attn_mask,
                                                return_dict=True, mode='text')
        text_neg_feat = F.normalize(self.text_proj(text_neg_output.last_hidden_state[:, 0, :]), dim=-1)
        text_neg_feat = text_neg_feat.view(batch_size, num_neg_text, -1)  # B,num_neg,C
        text_neg_feat_all = torch.cat([text_feat.unsqueeze(1), text_neg_feat], dim=1)  # B,1+num_neg,C
        
        ## contrastive loss for rep words
        sim_neg = torch.matmul(image_feat.unsqueeze(1), text_neg_feat_m_all.transpose(1, 2)).squeeze(1) / self.temp_neg
        sim_neg_m = torch.matmul(image_feat_m.unsqueeze(1), text_neg_feat_all.transpose(1, 2)).squeeze(1) / self.temp_neg
        
        new_sim_neg, new_sim_neg_targets = self.pos_repeat(sim_neg, pos_number=16)
        new_sim_neg_m, _ = self.pos_repeat(sim_neg_m, pos_number=16)
        loss_inter = -torch.log((F.softmax(new_sim_neg, dim=-1) * new_sim_neg_targets).sum(1)).mean()
        loss_inter_m = -torch.log((F.softmax(new_sim_neg_m, dim=-1) * new_sim_neg_targets).sum(1)).mean()
#         loss_neg = -torch.sum(F.log_softmax(sim_neg, dim=1) * sim_neg_targets, dim=1)[text.loss_mask].sum() / text.loss_mask.sum()
#         loss_neg_m = -torch.sum(F.log_softmax(sim_neg_m, dim=1) * sim_neg_targets, dim=1)[text.loss_mask].sum() / text.loss_mask.sum()

        loss_inter = (loss_inter + loss_inter_m) / 2.

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx)

        return {'loss_ita': loss_ita, 'loss_neg': loss_inter}
    
    def pos_repeat(self, sim_neg, pos_number=16):
        pos_score = sim_neg[:, :1]
        neg_score = sim_neg[:, 1:]
        pos_score_expand = torch.cat([pos_score for _ in range(pos_number)], dim=1)
        new_sim_neg = torch.cat([pos_score_expand, neg_score], dim=1)
        new_targets = torch.zeros_like(new_sim_neg)
        new_targets[:, :pos_number] = 1.
        return new_sim_neg, new_targets
        
    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    def encode_text(self, text):
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
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
    def _dequeue_and_enqueue(self, image_feat, text_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
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