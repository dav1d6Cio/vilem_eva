'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert_blip1 import BertConfig, BertForMaskedLM

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

        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])

    def forward(self, image, text, alpha=0, idx=None):
        if idx is None:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
        image_embeds_list = self.visual_encoder(image, return_all_hidden_states=True)
        image_embeds = image_embeds_list[-1]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        image_feat_all = AllGather.apply(image_feat)
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                             mode='text', output_hidden_states=True)
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        text_feat_all = AllGather.apply(text_feat)

        sim_i2t = image_feat_all @ text_feat_all.t() / self.temp
        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = AllGather.apply(idx)
            pos_idx = torch.eq(idx_all, idx_all.t()).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        else:
            sim_targets = torch.eye(sim_i2t.shape[0], device=image.device)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_i2t, dim=0) * sim_targets, dim=0).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix=probability_matrix)
        #
        # with torch.no_grad():
        #     logits_m = self.text_encoder_m(input_ids,
        #                                    attention_mask=text.attention_mask,
        #                                    encoder_hidden_states=image_embeds_m,
        #                                    encoder_attention_mask=image_atts,
        #                                    return_dict=True,
        #                                    return_logits=True,
        #                                    )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds_list[2:-1],
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=labels,
                                       # soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha,
                                       output_hidden_states=True
                                       )
        loss_mlm = mlm_output.loss

        return {'loss_mlm': loss_mlm, 'loss_ita': loss_ita}

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
        # torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM)
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )