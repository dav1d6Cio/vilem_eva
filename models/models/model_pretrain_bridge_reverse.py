'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import interpolate_pos_embed
from models.bridge_reverse import VisionTransformer, Bridgeformer
from models.xbert import BertConfig, BertModel
import pdb
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


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
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config)
        text_width = self.text_encoder.config.hidden_size

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.bridger = Bridgeformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.num_patches = (config['image_res'] // 16)**2
        if init_deit:
            checkpoint = torch.load(config['vit_weights'])
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.bridge_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.bridger_m = Bridgeformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
                                    mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.bridge_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.bridger, self.bridger_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.bridge_proj, self.bridge_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("image_crop_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("bridge_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.image_crop_queue = nn.functional.normalize(self.image_crop_queue, dim=0)
        self.bridge_queue = nn.functional.normalize(self.bridge_queue, dim=0)

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        ##======================== global contrastive learning ==========================##
#         pdb.set_trace()
        question, answer = self.text_masking(text.input_ids, text.attention_mask)

        answer_embeds = self.text_encoder(text.input_ids, return_dict=True, mode='text').last_hidden_state
        answer_feat = F.normalize(self.text_proj(answer_embeds[:, 0, :]), dim=-1)

        img_embeds_list = self.visual_encoder(image, output_hidden_states=True)
        question_embeds_list = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                            return_dict=True, mode='text', output_hidden_states=True).hidden_states

        bridge_embeds = self.bridger(img_embeds_list[:-1], question_embeds_list[1:], text.attention_mask)
        bridge_feat = F.normalize(self.vision_proj(bridge_embeds[:, 0, :]), dim=-1)

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            answer_embeds_m = self.text_encoder_m(text.input_ids, return_dict=True, mode='text').last_hidden_state
            answer_feat_m = F.normalize(self.text_proj_m(answer_embeds_m[:, 0, :]), dim=-1)
            answer_feat_all = torch.cat([answer_feat_m.t(), self.image_crop_queue.clone().detach()], dim=1)

            img_embeds_list_m = self.visual_encoder_m(image, output_hidden_states=True)
            question_embeds_list_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='text', output_hidden_states=True).hidden_states

            bridge_embeds_m = self.bridger_m(img_embeds_list_m[:-1], question_embeds_list_m[1:], text.attention_mask)
            bridge_feat_m = F.normalize(self.vision_proj_m(bridge_embeds_m[:, 0, :]), dim=-1)
            bridge_feat_all = torch.cat([bridge_feat_m.t(), self.bridge_queue.clone().detach()], dim=1)

            sim_i2t_m = bridge_feat_m @ answer_feat_all / self.temp 
            sim_t2i_m = answer_feat_m @ bridge_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = bridge_feat @ answer_feat_all / self.temp
        sim_t2i = answer_feat @ bridge_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(answer_feat_m, bridge_feat_m)

        return loss_ita

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
    def _dequeue_and_enqueue(self, image_crop_feat, bridge_feat):
        # gather keys before updating queue
        image_crop_feats = concat_all_gather(image_crop_feat)
        bridge_feats = concat_all_gather(bridge_feat)

        batch_size = bridge_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_crop_queue[:, ptr:ptr + batch_size] = image_crop_feats.T
        self.bridge_queue[:, ptr:ptr + batch_size] = bridge_feats.T
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

    def text_masking(self, input_ids, text_mask):
        batch_size = input_ids.shape[0]
        question = input_ids.clone()
        answer = torch.full((batch_size, 8), self.tokenizer.mask_token_id, device=input_ids.device)
        for i in range(len(input_ids)):
            word_pos_index = torch.where(text_mask[i]==1)[0]
            select_index = torch.randint(1 ,len(word_pos_index), (2,), device=input_ids.device)
            answer_pos_index = word_pos_index[select_index]
            answer[i, 6:8] = input_ids[i, answer_pos_index]
            question[i, answer_pos_index] = self.tokenizer.mask_token_id
        answer[:, 0] = self.tokenizer.cls_token_id
        return question, answer

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