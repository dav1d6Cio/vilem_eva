'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit_twice_forward import VisionTransformer, interpolate_pos_embed
from object_models.vit import Mlp
from models.xbert import BertConfig, BertModel
from object_models.embeds import LearnedObjectTokens
from object_models.attentions import SlotAttention
from loss.loss_objects import align_obj_phrase

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
        self.mlm_probability = config['mlm_probability'] if 'mlm_probability' in config.keys() else 0
        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # object extract
        vision_width = config['vision_width']
        self.vision_width = vision_width
        self.obj_queries = LearnedObjectTokens(vision_width, 11)
        self.obj_fn = SlotAttention(dim=vision_width, iters=3, hidden_dim=int(np.round(2 * vision_width)))
        self.obj_proj = nn.Linear(vision_width, embed_dim)


        if init_deit:
            checkpoint = torch.load('weights/deit_base_patch16_224-b5f2ef4d.pth')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        bert_config = BertConfig.from_json_file(config['bert_config'])

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        # object
        self.obj_queries_m = LearnedObjectTokens(vision_width, 11)
        self.obj_fn_m = SlotAttention(dim=vision_width, iters=3, hidden_dim=int(np.round(2 * vision_width)))
        self.obj_proj_m = nn.Linear(vision_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.obj_queries, self.obj_queries_m],
                            [self.obj_fn, self.obj_fn_m],
                            [self.obj_proj, self.obj_proj_m]
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("obj_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("idx_queue", torch.full((1, self.queue_size), -100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.obj_queue = nn.functional.normalize(self.obj_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, image, text, phrase, alpha=0, idx=None):
        if idx is None:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)
        num_batch, num_phrase, len_phrase = phrase['input_ids'].shape
        phrase_ids = phrase['input_ids'].view(-1, len_phrase)
        phrase_att_mask = phrase['attention_mask'].view(-1, len_phrase)
        phrase_mask = phrase['mask']

        # extract image and object feat
        image_mid_embeds = self.visual_encoder.forward_pre(image, stop_layer=6)
        obj_embeds = self.extract_obj_feat(image_mid_embeds)  # B,S,C
        obj_mean_feat = F.normalize(self.obj_proj(obj_embeds.mean(1)), dim=-1)  # B,C
        image_embeds = self.visual_encoder.forward_post(torch.cat([image_mid_embeds, obj_embeds], dim=1), start_layer=6)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        phrase_embeds = self.text_encoder(phrase_ids, attention_mask=phrase_att_mask, return_dict=True,
                                          mode='text').last_hidden_state[:, 0].view(num_batch, num_phrase, -1)

        # obj-phrase alignment
        sim_op_targets = align_obj_phrase(F.normalize(obj_embeds, dim=-1), F.normalize(phrase_embeds, dim=-1), phrase_mask)
        obj_idx, phrase_idx = torch.where(sim_op_targets==1)
        obj_aln_embeds = obj_embeds.view(-1, self.vision_width)[obj_idx]
        phrase_aln_embeds = phrase_embeds.view(-1, self.vision_width)[phrase_idx]
        loss_opa = -self.cos(obj_aln_embeds, phrase_aln_embeds.detach()).mean()  # 也可以尝试用momentum encoder得到obj,phrase embeds，然后对称计算loss

        if idx is not None:
            idx = idx.view(-1, 1)
            idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(idx, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            # image-text ctr
            image_mid_embeds_m = self.visual_encoder_m.forward_pre(image, stop_layer=6)
            obj_embeds_m = self.extract_obj_feat_m(image_mid_embeds_m)
            obj_mean_feat_m = F.normalize(self.obj_proj_m(obj_embeds_m.mean(1)), dim=-1)  # B,C
            obj_mean_feat_all = torch.cat([obj_mean_feat_m.t(), self.obj_queue.clone().detach()], dim=1)

            image_embeds_m = self.visual_encoder_m.forward_post(torch.cat([image_mid_embeds_m, obj_embeds_m], dim=1), start_layer=6)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp
            sim_o2t_m = obj_mean_feat_m @ text_feat_all / self.temp
            sim_t2o_m = text_feat_m @ obj_mean_feat_all / self.temp

            if idx is None:
                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets
            sim_o2t_targets = alpha * F.softmax(sim_o2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2o_targets = alpha * F.softmax(sim_t2o_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp
        sim_o2t = obj_mean_feat @ text_feat_all / self.temp
        sim_t2o = text_feat @ obj_mean_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_o2t = -torch.sum(F.log_softmax(sim_o2t, dim=1) * sim_o2t_targets, dim=1).mean()
        loss_t2o = -torch.sum(F.log_softmax(sim_t2o, dim=1) * sim_t2o_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        loss_ota = (loss_o2t + loss_t2o) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, obj_mean_feat_m, idx)

        return {'loss_ita': loss_ita, 'loss_ota': loss_ota, 'loss_opa': loss_opa}

    def extract_obj_feat(self, image_mid_embeds):
        patch_embeds = image_mid_embeds[:, 1:]
        B, N, C = patch_embeds.shape
        q_objs = self.obj_queries().expand(B, -1, -1)

        # f_slots, p_slots: [B S C]
        f_slots = self.obj_fn(q_objs, patch_embeds)
        return f_slots

    def extract_obj_feat_m(self, image_mid_embeds):
        patch_embeds = image_mid_embeds[:, 1:]
        B, N, C = patch_embeds.shape
        q_objs = self.obj_queries_m().expand(B, -1, -1)

        # f_slots, p_slots: [B S C]
        f_slots = self.obj_fn_m(q_objs, patch_embeds)
        return f_slots

    def encode_image(self, image):
        image_mid_embeds = self.visual_encoder.forward_pre(image, stop_layer=6)
        obj_embeds = self.extract_obj_feat(image_mid_embeds)  # B,S,C
        obj_mean_feat = F.normalize(self.obj_proj(obj_embeds.mean(1)), dim=-1)  # B,C
        image_embeds = self.visual_encoder.forward_post(torch.cat([image_mid_embeds, obj_embeds], dim=1), start_layer=6)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        
        img_obj_feat = obj_mean_feat + image_feat
        
        return image_embeds, img_obj_feat

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
    def _dequeue_and_enqueue(self, image_feat, text_feat, obj_feat, idx):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        obj_feats = concat_all_gather(obj_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.obj_queue[:, ptr:ptr + batch_size] = obj_feats.T
        if idx is not None:
            idxs = concat_all_gather(idx)
            self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

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