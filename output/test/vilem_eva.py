'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert_add_cls_w_ca import BertConfig, BertTECModel
from eva_clip import create_model

import torch
import torch.nn.functional as F
from torch import nn

import copy
import numpy as np
import random
import pdb


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True,
                 lambda_ted=1.,
                 ):
        super().__init__()
        self.init_params = [] # parameter trained from scratch, use larger lr
        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability'] if 'mlm_probability' in config.keys() else 0.15
        embed_dim = config['embed_dim']
        if 'lambda_ted' in config.keys():
            self.lambda_ted = config['lambda_ted']
        else:
            self.lambda_ted = lambda_ted
        self.visual_encoder = create_model(config['vit_name'], config['vit_weights'], force_custom_clip=True, force_image_size=config['image_res']).visual

        vision_width = config['vision_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.encoder_width = self.visual_encoder.embed_dim
        if 'head_layer' in config.keys():
            bert_config.fusion_layer = config['head_layer']
            
        if ('accelerator' in config.keys()) and (config['accelerator']['FP16_OPT_LEVEL'] != 'O0'):
            bert_config.fp16 = True  # will use some operations to avoid gradient overflow
        self.text_encoder, msg = BertTECModel.from_pretrained(text_encoder, config=bert_config,
                                                         add_ted_head=config[
                                                             'add_ted_head'] if 'add_ted_head' in config.keys() else False,
                                                         add_mlm_head=config[
                                                             'add_mlm_head'] if 'add_mlm_head' in config.keys() else False,
                                                         output_loading_info=True)
        self.init_params.extend(['text_encoder.' + n for n in msg['missing_keys']])  # of cross attention

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = self.visual_encoder.head
        self.vision_cls_proj = copy.deepcopy(self.vision_proj)
        self.visual_encoder.head = None
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        
        self.init_params.extend(['vision_proj.' + n for n, _ in self.vision_proj.named_parameters()])
        self.init_params.extend(['text_proj.' + n for n, _ in self.text_proj.named_parameters()])
        self.init_params.extend(['temp'])
        

        # create momentum models
        self.visual_encoder_m = create_model(config['vit_name'], config['vit_weights'], force_custom_clip=True, force_image_size=config['image_res']).visual
        self.vision_proj_m = self.visual_encoder_m.head
        self.vision_cls_proj_m = copy.deepcopy(self.vision_proj_m)
        self.visual_encoder_m.head = None
        self.text_encoder_m = BertTECModel.from_pretrained(text_encoder, config=bert_config,
                                                           add_ted_head=config['add_ted_head'],
                                                           add_mlm_head=config['add_mlm_head'])
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.vision_cls_proj, self.vision_cls_proj_m]
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

    def forward(self, image, text, alpha=0, idx=None, beta=1.):
        if idx is None:
            with torch.no_grad():
                self.temp.clamp_(0.001, 0.5)

        image_embeds, image_embeds_list = self.visual_encoder.forward_features(image, return_all_hidden_states=True)
        image_atts = torch.ones(image_embeds_list[0].size()[:-1], dtype=torch.long).to(image.device)
        image_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)

        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True, mode='text')

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
            image_embeds_m, image_embeds_m_list = self.visual_encoder_m.forward_features(image, return_all_hidden_states=True)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)
            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
                                                     return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            if idx is None:
                sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
                sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2.
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idx, text.input_ids, text.attention_mask)
        
        ##================= ViLEM ========================##
        labels = text.input_ids.clone()
        det_labels, correction_labels = self.tec_mask(text.rep_token_ids, text.rep_ids, labels)

        if 'sw_mask' in text.keys():
            det_labels[text.sw_mask == 0] = -100
        
        # ViLEM with local visual features
        mlm_output1, loss_det1 = self.text_encoder(text.rep_token_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds_m_list[::2],
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   labels=correction_labels,
                                                   detection_labels=det_labels,
                                                   alpha=alpha
                                                   )
        loss_mlm1 = mlm_output1.loss
        # visual feat + momentum text encoder
        mlm_output2, loss_det2 = self.text_encoder_m(text.rep_token_ids,
                                                     attention_mask=text.attention_mask,
                                                     encoder_hidden_states=image_embeds_list[::2],
                                                     encoder_attention_mask=image_atts,
                                                     return_dict=True,
                                                     labels=correction_labels,
                                                     detection_labels=det_labels,
                                                     alpha=alpha
                                                     )

        loss_mlm2 = mlm_output2.loss
        
        # ViLEM with global visual features
        mlm_output3, loss_det3 = self.text_encoder(text.rep_token_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=self.vision_cls_proj(image_embeds_m_list[-1]),
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   labels=correction_labels,
                                                   detection_labels=det_labels,
                                                   alpha=alpha,
                                                   mode='multi_modal_wo_ca'
                                                   )
        loss_mlm3 = mlm_output3.loss
        # visual feat + momentum text encoder
        mlm_output4, loss_det4 = self.text_encoder_m(text.rep_token_ids,
                                                     attention_mask=text.attention_mask,
                                                     encoder_hidden_states=self.vision_cls_proj(image_embeds_list[-1]),
                                                     encoder_attention_mask=image_atts,
                                                     return_dict=True,
                                                     labels=correction_labels,
                                                     detection_labels=det_labels,
                                                     alpha=alpha,
                                                     mode='multi_modal_wo_ca'
                                                     )

        loss_mlm4 = mlm_output4.loss

        loss_mlm_l = (loss_mlm1 + loss_mlm2) / 2. * (1 - beta)
        loss_mlm_g = (loss_mlm3 + loss_mlm4) / 2. * beta
        loss_det_l = (loss_det1 + loss_det2) / 2. * (1 - beta)
        loss_det_g = (loss_det3 + loss_det4) / 2. * beta

        return {'loss_ita': loss_ita, 'loss_tec_l': loss_mlm_l, 'loss_det_l': loss_det_l, 'loss_tec_g': loss_mlm_g, 'loss_det_g': loss_det_g}
    
    def cat_frame_embeds_list(self, frame_embeds_list, batch_size, num_frame):
        new_list = []
        for i, embeds in enumerate(frame_embeds_list):
            if i == 0:
                # 0 is position embedding
                new_list.append(embeds)
                continue
            BN, L, C = embeds.shape
            embeds = embeds.view(batch_size, num_frame, L, C)
            embeds_cls = embeds[:, :, 0:1].mean(1)  # B,num_frame, 1, C ==> B,1,C
            embeds_others = embeds[:, :, 1:].contiguous().view(batch_size, -1, C) # B,all_patch,C
            new_embeds = torch.cat([embeds_cls, embeds_others], dim=1) # B,1+all_patch,C
            new_list.append(new_embeds)
        return new_list
    
    def encode_image(self, image):
        image_embeds = self.visual_encoder.forward_features(image)
        image_feat = F.normalize(self.vision_proj(image_embeds), dim=-1)
        
        image_embeds_m = self.visual_encoder_m.forward_features(image)
        image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m), dim=-1)

        return image_feat, image_feat_m

    def encode_text(self, text):
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
        
        text_embeds_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,
                                         return_dict=True, mode='text').last_hidden_state
        text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
        
        return text_feat, text_feat_m

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
        self.text_ids_queue[:, ptr:ptr + batch_size] = text_ids_gather.T
        self.text_mask_queue[:, ptr:ptr + batch_size] = text_mask_gather.T
        
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

    def tec_mask(self, input_ids, rep_ids, targets):
        det_labels = torch.ones_like(input_ids)  # N,L
        det_labels[input_ids == self.tokenizer.pad_token_id] = -100
        det_labels[rep_ids == 1] = 0
        det_labels[:, 0] = -100  # 永远将CLS置为1

        correction_labels = torch.full_like(targets, -100)
        correction_labels[rep_ids == 1] = targets[rep_ids == 1]

        return det_labels, correction_labels
    
    def resize_queue(self, state_dict):
        cur_queue_size = self.image_queue.shape[1]
        if state_dict['image_queue'].shape[1] != cur_queue_size:
            state_dict['image_queue'] = state_dict['image_queue'][:, :cur_queue_size]
            state_dict['text_queue'] = state_dict['text_queue'][:, :cur_queue_size]
            state_dict['idx_queue'] = state_dict['idx_queue'][:, :cur_queue_size]
            state_dict['text_ids_queue'] = state_dict['text_ids_queue'][:, :cur_queue_size]
            state_dict['text_mask_queue'] = state_dict['text_mask_queue'][:, :cur_queue_size]
        return state_dict


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

class VL_Transformer_ITC(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

#         self.visual_encoder = VisionTransformer(
#             img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
#             mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 
        self.visual_encoder_m = VisionTransformer(
            img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        self.text_encoder = BertTECModel.from_pretrained(text_encoder, config=bert_config)
#         self.text_encoder_m = BertTECModel.from_pretrained(text_encoder, config=bert_config)
        
#         self.vision_proj = nn.Linear(768, 256)
#         self.text_proj = nn.Linear(768, 256)

    def forward(self, image, text):
        image_embeds_m_list = self.visual_encoder_m(image, return_all_hidden_states=True)
        image_embeds_m = image_embeds_m_list[-1]
        image_atts = torch.ones(image_embeds_m.size()[:-1], dtype=torch.long).to(image.device)
        corr_logits, det_logits = self.text_encoder(text.input_ids,
                                                   attention_mask=text.attention_mask,
                                                   encoder_hidden_states=image_embeds_m_list[2:-1],
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   return_logits=True
                                                   )
        return corr_logits