'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.bert_blip1 import BertConfig, BertTECModel

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
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), last_norm=False)

        if init_deit:
            checkpoint = torch.load('weights/deit_base_patch16_224-b5f2ef4d.pth')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        bert_config = BertConfig.from_json_file(config['bert_config'])
        if 'head_layer' in config.keys():
            bert_config.fusion_layer = config['head_layer']

        self.text_encoder = BertTECModel.from_pretrained(text_encoder, config=bert_config, 
                                                         add_ted_head=config['add_ted_head'],
                                                         add_mlm_head=config['add_mlm_head'])
        
    def forward(self, image, text, alpha=0, idx=None):
        image_embeds_list = self.visual_encoder(image, return_all_hidden_states=True)
        image_embeds = image_embeds_list[-1]
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        ##================= MLM ========================##
#         pdb.set_trace()
        labels = text.input_ids.clone()
        det_labels, correction_labels = self.tec_mask(text.rep_token_ids, text.rep_ids, labels)
        # with torch.no_grad():
        #     corr_logits_m, det_logits_m = self.text_encoder_m(text.rep_token_ids,
        #                                    attention_mask=text.attention_mask,
        #                                    encoder_hidden_states=image_embeds_m,
        #                                    encoder_attention_mask=image_atts,
        #                                    return_dict=True,
        #                                    return_logits=True,
        #                                    )

        mlm_output, loss_det = self.text_encoder(text.rep_token_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds_list[2:-1],
                                       encoder_attention_mask=image_atts,
                                       return_dict=True,
                                       labels=correction_labels,
                                       detection_labels=det_labels,
                                       alpha=alpha
                                       )
        loss_mlm = mlm_output.loss

        return {'loss_tec': loss_mlm, 'loss_det': loss_det}

    def encode_image(self, image):
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        return image_feat

    def encode_text(self, text):
        text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
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

    def tec_mask(self, input_ids, rep_ids, targets):
        det_labels = torch.ones_like(input_ids)  # N,L
        det_labels[input_ids==self.tokenizer.pad_token_id] = -100
        det_labels[rep_ids==1] = 0
        det_labels[:, 0] = -100 # 永远将CLS置为1

        correction_labels = torch.full_like(targets, -100)
        correction_labels[rep_ids==1] = targets[rep_ids==1]

        return det_labels, correction_labels

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