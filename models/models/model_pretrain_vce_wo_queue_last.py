'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import interpolate_pos_embed, interpolate_patch_embed
from models.vit_concept import VisionTransformer, Conceptformer
from models.xbert import BertConfig, BertModel
from models.mask_generator import MaskingGenerator
import pdb
from loss.loss_objects import matching_contrastive_loss_squeeze
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
        self.concepter = Conceptformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12,
                                       num_heads=12,
                                       mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                       num_concept=16)
        self.num_patches = (config['image_res'] // 16) ** 2
        if init_deit:
            checkpoint = torch.load(config['vit_weights'])
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped

            patch_embed_reshaped = interpolate_patch_embed(state_dict['patch_embed.proj.weight'], self.visual_encoder)
            state_dict['patch_embed.proj.weight'] = patch_embed_reshaped

            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        vision_width = config['vision_width']

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.concept_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.temp_cpt = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models
        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.concepter_m = Conceptformer(img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12,
                                         num_heads=12,
                                         mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                         num_concept=16)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.concept_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.concepter, self.concepter_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            [self.concept_proj, self.concept_proj_m],
                            ]

        self.copy_params()

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # B,C

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                        mode='text')
        text_embeds = text_output.last_hidden_state
        text_projs = F.normalize(self.text_proj(text_embeds), dim=-1)  # B,C
        text_feat = text_projs[:, 0].contiguous()
        word_feat = text_projs[:, 1:].contiguous()

        concept_embeds = self.concepter(image_embeds)
        concept_feat = F.normalize(self.concept_proj(concept_embeds), dim=-1)  # B,num_concept,C

        image_feat_gather = all_gather_with_grad(image_feat)
        text_feat_gather = all_gather_with_grad(text_feat)
        sim_matrix = image_feat_gather @ text_feat_gather.t() / self.temp

        sim_targets = torch.zeros(sim_matrix.size()).to(image.device)
        sim_targets.fill_diagonal_(1)
        loss_i2t = -torch.sum(F.log_softmax(sim_matrix, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_matrix, dim=0) * sim_targets, dim=0).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        # visual concept loss
        # gather features
        concept_feat_gather = all_gather_with_grad(concept_feat)
        word_feat_gather = all_gather_with_grad(word_feat)
        text_mask = (text.attention_mask[:, 1:] * text.content_mask[:, 1:]).contiguous()
        text_mask_gather = all_gather_with_grad(text_mask)
#         concept_feat_m_gather = concat_all_gather(concept_feat_m)
#         word_feat_m_gather = concat_all_gather(word_feat_m)
#         pdb.set_trace()
        loss_vca = matching_contrastive_loss_squeeze(concept_feat_gather, word_feat_gather, text_mask_gather, self.temp_cpt, alpha=alpha)
        return loss_ita, loss_vca

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