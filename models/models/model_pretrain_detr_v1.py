'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit import interpolate_pos_embed, interpolate_patch_embed
from models.vit_gen import VisionTransformer
from models.xbert import BertConfig, BertModel
import pdb
from loss.loss_objects import matching_contrastive_loss
import torch
import torch.nn.functional as F
from torch import nn


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True,
                 num_obj_query=8,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        self.num_obj_query = num_obj_query
        embed_dim = config['embed_dim']
        self.embed_dim = embed_dim

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config)
        text_width = self.text_encoder.config.hidden_size

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

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

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.temp_cpt = nn.Parameter(torch.ones([]) * config['temp'])

        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

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
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("object_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("phrase_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("obj_queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        self.object_queue = nn.functional.normalize(self.object_queue, dim=0)
        self.phrase_queue = nn.functional.normalize(self.phrase_queue, dim=0)

    def forward(self, image, text, phrase, alpha=0):
        # phrase: Bxnum_phrase, L
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)
        # pdb.set_trace()
        num_batch, num_phrase, len_phrase = phrase['input_ids'].shape
        phrase_ids = phrase['input_ids'].view(-1, len_phrase)
        phrase_att_mask = phrase['attention_mask'].view(-1, len_phrase)
        
        image_embeds = self.visual_encoder(image)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # B,C
        obj_feat = F.normalize(self.vision_proj(image_embeds[:, -self.num_obj_query:, :]), dim=-1) # B,num_obj,C

        text_embeds = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text').last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)  # B,C

        phrase_embeds = self.text_encoder(phrase_ids, attention_mask=phrase_att_mask, return_dict=True, mode='text').last_hidden_state
        phrase_feat = F.normalize(self.text_proj(phrase_embeds[:, 0, :]), dim=-1).view(num_batch, num_phrase, -1)  # B,C

        # obj-phrase alignment
        obj_phrase_sim = obj_feat @ phrase_feat.transpose(-1,-2)  # B,num_obj,num_phrase
        obj_phrase_align = obj_phrase_sim.max(-2)[1]  # B,num_phrase.  the index of object, each phrase correspond

        phrase_mask = phrase['mask'].view(-1)
        aligned_obj_feat = obj_feat[torch.arange(num_batch).unsqueeze(-1), obj_phrase_align].view(-1, self.embed_dim)  # Bxnum_phrase,C
        aligned_obj_feat = aligned_obj_feat[phrase_mask==1]
        aligned_phrase_feat = phrase_feat.view(-1, self.embed_dim)[phrase_mask==1]  # num_valid_phrase,C
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)

            text_embeds_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
                                                mode='text').last_hidden_state
            text_feat_m = F.normalize(self.text_proj_m(text_embeds_m[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

            # 对concept和word进行配对
            obj_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, -self.num_obj_query:, :]), dim=-1)
            aligned_obj_feat_m = obj_feat_m[torch.arange(num_batch).unsqueeze(-1), obj_phrase_align].view(-1, self.embed_dim)  # Bxnum_phrase,C
            aligned_obj_feat_select_m = aligned_obj_feat_m[phrase_mask==1]
            aligned_obj_feat_all = torch.cat([aligned_obj_feat_select_m.t(), self.object_queue.clone().detach()], dim=1)

            phrase_embeds_m = self.text_encoder_m(phrase_ids, attention_mask=phrase_att_mask,
                                              return_dict=True, mode='text').last_hidden_state
            aligned_phrase_feat_m = F.normalize(self.text_proj_m(phrase_embeds_m[:, 0, :]), dim=-1) # Bxnum_phrase,C
            aligned_phrase_feat_select_m = aligned_phrase_feat_m[phrase_mask==1]
            aligned_phrase_feat_all = torch.cat([aligned_phrase_feat_select_m.t(), self.phrase_queue.clone().detach()], dim=1)

            sim_o2p_m = aligned_obj_feat_select_m @ aligned_phrase_feat_all / self.temp
            sim_p2o_m = aligned_phrase_feat_select_m @ aligned_obj_feat_all / self.temp

            sim_op_targets = torch.zeros(sim_o2p_m.size()).to(image.device)
            sim_op_targets.fill_diagonal_(1)

            sim_o2p_targets = alpha * F.softmax(sim_o2p_m, dim=1) + (1 - alpha) * sim_op_targets
            sim_p2o_targets = alpha * F.softmax(sim_p2o_m, dim=1) + (1 - alpha) * sim_op_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        sim_o2p = aligned_obj_feat @ aligned_phrase_feat_all / self.temp
        sim_p2o = aligned_phrase_feat @ aligned_obj_feat_all / self.temp
        pdb.set_trace()

        loss_o2p = -torch.sum(F.log_softmax(sim_o2p, dim=1) * sim_o2p_targets, dim=1).mean()
        loss_p2o = -torch.sum(F.log_softmax(sim_p2o, dim=1) * sim_p2o_targets, dim=1).mean()

        loss_opa = (loss_o2p + loss_p2o) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m, aligned_obj_feat_m, aligned_phrase_feat_m, phrase_mask)

        return loss_ita, torch.tensor(0., device=loss_ita.device)#loss_opa

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
    def _dequeue_and_enqueue(self, image_feat, text_feat, obj_feat, phrase_feat, phrase_mask):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        obj_feats = concat_all_gather(obj_feat)
        phrase_feats = concat_all_gather(phrase_feat)
        phrase_masks = concat_all_gather(phrase_mask)
        valid_obj_feats = obj_feats[phrase_masks==1]
        valid_phrase_feats = phrase_feats[phrase_masks==1]

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

        valid_obj_num = valid_obj_feats.shape[0]
        obj_ptr = int(self.obj_queue_ptr)
        end_idx = min(obj_ptr+valid_obj_num, self.queue_size)
        self.object_queue[:, obj_ptr:end_idx] = valid_obj_feats.T[:, :end_idx-obj_ptr]
        self.phrase_queue[:, obj_ptr:end_idx] = valid_phrase_feats.T[:, :end_idx-obj_ptr]
        if end_idx == self.queue_size:
            obj_ptr = 0
        else:
            obj_ptr = obj_ptr + valid_obj_num
        self.obj_queue_ptr[0] = obj_ptr


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