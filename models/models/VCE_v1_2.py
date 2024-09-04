from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.detr_bert_code import VcTModel

import torch
from torch import nn
import torch.nn.functional as F

import pdb

class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.distill = config['distill']
        embed_dim = config['embed_dim']
        vision_width = config['vision_width']
        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        vct_config = BertConfig.from_json_file(config['vct_config'])
        self.vct = VcTModel.from_pretrained(text_encoder, config=vct_config, num_queries=config['num_queries'])

        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.concept_proj = nn.Linear(vision_width, embed_dim)
        self.word_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.temp_cpt = nn.Parameter(torch.ones([]) * config['temp'])
        self.itm_head = nn.Linear(text_width, 2)

    def forward(self, image, text, alpha, idx):

        image_layers = self.visual_encoder(image, return_all_hidden_states=True)
        image_embeds = image_layers[-1]  # image layers共有15个元素，分别是pos embeds，patchify的结果，12层的输出，最后norm层的输出
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask, return_dict=True, mode='text')
        text_embeds = text_output.last_hidden_state

        # loss_ita = self.forward_cls_contrastive(image_embeds, text_embeds)
        loss_ita = torch.tensor(0).to(text_embeds.device)

        # concept-level contrastive learning
        encoder_hidden_states = [image_layers[i][:, 1:] for i in range(1, 13, 2)]  # remove CLS
        cvt_layers = self.vct(encoder_hidden_states=encoder_hidden_states,
                                encoder_attention_mask=image_atts[:, 1:],  # remove CLS
                                return_dict=True,
                                )
        concept_feat = cvt_layers.last_hidden_state
        loss_itm = self.forward_concept_contrastive(concept_feat, text_embeds, text.attention_mask)
        return loss_ita, loss_itm

    def forward_cls_contrastive(self, image_embeds, text_embeds):
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        image_feat_all = AllGather.apply(image_feat)
        text_feat_all = AllGather.apply(text_feat)

        sim_i2t = image_feat @ text_feat_all.t() / self.temp
        sim_t2i = text_feat @ image_feat_all.t() / self.temp

        rank_id = torch.distributed.get_rank()
        batch_size = text_feat.shape[0]
        total_batch_size = text_feat_all.shape[0]
        sim_targets = torch.zeros((total_batch_size, total_batch_size), dtype=sim_i2t.dtype, device=sim_i2t.device)
        sim_targets.fill_diagonal_(1)
        sim_targets = sim_targets[rank_id * batch_size: (rank_id + 1) * batch_size]

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

    def forward_concept_contrastive(self, concept_feat, text_embeds, text_mask):
        # B,L,C
        concept_feat = F.normalize(self.concept_proj(concept_feat), dim=-1)
        word_feat = F.normalize(self.word_proj(text_embeds), dim=-1)
        concept_mask = torch.ones(concept_feat.size()[:-1], dtype=torch.long).to(concept_feat.device)
        word_mask = self.remove_CLS_SEP(text_mask)

        concept_feat_all = AllGather.apply(concept_feat)
        concept_mask_all = AllGather.apply(concept_mask)
        word_feat_all = AllGather.apply(word_feat)
        word_mask_all = AllGather.apply(word_mask)

        sim_i2t = self.get_sim_with_alignment(word_feat_all, concept_feat, word_mask_all, concept_mask) / self.temp_cpt
        sim_t2i = self.get_sim_with_alignment(concept_feat_all, word_feat, concept_mask_all, word_mask) / self.temp_cpt

        rank_id = torch.distributed.get_rank()
        batch_size = word_feat.shape[0]
        total_batch_size = word_feat_all.shape[0]
        sim_targets = torch.zeros((total_batch_size, total_batch_size), dtype=sim_i2t.dtype, device=sim_i2t.device)
        sim_targets.fill_diagonal_(1)
        sim_targets = sim_targets[rank_id * batch_size: (rank_id + 1) * batch_size]

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        return loss_ita

    def get_sim_with_alignment(self, sequence_output, visual_output, attention_mask, video_mask):
        b_ref, s_ref, h_ref = visual_output.size()
        b_cand, s_cand, h_cand = sequence_output.size()
        # 计算mask
        sim_matrix = []
        for i in range(b_ref):
            sim_matrix_row = self.max_region_sum_word(sequence_output, visual_output[i:i+1], attention_mask, video_mask[i:i+1])
            sim_matrix.append(sim_matrix_row)
        # sim_matrix = self.max_region_sum_word(sequence_output, visual_output, seq_mask_expand, video_mask_expand, need_norm=False)
        sim_matrix = torch.stack(sim_matrix, dim=0)
        return sim_matrix

    def max_region_sum_word(self, sequence_output, visual_output, attention_mask, video_mask):
        video_pad = ~video_mask.bool().unsqueeze(-1)  # B,N
        seq_pad = ~attention_mask.bool().unsqueeze(-2)  # B,L

        alignments = torch.matmul(visual_output, sequence_output.transpose(-2, -1))
        alignments.masked_fill_(video_pad, -2).masked_fill_(seq_pad, -2)

        # row max col mean
        sim_matrix_row = alignments.max(-2)[0]  # B,B,L
        num_words = (sim_matrix_row > -2).sum(-1)
        sim_matrix_row[sim_matrix_row < -1] = 0
        sim_matrix_row = sim_matrix_row.sum(-1) / num_words  # B,B

        # col max row mean
        sim_matrix_col = alignments.max(-1)[0]  # B,B,L
        num_queries = (sim_matrix_col > -2).sum(-1)
        sim_matrix_col[sim_matrix_col < -1] = 0
        sim_matrix_col = sim_matrix_col.sum(-1) / num_queries  # B,B
        sim_matrix = sim_matrix_row + sim_matrix_col
        return sim_matrix

    def remove_CLS_SEP(self, mask):
        """
        set [CLS] and [SEP] as 0 in the mask
        """
        # 这里必须得先重新生成一个mask
        new_mask = torch.zeros_like(mask)
        new_mask[:] = mask[:]
        new_mask[:, 0] = 0  # N,L
        # new_mask[list(range(new_mask.shape[0])), (new_mask != 0).cumsum(1).argmax(1)] = 0
        for i in range(len(new_mask)):
            sep_idx = torch.max(torch.nonzero(new_mask[i]))
            new_mask[i][sep_idx] = 0
        return new_mask

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
        output = [torch.empty_like(grad_output)
                  for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, grad_output)
        output = sum(output)
        return (
            output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
        )

        # torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM)
        # return (
        #     grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
        #     None,
        # )