from transformers import BertForMaskedLM
import transformers
import pdb
import json

transformers.logging.set_verbosity_error()

import torch
from torch import nn

stopwords = json.load(open('data/sw_spacy_rm.json', 'r'))
unxp_vocab = ['.', ';', '?', '!', '|', '...', ':', '-', '।', ',', '"', "'", '##¤', ')', '[UNK]', '॥', '#', '+',
              '$', ']']


class Replace_Model(nn.Module):
    def __init__(self,
                 tokenizer,
                 config,
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertForMaskedLM.from_pretrained('weights/bert-base-uncased')
        self.topk = config['topk']
        self.consider_num = config['consider_num']
        self.rerank = config['rerank']
        self.unexpect_vocab = unxp_vocab
        if config['avoid_stopwords']:
            self.unexpect_vocab += stopwords
        self.vocab_mask = self.create_vocab_mask(self.unexpect_vocab)

    def forward(self, text, rep_ids, template_token_ids=None):
        # rep_ids：需要替换的token对应位置的index
        input_ids = text.input_ids
        batch_size, text_len = input_ids.shape
        masked_input_ids = input_ids.clone()
        origin_ids = input_ids[range(batch_size), rep_ids]  # 64
        masked_input_ids[range(batch_size), rep_ids] = self.tokenizer.mask_token_id
        output = self.bert(masked_input_ids, attention_mask=text.attention_mask)
        logits = output.logits  # N,L,V
        logits += (1 - self.vocab_mask.to(logits.device)) * -10000  # logit: 64,30,xxxx   rep_ids: 64
        # 将logits中rep_ids位置原本的token id mask掉
        logits[range(batch_size), rep_ids, origin_ids] = -10000

        if self.rerank:
            true_logits = self.bert(text.input_ids, attention_mask=text.attention_mask).logits
            true_logits += (1 - self.vocab_mask.to(logits.device)) * -10000  # logit: 64,30,xxxx   rep_ids: 64
            true_logits[range(batch_size), rep_ids, origin_ids] = -10000
        else:
            true_logits = None

        rep_token_ids = self.sample_multinomial(logits, true_logits)
        if rep_token_ids is None:
            rep_output_ids = input_ids.clone()
        else:
            rep_output_ids = template_token_ids.clone()
        rep_output_ids[range(batch_size), rep_ids] = rep_token_ids[range(batch_size), rep_ids]

        return rep_output_ids

    def sample_uniform(self, logits, true_logits=None):
        batch_size, text_len, _ = logits.shape
        if true_logits is not None:
            mask_predictions = logits.softmax(-1)
            predictions = true_logits.softmax(-1)
            mask_max_ids = torch.topk(mask_predictions, self.consider_num, 2)[1]
            true_prob_mask_max_ids = torch.gather(predictions, 2, mask_max_ids)
            lang_prob_mask_max_ids = torch.gather(mask_predictions, 2, mask_max_ids)
            score = lang_prob_mask_max_ids / (true_prob_mask_max_ids + 1e-6)  # B,L,consider_num
            score_idx = torch.topk(score, self.topk, dim=-1)[1]
            topk_ids = torch.gather(mask_max_ids, 2, score_idx)
        else:
            topk_logits, topk_ids = logits.topk(self.topk, dim=-1)  # N,L,topk
        select_random = torch.randint(0, self.topk, (batch_size, text_len, 1), device=logits.device)  # N,L,1
        rep_token_ids = torch.gather(topk_ids, -1, select_random).squeeze(-1)
        return rep_token_ids

    def sample_multinomial(self, logits, true_logits=None):
        # logits: N,L,V
        # 暂定选前20个进行softmax
        if true_logits is not None:
            mask_predictions = logits.softmax(-1)
            predictions = true_logits.softmax(-1)
            mask_max_ids = torch.topk(mask_predictions, self.consider_num, 2)[1]
            true_prob_mask_max_ids = torch.gather(predictions, 2, mask_max_ids)
            lang_prob_mask_max_ids = torch.gather(mask_predictions, 2, mask_max_ids)
            score = lang_prob_mask_max_ids / (true_prob_mask_max_ids + 1e-6)  # B,L,consider_num
            score_idx = torch.topk(score, self.topk, dim=-1)[1]
            topk_ids = torch.gather(mask_max_ids, 2, score_idx)
            topk_logits = torch.gather(logits, 2, topk_ids)
        else:
            topk_logits, topk_ids = logits.topk(20, dim=-1)
        topk_probs = torch.softmax(topk_logits, dim=-1)
        batch_size, text_len, vocab_size = topk_probs.shape
        index_in_topk = torch.multinomial(topk_probs.view(-1, vocab_size), 1)
        index_in_topk = index_in_topk.view(batch_size, text_len, 1)
        rep_token_ids = torch.gather(topk_ids, -1, index_in_topk).squeeze(-1)
        return rep_token_ids

    def create_vocab_mask(self, vocab):
        mask = torch.ones(1, 1, len(self.tokenizer), dtype=torch.float32)
        vocab = set(vocab)
        for i in range(len(self.tokenizer)):
            token = self.tokenizer.convert_ids_to_tokens(i)
            if token in vocab:
                mask[0, 0, i] = 0
        return mask