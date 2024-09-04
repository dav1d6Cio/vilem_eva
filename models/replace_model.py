from transformers import BertForMaskedLM
import transformers
import pdb
import json

transformers.logging.set_verbosity_error()

import torch
from torch import nn
import torch.nn.functional as F

stopwords = json.load(open('data/sw_spacy_keep_preposition_v1.json', 'r'))
sim_words = json.load(open('data/sim_words_wordnet_add_a_one_1.json', 'r'))
unxp_vocab = ['.', ';', '?', '!', '|', '...', ':', '-', '।', ',', '"', "'", '##¤', ')', '[UNK]', '॥', '#', '+',
              '$', ']', '&', '/']


class Replace_Model(nn.Module):
    def __init__(self,
                 tokenizer,
                 config,
                 bert_size=None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.bert = BertForMaskedLM.from_pretrained(bert_size if bert_size is not None else 'weights/bert-base-uncased')
        self.strategy = config['strategy'] if 'strategy' in config.keys() else 'random'
        self.topk = config['topk'] if 'topk' in config.keys() else 4
        self.consider_num = config['consider_num'] if 'consider_num' in config.keys() else 20
        self.rerank = config['rerank'] if 'rerank' in config.keys() else False
        self.avoid_simwords = config['avoid_simwords'] if 'avoid_simwords' in config.keys() else False
        self.avoid_subwords = config['avoid_subwords'] if 'avoid_subwords' in config.keys() else False
        self.avoid_stopwords = config['avoid_stopwords'] if 'avoid_stopwords' in config.keys() else False
        self.unexpect_vocab = unxp_vocab
        if self.avoid_stopwords:
            self.unexpect_vocab += stopwords
        self.vocab_mask = self.create_vocab_mask(self.unexpect_vocab)
        self.sim_words = self.convert_tokens_to_ids(sim_words)
        self.subword_mask = self.create_subword_mask()

    def convert_tokens_to_ids(self, sim_words_dict):
        sim_ids_dict = {}
        for k, v in sim_words_dict.items():
            new_k = self.tokenizer.convert_tokens_to_ids([k])[0]
            new_v = self.tokenizer.convert_tokens_to_ids(v)
            sim_ids_dict[new_k] = new_v
        return sim_ids_dict

    def mask_sim_words(self, logits, rep_ids, original_ids):
        for i in range(len(rep_ids)):
            origin_id = original_ids[i].item()
            if origin_id not in self.sim_words.keys():
                continue
            sim_ids = self.sim_words[origin_id]
            logits[i, rep_ids[i], sim_ids] = -10000
        return logits
    
    def create_subword_mask(self):
        mask = torch.zeros(len(self.tokenizer), dtype=torch.bool)
        for i in range(len(self.tokenizer)):
            token = self.tokenizer.convert_ids_to_tokens(i)
            if token[:2] == '##':
                mask[i] = True
        return mask
                
    def mask_subwords(self, logits, rep_ids, origin_ids):
        for i in range(len(rep_ids)):
            origin_token = self.tokenizer.convert_ids_to_tokens(origin_ids[i].item())
            if not origin_token[:2] == '##':
                logits[i, rep_ids[i], self.subword_mask] = -10000
        return logits

    def forward(self, input_ids, attention_mask, rep_ids, template_token_ids=None):
        #text_input, 
        # rep_ids：需要替换的token对应位置的index
        # input_ids = text_input.input_ids
        batch_size, text_len = input_ids.shape
        masked_input_ids = input_ids.clone()
        origin_ids = input_ids[range(batch_size), rep_ids]  # 64
        masked_input_ids[range(batch_size), rep_ids] = self.tokenizer.mask_token_id
        if self.strategy == 'wo_bert':
            rep_token_ids = torch.randint(len(self.tokenizer), (batch_size,)).to(input_ids.device)
            if template_token_ids is None:
                rep_output_ids = input_ids.clone()
            else:
                rep_output_ids = template_token_ids.clone()
            rep_output_ids[range(batch_size), rep_ids] = rep_token_ids
        
        elif self.strategy == 'topk':
            rep_token_ids = self.sample_topk(logits, true_logits, rep_ids, input_ids, attention_mask) # N,topk
            rep_output_ids = input_ids.unsqueeze(1).repeat(1, self.topk, 1)  # B,topk,L
            rep_output_ids[range(batch_size), :, rep_ids] = rep_token_ids
        else:
            output = self.bert(masked_input_ids, attention_mask=attention_mask)
            logits = output.logits  # N,L,V
            logits += (1 - self.vocab_mask.to(logits.device)) * -10000  # logit: 64,30,xxxx   rep_ids: 64
            # 将logits中rep_ids位置原本的token id mask掉
            logits[range(batch_size), rep_ids, origin_ids] = -10000

            if self.avoid_simwords:
                logits = self.mask_sim_words(logits, rep_ids, origin_ids)
            if self.avoid_subwords:
                logits = self.mask_subwords(logits, rep_ids, origin_ids)
            if self.rerank:
                true_logits = self.bert(input_ids, attention_mask=attention_mask).logits
                true_logits += (1 - self.vocab_mask.to(logits.device)) * -10000  # logit: 64,30,xxxx   rep_ids: 64
                true_logits[range(batch_size), rep_ids, origin_ids] = -10000
            else:
                true_logits = None

            if self.strategy == 'uniform':
                rep_token_ids = self.sample_uniform(logits, true_logits)
            elif self.strategy == 'embed_sim':
                rep_token_ids = self.sample_embed_sim(logits, origin_ids, rep_ids)
            elif self.strategy == 'greedy':
                _, rep_token_ids = logits.max(dim=-1) # N,L
            else:
                rep_token_ids = self.sample_multinomial_simple(logits, true_logits, rep_ids, input_ids, attention_mask)
            if template_token_ids is None:
                rep_output_ids = input_ids.clone()
            else:
                rep_output_ids = template_token_ids.clone()
            if len(rep_output_ids.shape) == len(rep_token_ids.shape):
                rep_output_ids[range(batch_size), rep_ids] = rep_token_ids[range(batch_size), rep_ids]
            else:
                rep_output_ids[range(batch_size), rep_ids] = rep_token_ids

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

    def sample_multinomial(self, logits, true_logits=None, rep_ids=None, input_ids=None):
        # logits: N,L,V
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
    
    def sample_topk(self, logits, true_logits=None, rep_ids=None, input_ids=None, attention_mask=None):
        logits = logits[torch.arange(logits.shape[0]), rep_ids] # N,vocab_size
        topk_logits, topk_ids = logits.topk(self.topk, dim=-1)  # N,20
        return topk_ids
        
    def sample_multinomial_simple(self, logits, true_logits=None, rep_ids=None, input_ids=None, attention_mask=None):
        # logits: N,L,V
        # 只考虑replace words的logits
        logits = logits[torch.arange(logits.shape[0]), rep_ids] # N,vocab_size
        topk_logits, topk_ids = logits.topk(20, dim=-1)  # N,20
        topk_probs = torch.softmax(topk_logits, dim=-1)  # N,20
        if self.strategy == 'uniform':
            topk_probs = torch.ones_like(topk_probs)
        index_in_topk = torch.multinomial(topk_probs, 10, replacement=True)  # N,1
        rep_token_ids = topk_ids[torch.arange(logits.shape[0]).unsqueeze(-1), index_in_topk]
        return rep_token_ids[:, 0]
    
    def sample_embed_sim(self, logits, origin_ids, rep_pos_ids):
        # logits: N,L,V
        batch_size, text_len, vocab_size = logits.shape
        topk_logits, topk_ids = logits.topk(self.topk, dim=-1) # N,L,self.topk
        rep_tokens = topk_ids[torch.arange(batch_size), rep_pos_ids]  # N,self.topk
        
        origin_embeds = self.bert.bert.embeddings.word_embeddings(origin_ids) # N,C
        rep_embeds = self.bert.bert.embeddings.word_embeddings(rep_tokens) # N, self.topk, C
        sim_matrix = F.normalize(origin_embeds, dim=-1).unsqueeze(1) @ F.normalize(rep_embeds, dim=-1).transpose(1,2) # N,1,self.topk
        sim_matrix = 1 - sim_matrix.squeeze(1)  # N,self.topk, convert sim to distance, larger disntance --> larger prob
        prob_matrix = torch.softmax(sim_matrix, dim=-1)
        index_in_topk = torch.multinomial(prob_matrix, 1)
        
        output_token_ids = torch.gather(rep_tokens, -1, index_in_topk).squeeze(-1)
        
        rep_token_ids = torch.zeros((batch_size, text_len), device=logits.device).long()
        rep_token_ids[torch.arange(batch_size), rep_pos_ids] = output_token_ids

        return rep_token_ids

    def create_vocab_mask(self, vocab):
        mask = torch.ones(1, 1, len(self.tokenizer), dtype=torch.float32)
        vocab = set(vocab)
        for i in range(len(self.tokenizer)):
            token = self.tokenizer.convert_ids_to_tokens(i)
            if token in vocab:
                mask[0, 0, i] = 0
        return mask