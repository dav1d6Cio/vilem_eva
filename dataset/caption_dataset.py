import json
import os
import random
import torch
import pdb
import numpy as np

from torch.utils.data import Dataset
from transformers import BertTokenizer, BertTokenizerFast

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption, pre_caption_keep_comma_period

import nltk

nltk.data.path.append('nltk_data/')


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class re_train_dataset_replace(Dataset):
    def __init__(self, ann_file, transform, image_root, config, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}
        self.tokenizer = BertTokenizer.from_pretrained('weights/bert-base-uncased')
        self.stopwords = json.load(open('data/sw_spacy_choose_word_to_replace.json', 'r'))
        # 把逗号和句号也加进stopwords
        self.stopwords += [',', '.']

        if config['func'] == 'mlm':
            self.replace_func = self.replace_mlm
            self.rep_prob = config['prob']
            self.filter_sw = config['filter_sw']
            self.max_rep_num = config['max_rep_num']
            self.min_rep_num = config['min_rep_num']
        else:
            self.replace_func = self.replace_k
            self.rep_num = config['rep_num']
            self.filter_sw = config['filter_sw']

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def stopword_mask(self, tokens):
        rep_tokens = [token for token in tokens if token not in self.stopwords]
        mask = [1 if token not in self.stopwords else 0 for token in tokens]
        return rep_tokens, mask

    def replace_k(self, caption):
        """
        non_rep_tokens: 不需要替换的词的列表，比如stopwords
        rep_num: replace位置的数量，默认replace一个位置，如果replace多个的话，整个句子的语义缺失比较严重，
        replace的结果会不通顺
        filter_sw：不更换stopwords
        """
        token_input = self.tokenizer(caption, add_special_tokens=True, max_length=30, return_tensors='pt')
        input_ids = token_input['input_ids']
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        if self.filter_sw:
            rep_tokens, rep_mask = self.stopword_mask(input_tokens)
            rep_mask = np.array(rep_mask)

            rep_ids_candidate = np.where(rep_mask == 1)[0][1:-1]
            # 如果没有等于1的地方
            if len(rep_ids_candidate) == 0:
                mask_idx = torch.tensor(0).long()
            else:
                mask_idx = np.random.choice(rep_ids_candidate, self.rep_num, replace=False)[0]
                mask_idx = torch.tensor(mask_idx)
        else:
            mask_idx = torch.randint(1, (input_ids[0] != 0).sum() - 1, (self.rep_num,))[0]
        return mask_idx

    def replace_mlm(self, caption):
        mask_idx = torch.zeros(self.max_rep_num)
        token_input = self.tokenizer(caption, add_special_tokens=True, return_tensors='pt')
        input_ids = token_input['input_ids'][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 生成概率分布并采样
        prob_matrix = torch.full(input_ids.shape, self.rep_prob)
        rep_matrix = torch.bernoulli(prob_matrix)
        rep_matrix[input_ids == self.tokenizer.pad_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.cls_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.sep_token_id] = 0
        # 逗号和句号也mask掉
        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids([','])[0]] = 0
        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids(['.'])[0]] = 0

        if self.filter_sw:
            rep_tokens, rep_mask = self.stopword_mask(input_tokens)
            rep_mask = torch.tensor(rep_mask)
            rep_matrix[rep_mask == 0] = 0
        rep_ids_candidate = torch.where(rep_matrix == 1)[0][:self.max_rep_num]
        if len(rep_ids_candidate) < self.min_rep_num:
            # 如果需要filter_sw，且除了stopwords外有其他词，则从非stopwords中选min(min_rep_num, num_non_stopwords)个词
            if self.filter_sw and rep_mask.sum() > 2:
                rep_ids_candidate = torch.where(rep_mask == 1)[0][1:-1]  # [1:-1]去掉CLS和SEP
                num_select = min(self.min_rep_num, len(rep_ids_candidate))
                rep_ids_candidate = np.random.choice(rep_ids_candidate.numpy(), num_select, replace=False)
            else:
                num_select = min(self.min_rep_num, len(input_ids)-2)
                rep_ids_candidate = np.random.choice(list(range(1, len(input_ids)-1)), num_select, replace=False)  # 随机选一个
            rep_ids_candidate = torch.from_numpy(rep_ids_candidate)
        mask_idx[:len(rep_ids_candidate)] = rep_ids_candidate
        return mask_idx.long()
    
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        raw_caption = pre_caption_keep_comma_period(ann['caption'])
        caption = pre_caption(ann['caption'], self.max_words)
        rep_idx = self.replace_func(raw_caption)

        return image, raw_caption, caption, rep_idx, self.img_ids[ann['image_id']]

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30, debug=False, img_res=256):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
                    self.ann += json.load(open(f, 'r'))[:1000000]  # 用20w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.img_res = img_res
        self.bad_images = []

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        suffix = '/data/algceph/uasonchen/dataset'
        prefix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(suffix, prefix)).convert('RGB')
        except:
            image = Image.new('RGB', (self.img_res, self.img_res))
        image = self.transform(image)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return image, caption

class pretrain_dataset_replace(Dataset):
    def __init__(self, ann_file, transform, config, max_words=30, debug=False):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
                    self.ann += json.load(open(f, 'r'))[:1000000]  # 用20w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = BertTokenizer.from_pretrained('weights/bert-base-uncased')
        self.stopwords = json.load(open('data/sw_spacy_choose_word_to_replace.json', 'r'))
        # 把逗号和句号也加进stopwords
        self.stopwords += [',', '.']

        if config['func'] == 'mlm':
            self.replace_func = self.replace_mlm
            self.rep_prob = config['prob']
            self.filter_sw = config['filter_sw']
            self.max_rep_num = config['max_rep_num']
            self.min_rep_num = config['min_rep_num']
            
        elif config['func'] == 'tag':
            self.replace_func = self.replace_pos
        else:
            self.replace_func = self.replace_k
            self.rep_num = config['rep_num']
            self.filter_sw = config['filter_sw']
        self.filter_subword = config['filter_subword'] if 'filter_subword' in config.keys() else False
        if self.filter_subword:
            self.subwords = self.get_subwords()
            
    def get_subwords(self):
        sub_ids = []
        for i in range(len(self.tokenizer)):
            token = self.tokenizer.convert_ids_to_tokens(i)
            if token[:2] == '##':
                sub_ids.append(token)
        return sub_ids
    
    def subword_mask(self, tokens):
        sub_tokens = [token for token in tokens if token not in self.subwords]
        mask = [1 if token not in self.subwords else 0 for token in tokens]
        return sub_tokens, mask
    
    def stopword_mask(self, tokens):
        rep_tokens = [token for token in tokens if token not in self.stopwords]
        mask = [1 if token not in self.stopwords else 0 for token in tokens]
        return rep_tokens, mask

    def replace_k(self, caption):
        """
        non_rep_tokens: 不需要替换的词的列表，比如stopwords
        rep_num: replace位置的数量，默认replace一个位置，如果replace多个的话，整个句子的语义缺失比较严重，
        replace的结果会不通顺
        filter_sw：不更换stopwords
        """
        token_input = self.tokenizer(caption, add_special_tokens=True, max_length=30, return_tensors='pt')
        input_ids = token_input['input_ids']
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        if self.filter_sw or self.filter_subword:
            if self.filter_sw:
                rep_tokens, rep_mask = self.stopword_mask(input_tokens)
                rep_mask = np.array(rep_mask)
            else:
                rep_mask = np.ones(len(input_tokens))
            if self.filter_subword:
                sub_tokens, sub_mask = self.stopword_mask(input_tokens)
                sub_mask = np.array(sub_mask)
            else:
                sub_mask = np.ones(len(input_tokens))
            rep_mask = rep_mask * sub_mask

            rep_ids_candidate = np.where(rep_mask == 1)[0][1:-1]
            # 如果没有等于1的地方
            if len(rep_ids_candidate) < 2:
                mask_idx = torch.zeros((self.rep_num,)).long()
            else:
                mask_idx = np.random.choice(rep_ids_candidate, self.rep_num, replace=False)
                mask_idx = torch.tensor(mask_idx)
        else:
            mask_idx = torch.randint(1, (input_ids[0] != 0).sum() - 1, (self.rep_num,))[0]
        return mask_idx

    def replace_pos(self, caption, tag='NN'):
        mask_idx = torch.zeros(1)
        token_input = self.tokenizer(caption, add_special_tokens=True, return_tensors='pt')
        input_ids = token_input['input_ids'][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        pos_tags = nltk.pos_tag(input_tokens[1:-1])

        tag_pos_ids = []
        for i, pos_tag in enumerate(pos_tags):  # 不管cls和sep
            token, pos = pos_tag
            if tag in pos:
                tag_pos_ids.append(i + 1)
        if len(tag_pos_ids) == 0:
            return mask_idx.long()
        else:
            target_pos_ids = np.random.choice(tag_pos_ids, 1)[0]
            mask_idx[0] = target_pos_ids
            return mask_idx.long()

    def replace_random(self, caption):
        """
        像MLM一样，每个词都有20%的概率被replace
        filter_sw：不更换stopwords
        """
        mask_idx = torch.zeros(self.max_rep_num)
        token_input = self.tokenizer(caption, add_special_tokens=True, return_tensors='pt')
        input_ids = token_input['input_ids'][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 生成概率分布并采样
        prob_matrix = torch.full(input_ids.shape, self.rep_prob)
        rep_matrix = torch.bernoulli(prob_matrix)
        rep_matrix[input_ids == self.tokenizer.pad_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.cls_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.sep_token_id] = 0
        # 逗号和句号也mask掉

        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids([','])[0]] = 0
        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids(['.'])[0]] = 0

        if self.filter_sw:
            rep_tokens, rep_mask = self.stopword_mask(input_tokens)
            rep_mask = torch.tensor(rep_mask)

            rep_prob_mask = rep_mask * rep_matrix
            # 如果没有等于1的地方
            if rep_prob_mask.sum() == 0:  # 没选到可以replace的地方
                if rep_mask.sum() == 2:  # 没有可以replace的地方, 等于2是因为CLS和SEP位置必为1
                    #                     print('No stopwords: ', input_tokens)
                    # 直接从rep_matrix里挑出出些replace的word的pos index
                    rep_ids_candidate = torch.where(rep_matrix == 1)[0][:self.max_rep_num]  # 只取前8个
                    mask_idx[:len(rep_ids_candidate)] = rep_ids_candidate
                else:
                    num_rep = rep_mask.sum()
                    rep_ids_candidate = torch.where(rep_mask == 1)[0][1:-1]  # [1:-1]去掉CLS和SEP
                    num_rep_select = torch.randint(1, num_rep - 1, (1,))[0]  # num_rep里算上了CLS和SEP，所以只用-1即可
                    rep_ids_candidate = np.random.choice(rep_ids_candidate, num_rep_select.item(), replace=False)
                    mask_idx[:len(rep_ids_candidate)] = torch.from_numpy(rep_ids_candidate)[:self.max_rep_num]

            else:  # 有可以replace的地方
                rep_ids_candidate = torch.where(rep_prob_mask == 1)[0][:self.max_rep_num]
                mask_idx[:len(rep_ids_candidate)] = rep_ids_candidate
        else:
            rep_ids_candidate = torch.where(rep_matrix == 1)[0][:self.max_rep_num]
            mask_idx[:len(rep_ids_candidate)] = rep_ids_candidate
        return mask_idx.long()

    def replace_mlm(self, caption):
        mask_idx = torch.zeros(self.max_rep_num)
        token_input = self.tokenizer(caption, add_special_tokens=True, return_tensors='pt')
        input_ids = token_input['input_ids'][0]
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # 生成概率分布并采样
        prob_matrix = torch.full(input_ids.shape, self.rep_prob)
        rep_matrix = torch.bernoulli(prob_matrix)
        rep_matrix[input_ids == self.tokenizer.pad_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.cls_token_id] = 0
        rep_matrix[input_ids == self.tokenizer.sep_token_id] = 0
        # 逗号和句号也mask掉
        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids([','])[0]] = 0
        rep_matrix[input_ids == self.tokenizer.convert_tokens_to_ids(['.'])[0]] = 0
        
        if self.filter_sw:
            rep_tokens, rep_mask = self.stopword_mask(input_tokens)
            rep_mask = torch.tensor(rep_mask)
            rep_matrix[rep_mask == 0] = 0
        if self.filter_subword:
            sub_tokens, sub_mask = self.subword_mask(input_tokens)
            sub_mask = torch.tensor(sub_mask)
            rep_matrix[sub_mask == 0] = 0
            
        rep_ids_candidate = torch.where(rep_matrix == 1)[0][:self.max_rep_num]
        if len(rep_ids_candidate) < self.min_rep_num:
            # 如果需要filter_sw，且除了stopwords外有其他词，则从非stopwords中选min(min_rep_num, num_non_stopwords)个词
            if self.filter_sw and rep_mask.sum() > 2:
                rep_ids_candidate = torch.where(rep_mask == 1)[0][1:-1]  # [1:-1]去掉CLS和SEP
                num_select = min(self.min_rep_num, len(rep_ids_candidate))
                rep_ids_candidate = np.random.choice(rep_ids_candidate.numpy(), num_select, replace=False)
            else:
                num_select = min(self.min_rep_num, len(input_ids)-3) # 这里减3是因为还要排除句子末尾的句号
                # len(input_ids)-2也是为了排除SEP和句号
                rep_ids_candidate = np.random.choice(list(range(1, len(input_ids)-2)), num_select, replace=False)  # 随机选一个
            rep_ids_candidate = torch.from_numpy(rep_ids_candidate)
        mask_idx[:len(rep_ids_candidate)] = rep_ids_candidate
        return mask_idx.long()

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        if type(ann['caption']) == list:
            caption = random.choice(ann['caption'])
        else:
            caption = ann['caption']

        raw_caption = pre_caption_keep_comma_period(caption)
        caption = pre_caption(caption, self.max_words)
        rep_idx = self.replace_func(raw_caption)

        src_suffix = '/data/algceph/uasonchen/dataset'
        tgt_suffix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
            if 'bbox' in ann.keys():
                x, y, w, h = ann['bbox']
                image = image.crop((x, y, x+w, y+h))
        except:
            print(ann['image'])
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        return image, raw_caption, caption, rep_idx, ann['image'].replace(src_suffix, tgt_suffix)