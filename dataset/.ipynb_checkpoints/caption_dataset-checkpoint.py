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
#         self.stopwords = json.load(open('data/sw_spacy_keep_preposition.json', 'r'))
        #         self.stopwords = json.load(open('data/sw_spacy_rm.json', 'r'))
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


class re_train_dataset_two_augs(Dataset):
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
        image1 = self.transform(image)
        image2 = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image1, image2, caption, self.img_ids[ann['image_id']]


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


class re_phrase_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, tokenizer, max_words=30, max_phrases=8,
                 max_words_per_phrase=12):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_phrases = max_phrases
        self.max_words_per_phrase = max_words_per_phrase
        self.img_ids = {}

        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def _process_phrases_v1(self, phrases):
        # 对于每个phrases，在前面加上三个[MASK]，然后再PAD到统一长度

        MASK_HEAD = '[MASK] [MASK] [MASK] '
        NON_PHRASE = ('[MASK] ' * 5).strip()
        phrases_add_mask = [MASK_HEAD + p for p in phrases]
        if len(phrases_add_mask) < self.max_phrases:
            phrases_add_mask += [NON_PHRASE] * (self.max_phrases - len(phrases_add_mask))
        if len(phrases_add_mask) > self.max_phrases:
            phrases_add_mask = phrases_add_mask[:self.max_phrases]
        phrase_input = self.tokenizer(phrases_add_mask, max_length=self.max_words_per_phrase, padding='max_length',
                                      return_tensors='pt', truncation=True)
        phrase_mask = torch.zeros((self.max_phrases,), dtype=torch.int)
        phrase_mask[:len(phrases)] = 1
        phrase_input['mask'] = phrase_mask
        return phrase_input

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        phrases = self._process_phrases_v1(ann['noun_phrases'])

        return image, caption, phrases, self.img_ids[ann['image_id']]


class re_phrase_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, tokenizer, max_words=30, max_phrases=8,
                 max_words_per_phrase=12):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_phrases = max_phrases
        self.max_words_per_phrase = max_words_per_phrase

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        self.phrase = []

        # 先不加phrase
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption, self.max_words))
                self.phrase.append(self._process_phrases_v1(ann['noun_phrases'][i]))  # 一个text，对应一个phrase list
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __len__(self):
        return len(self.image)

    def _process_phrases_v1(self, phrases):
        # 对于每个phrases，在前面加上三个[MASK]，然后再PAD到统一长度

        MASK_HEAD = '[MASK] [MASK] [MASK] '
        NON_PHRASE = ('[MASK] ' * 5).strip()
        phrases_add_mask = [MASK_HEAD + p for p in phrases]
        if len(phrases_add_mask) < self.max_phrases:
            phrases_add_mask += [NON_PHRASE] * (self.max_phrases - len(phrases_add_mask))
        if len(phrases_add_mask) > self.max_phrases:
            phrases_add_mask = phrases_add_mask[:self.max_phrases]
        phrase_input = self.tokenizer(phrases_add_mask, max_length=self.max_words_per_phrase, padding='max_length',
                                      return_tensors='pt', truncation=True)
        phrase_mask = torch.zeros((self.max_phrases,), dtype=torch.int)
        phrase_mask[:len(phrases)] = 1
        phrase_input['mask'] = phrase_mask
        return phrase_input

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


class pretrain_dataset_two_augs(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        try:
            #             image = Image.open(ann['image']).convert('RGB')
            image = Image.open(
                ann['image'].replace('/data/algceph/uasonchen/dataset', '/group/30042/uasonchen/datasets')).convert(
                'RGB')
        except:
            #             print(ann['image'].replace('/data/algceph/uasonchen/dataset', '/group/30042/uasonchen/datasets'), 'using blank images')
            image = Image.new('RGB', (256, 256))
        image1 = self.transform(image)
        image2 = self.transform(image)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return image1, image2, caption

class pretrain_dataset_replace(Dataset):
    def __init__(self, ann_file, transform, config, max_words=30, debug=False):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
#                 if 'cc12m' in f:
                    self.ann += json.load(open(f, 'r'))[:1000000]  # 用20w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = BertTokenizer.from_pretrained('weights/bert-base-uncased')
        self.stopwords = json.load(open('data/sw_spacy_choose_word_to_replace.json', 'r'))
#         self.stopwords = json.load(open('data/sw_spacy_keep_preposition.json', 'r'))
        #         self.stopwords = json.load(open('data/sw_spacy_rm.json', 'r'))
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
#             with open('json_pretrain/bad_images.txt', 'a') as f:
            print(ann['image'])
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        return image, raw_caption, caption, rep_idx, ann['image'].replace(src_suffix, tgt_suffix)


class pretrain_data_multi_replace(pretrain_dataset_replace):
    def __getitem__(self, index):
        ann = self.ann[index]

        if type(ann['caption']) == list:
            caption = random.choice(ann['caption'])
        else:
            caption = ann['caption']

        raw_caption = pre_caption_keep_comma_period(caption)
        caption = pre_caption(caption, self.max_words)
        rep_idx_list = []
        for i in range(4):
            rep_idx = self.replace_func(raw_caption)
            rep_idx_list.append(rep_idx)
        rep_idx_list = torch.stack(rep_idx_list, dim=0)

        src_suffix = '/data/algceph/uasonchen/dataset'
        tgt_suffix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
        except:
#             with open('json_pretrain/bad_images.txt', 'a') as f:
            print(ann['image'])
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        return image, raw_caption, caption, rep_idx_list, ann['image'].replace(src_suffix, tgt_suffix)

class pretrain_dataset_phrase(Dataset):
    def __init__(self, ann_file, transform, tokenizer, max_words=30, max_phrases=8, max_words_per_phrase=12):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_phrases = max_phrases
        self.max_words_per_phrase = max_words_per_phrase

    def __len__(self):
        return len(self.ann)

    def _process_phrases(self, phrases):
        phrases_output = torch.full((self.max_phrases, self.max_words_per_phrase), self.tokenizer.mask_token_id)
        phrases_output[:, 0] = self.tokenizer.cls_token_id
        if len(phrases) > self.max_phrases:
            phrases = phrases[:self.max_phrases]
        for p_idx, phrase in enumerate(phrases):
            template_token = ['[CLS]'] + ['[MASK]'] * (self.max_words_per_phrase - 1)
            phrase = pre_caption(phrase, self.max_words)
            phrase_token = self.tokenizer.tokenize(phrase)
            if len(phrase_token) > self.max_words_per_phrase - 1:
                phrase_token = phrase_token[1 - self.max_words_per_phrase:]
            if len(phrase_token) > 0:  # 有token的时候才需要这么操作，没有token不需要
                template_token[-len(phrase_token):] = phrase_token
            token_ids = self.tokenizer.convert_tokens_to_ids(template_token)
            phrases_output[p_idx] = torch.tensor(token_ids, dtype=torch.int)
        # phrase_mask, shape: max_phrases. 1 for valid phrase, 0 for no phrases
        phrase_mask = torch.zeros((self.max_phrases,), dtype=torch.int)
        phrase_mask[:len(phrases)] = 1
        phrase_attention_mask = torch.ones_like(phrases_output)
        return {'input_ids': phrases_output, 'attention_mask': phrase_attention_mask, 'mask': phrase_mask}

    def _process_phrases_v1(self, phrases):
        # 对于每个phrases，在前面加上三个[MASK]，然后再PAD到统一长度

        MASK_HEAD = '[MASK] [MASK] [MASK] '
        NON_PHRASE = ('[MASK] ' * 5).strip()
        phrases_add_mask = [MASK_HEAD + p for p in phrases]
        if len(phrases_add_mask) < self.max_phrases:
            phrases_add_mask += [NON_PHRASE] * (self.max_phrases - len(phrases_add_mask))
        if len(phrases_add_mask) > self.max_phrases:
            phrases_add_mask = phrases_add_mask[:self.max_phrases]
        phrase_input = self.tokenizer(phrases_add_mask, max_length=self.max_words_per_phrase, padding='max_length',
                                      return_tensors='pt', truncation=True)
        phrase_mask = torch.zeros((self.max_phrases,), dtype=torch.int)
        phrase_mask[:len(phrases)] = 1
        phrase_input['mask'] = phrase_mask
        return phrase_input

    def __getitem__(self, index):
        ann = self.ann[index]
        src_suffix = '/data/algceph/uasonchen/dataset'
        tgt_suffix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
        phrases = self._process_phrases_v1(ann['noun_phrases'])

        return image, caption, phrases


class pretrain_dataset_attribute(Dataset):
    def __init__(self, ann_file, transform, tokenizer, max_words=30, max_phrases=8, max_words_per_phrase=12):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_phrases = max_phrases
        self.max_words_per_phrase = max_words_per_phrase
        self.tokenizer = tokenizer
        self.stopwords = json.load(open('data/sw_spacy_rm.json', 'r'))

    def stopword_mask(self, tokens):
        rep_tokens = [token for token in tokens if token not in self.stopwords]
        return rep_tokens

    def _process_labels(self, caption):
        caption_input = self.tokenizer(caption)
        input_ids = caption_input['input_ids'][1:-1]
        input_token = self.tokenizer.convert_ids_to_tokens(input_ids)
        content_token = self.stopword_mask(input_token)
        content_ids = self.tokenizer.convert_tokens_to_ids(content_token)
        label = torch.zeros((len(self.tokenizer),), dtype=torch.int)
        label[content_ids] = 1
        return label

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        src_suffix = '/data/algceph/uasonchen/dataset'
        tgt_suffix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        label = self._process_labels(caption)
        return image, caption, label


class re_attribute_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, tokenizer, max_words=30, max_phrases=8,
                 max_words_per_phrase=12):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.max_phrases = max_phrases
        self.max_words_per_phrase = max_words_per_phrase
        self.stopwords = json.load(open('data/sw_spacy_rm.json', 'r'))

    def stopword_mask(self, tokens):
        rep_tokens = [token for token in tokens if token not in self.stopwords]
        return rep_tokens

    def _process_multi_labels(self, captions):
        token_ids = []
        for caption in captions:
            caption_input = self.tokenizer(caption)
            input_ids = caption_input['input_ids'][1:-1]
            input_token = self.tokenizer.convert_ids_to_tokens(input_ids)
            content_token = self.stopword_mask(input_token)
            content_ids = self.tokenizer.convert_tokens_to_ids(content_token)
            token_ids += content_ids
        label = torch.zeros((len(self.tokenizer),), dtype=torch.int)
        label[token_ids] = 1
        return label

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        captions = self.ann[index]['caption']
        label = self._process_multi_labels(captions)

        return image, label


class pretrain_dataset_tree(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.tokenizer = BertTokenizerFast.from_pretrained('weights/bert-base-uncased')

    def __len__(self):
        return len(self.ann)

    def clear_tree(self, adj, word_offsets, cur_len):
        new_offsets = []
        for offset in word_offsets:
            beg_idx, end_idx = offset
            if beg_idx >= cur_len:
                break
            elif end_idx > cur_len:
                new_offsets.append([beg_idx, cur_len])
            else:
                new_offsets.append(offset)
        new_adj = adj[:len(new_offsets)]
        return new_adj, new_offsets

    def sample_phrase(self, adj, word_offsets, token_offsets, num_sample=1, caption_input=None, text=None, ann=None):
        phrase_beg_end_idx = []
        if len(adj) < (num_sample + 1):
            indices = np.random.choice(range(1, len(adj)), num_sample, replace=True).tolist()
            print(text)
            print(adj)
        else:
            indices = np.random.choice(range(1, len(adj)), num_sample, replace=False).tolist()  # 暂时不选择整句话进行mask
        for index in indices:
            word_beg_idx, word_end_idx = word_offsets[index]
            phrase_span = []
            for token_pos_idx, token_offset in enumerate(token_offsets):
                token_beg_idx, token_end_idx = token_offset
                if token_beg_idx >= word_beg_idx and token_end_idx <= word_end_idx:
                    phrase_span.append(token_pos_idx + 1)  # 需要+1，因为index=0的位置是CLS
            phrase_beg_end_idx.append([phrase_span[0], phrase_span[-1] + 1])  # 需要+1，因为span是包含phrase_span[-1]的
        return torch.Tensor(phrase_beg_end_idx).long()

    def __getitem__(self, index):
        pre_suffix = '/data/algceph/uasonchen/dataset'
        cur_suffix = '/group/30042/uasonchen/datasets'
        ann = self.ann[index]
        try:
            image = Image.open(ann['image'].replace(pre_suffix, cur_suffix)).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)
        caption = ann['caption']
        caption = caption.strip(' ')
        adj = ann['adj']
        word_offsets = ann['offsets']
        caption_input = self.tokenizer(caption, max_length=self.max_words + 1, return_offsets_mapping=True,
                                       truncation=True)
        token_offsets = caption_input['offset_mapping'][1:-1]  # 去掉头尾的offsets，因为对应的是CLS和SEP的offsets
        adj, word_offsets = self.clear_tree(ann['adj'], ann['offsets'], cur_len=token_offsets[-1][-1])
        phrase_spans = self.sample_phrase(adj, word_offsets, token_offsets, num_sample=3, caption_input=caption_input,
                                          text=caption, ann=ann)

        return image, caption, phrase_spans


class pretrain_dataset_beit(Dataset):
    def __init__(self, ann_file, transform, max_words=30, debug=False):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
                    self.ann += json.load(open(f, 'r'))[:1000000]  # 用100w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words

        from torchvision import transforms
        from .beit_transforms import RandomResizedCropAndInterpolationWithTwoPic
        from dall_e.utils import map_pixels
        from .masking_generator import MaskingGenerator

        self.common_transform = transforms.Compose([
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(size=256, second_size=128,
                interpolation='bicubic', second_interpolation="lanczos",),
            ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711]))
        ])

        self.visual_token_transform = transforms.Compose([
            transforms.ToTensor(),
            map_pixels,
        ])

        self.masked_position_generator = MaskingGenerator(
            input_size=16, num_masking_patches=100,
            max_num_patches=None,
            min_num_patches=24,
        )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        pre_suffix = '/data/algceph/uasonchen/dataset'
        cur_suffix = '/group/30042/uasonchen/datasets'

        try:
            image = Image.open(ann['image'].replace(pre_suffix, cur_suffix)).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
        image_ret = self.transform(image)
        caption = pre_caption(ann['caption'], self.max_words)

        for_patches, for_visual_tokens = self.common_transform(image)
        image_beit = self.patch_transform(for_patches)
        image_dalle = self.visual_token_transform(for_visual_tokens)
        image_mask = self.masked_position_generator()

        return image_ret, caption, image_beit, image_dalle, image_mask

class pretrain_dataset_cutmix(Dataset):
    def __init__(self, ann_file, transform, max_words=30, debug=False):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
                    self.ann += json.load(open(f, 'r'))  # 用100w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.bad_images = []
        from .masking_generator import MaskingGenerator
        self.masked_position_generator = MaskingGenerator(
            input_size=16, num_masking_patches=128,
            max_num_patches=None,
            min_num_patches=24,
        )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        pre_fix = '/data/algceph/uasonchen/dataset'
        cur_fix = '/group/30042/uasonchen/datasets'
        # load indexed image
        try:
            image = Image.open(ann['image'].replace(pre_fix, cur_fix)).convert('RGB')
        except:
            image = Image.new('RGB', (256, 256))
        image = self.transform(image)

        # load random image for mixup
        rand_idx = np.random.randint(0, len(self.ann))
        try:
            rand_image = Image.open(self.ann[rand_idx]['image'].replace(pre_fix, cur_fix)).convert('RGB')
        except:
            rand_image = Image.new('RGB', (256, 256))
        rand_image = self.transform(rand_image)
        image_mask = self.masked_position_generator().reshape(-1)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, rand_image, image_mask

class re_train_dataset_cutmix(Dataset):
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

        from .masking_generator import MaskingGenerator
        self.masked_position_generator = MaskingGenerator(
            input_size=24, num_masking_patches=288,
            max_num_patches=None,
            min_num_patches=24,
        )

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        rand_idx = np.random.randint(0, len(self.ann))
        rand_image_path = os.path.join(self.image_root, self.ann[rand_idx]['image'])
        rand_image = Image.open(rand_image_path).convert('RGB')
        rand_image = self.transform(rand_image)
        image_mask = self.masked_position_generator().reshape(-1)
        
        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, rand_image, image_mask, self.img_ids[ann['image_id']]

class pretrain_dataset_cutmix_weak_aug(Dataset):
    def __init__(self, ann_file, transform, max_words=30, debug=False):
        self.ann = []
        if debug:
            for f in ann_file:
                if 'cc3m_train' in f:
                    self.ann += json.load(open(f, 'r'))  # 用100w数据进行调试
        else:
            for f in ann_file:
                self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.max_words = max_words
        self.bad_images = []
        from .masking_generator import MaskingGenerator
        self.masked_position_generator = MaskingGenerator(
            input_size=16, num_masking_patches=128,
            max_num_patches=None,
            min_num_patches=24,
        )

        from torchvision import transforms
        self.common_transform = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]),
                std=torch.tensor([0.26862954, 0.26130258, 0.27577711]))
        ])

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        pre_fix = '/data/algceph/uasonchen/dataset'
        cur_fix = '/group/30042/uasonchen/datasets'
        # load indexed image
        try:
            raw_image = Image.open(ann['image'].replace(pre_fix, cur_fix)).convert('RGB')
        except:
            raw_image = Image.new('RGB', (256, 256))
        image = self.transform(raw_image)
        image_weak = self.common_transform(raw_image)

        # load random image for mixup
        rand_idx = np.random.randint(0, len(self.ann))
        try:
            rand_image = Image.open(self.ann[rand_idx]['image'].replace(pre_fix, cur_fix)).convert('RGB')
        except:
            rand_image = Image.new('RGB', (256, 256))
        rand_image = self.common_transform(rand_image)
        image_mask = self.masked_position_generator().reshape(-1)

        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, image_weak, rand_image, image_mask