import json
import os
import random
import torch
import pdb
import numpy as np
import math

from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip, resize
from transformers import BertTokenizer, BertTokenizerFast

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption, pre_caption_keep_comma_period

import nltk

nltk.data.path.append('nltk_data/')

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
        result = None
        while result is None:
            try:
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
                image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
                image = self.transform(image)
                result = True
            except:
                 index = random.randint(0, self.length-1)

        return image, raw_caption, caption, rep_idx, ann['image'].replace(src_suffix, tgt_suffix)


class pretrain_dataset_region(pretrain_dataset_replace):
    def __init__(self, ann_file, transform, config, max_words=30, debug=False):
        super().__init__(ann_file=ann_file, transform=transform, config=config['rep_sampler'], max_words=max_words, debug=debug)
        self.max_regions = config['region']['max_regions']
        self.image_res = config['image_res']
        self.patch_size = config['patch_size']
        self.batch_size = config['region']['batch_size']
        assert self.image_res % self.patch_size == 0
        self.num_patch = int(self.image_res / self.patch_size)
        self.min_perc_in_image = config['region']['min_perc_in_image']
    
    def get_bbox(self, ann):
        x, y, w, h = ann['bbox']
        return int(x), int(y), int(w), int(h)
    
    def __getitem__(self, index):
        ann = self.ann[index]
        
        # load image
        src_suffix = '/data/algceph/uasonchen/dataset'
        tgt_suffix = '/group/30042/uasonchen/datasets'
        try:
            image = Image.open(ann['image'].replace(src_suffix, tgt_suffix)).convert('RGB')
        except:
            print(ann['image'], "Not Found!!!")
            image = Image.new('RGB', (256, 256))
        
        W, H = image.size
        # random crop
        x, y, w, h = self.get_bbox(random.choice(ann['elems']))
        assert (x >= 0) and (y >= 0) and (x + w <= W) and (y + h <= H) and (w > 0) and (h > 0), "elem invalid"

        x0, y0 = random.randint(0, math.floor(x)), random.randint(0, math.floor(y))
        x1, y1 = random.randint(min(math.ceil(x + w), W), W), random.randint(min(math.ceil(y + h), H), H)
        w0, h0 = x1 - x0, y1 - y0
        assert (x0 >= 0) and (y0 >= 0) and (x0 + w0 <= W) and (y0 + h0 <= H) and (w0 > 0) and (h0 > 0), "elem randomcrop, invalid"

        image = image.crop((x0, y0, x0 + w0, y0 + h0))
        W, H = image.size
        
        do_hflip = False
        if random.random() < 0.5:
            image = hflip(image)
            do_hflip = True
                        
        image = resize(image, [256, 256], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        raw_caps = []
        caps = []
        rep_idx_list = []
        image_atts_list = []
        max_elems = self.max_regions
        
        if 'caption' in ann.keys():
            caption = random.choice(ann['caption']) if isinstance(ann['caption'], list) else ann['caption']

            raw_caption = pre_caption_keep_comma_period(caption)
            caption = pre_caption(caption, self.max_words)
            rep_idx = self.replace_func(raw_caption)
            
            raw_caps.append(raw_caption)
            caps.append(caption)
            rep_idx_list.append(rep_idx)
            image_atts_list.append([1] * (self.num_patch ** 2 + 1))
            max_elems -= 1
        
        elems = random.sample(ann['elems'], len(ann['elems']))
        for elem in elems:
            if max_elems <= 0:
                break

            x, y, w, h = self.get_bbox(elem)

            xx, yy = max(x0, x), max(y0, y)
            xm, ym = min(x0 + w0, x + w), min(y0 + h0, y + h)
            if (xm > xx) and (ym > yy):
                if (xm - xx) * (ym - yy) / (w * h) > self.min_perc_in_image:
                    x, y, w, h = xx, yy, xm - xx, ym - yy  # part inside the cropped image

                    # axis transform: after crop
                    x = x - x0
                    y = y - y0

                    if do_hflip:  # flipped applied
                        x = (W - x) - w  # W is w0

                    # resize applied
                    x = self.image_res / W * x
                    w = self.image_res / W * w
                    y = self.image_res / H * y
                    h = self.image_res / H * h

                    caption = random.choice(elem['caption']) if isinstance(elem['caption'], list) else elem['caption']

                    if 'attributes' in elem.keys():
                        elem_attr = random.choice(elem['attributes']) if isinstance(elem['attributes'], list) else elem['attributes']
                        caption = elem_attr + ' ' + caption
                    
                    raw_caption = pre_caption_keep_comma_period(caption)
                    caption = pre_caption(caption, self.max_words)
                    rep_idx = self.replace_func(raw_caption)
                    raw_caps.append(raw_caption)
                    caps.append(caption)
                    rep_idx_list.append(rep_idx)
                    
                    image_atts = self.get_image_attns(x, y, w, h)
                    image_atts_list.append(image_atts)
                    
                    max_elems -= 1
        
        image_list = [image] if len(caps) else []
        return image_list, raw_caps, caps, rep_idx_list, image_atts_list
    
    def get_image_attns(self, x, y, w, h):
        x_min = min(math.floor(x / self.patch_size), self.num_patch - 1)
        x_max = max(x_min+1, min(math.ceil((x+w) / self.patch_size), self.num_patch))  # exclude

        y_min = min(math.floor(y / self.patch_size), self.num_patch - 1)
        y_max = max(y_min+1, min(math.ceil((y+h) / self.patch_size), self.num_patch))  # exclude

        image_atts = [0] * (1 + self.num_patch ** 2)
        image_atts[0] = 1  # always include [CLS]
        for j in range(x_min, x_max):
            for i in range(y_min, y_max):
                index = self.num_patch * i + j + 1
                assert (index > 0) and (index <= self.num_patch ** 2), f"patch index out of range, index: {index}"
                image_atts[index] = 1

        return image_atts
    
    def collate_fn(self, batch_sample):
        batch = []
        for x in zip(*batch_sample):
            batch.append(x)

        images, batch = batch[0], batch[1:]

        idx_to_group_img = []
        img_idx = -1
        for sample in batch[0]:
            n_elems = len(sample)
            if n_elems > 0:
                img_idx += 1
                idx_to_group_img.extend([img_idx] * n_elems)  # flatten

        batch_size = self.batch_size
        n_elems = len(idx_to_group_img)
        to_keep = list(range(n_elems))
        if n_elems >= batch_size:
            to_keep = random.sample(to_keep, batch_size)
        else:
            # fixed batch_size is required. otherwise, the process will be blocked. so, i do pad here.
            # but pad causes wrong calculation for contrastive learning.
            # Set appropriate batch_size, max_images, and max_regions to avoid frequent padding.
            try:
                to_pad = random.sample(to_keep, batch_size - n_elems)
                to_keep += to_pad
                print("### warning: pad region_batch by sampling, ", len(to_pad), flush=True)

            except ValueError:
                print("### warning: pad region_batch by expanding, ", batch_size-len(to_keep), flush=True)
                to_keep = (to_keep * math.ceil(batch_size/len(to_keep)))[:batch_size]

        images = torch.stack(sum(images, []))  # flatten
        idx_to_group_img = torch.tensor([idx_to_group_img[index] for index in to_keep], dtype=torch.long)

        batch_tensors = [images, idx_to_group_img]
        for x in [sum(x, []) for x in batch]:

            x = [x[index] for index in to_keep]

            if x[0] is None:
                batch_tensors.append(None)
            elif isinstance(x[0], str):
                batch_tensors.append(x)
            elif isinstance(x[0], torch.Tensor):
                batch_tensors.append(torch.stack(x))
            else:
                batch_tensors.append(torch.tensor(x, dtype=torch.long))

        return batch_tensors