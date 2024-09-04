from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from transformers import BertTokenizer, BertTokenizerFast
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import random
import decord
from decord import VideoReader
import json
import os
from dataset.utils import pre_caption, pre_caption_keep_comma_period
import pdb

decord.bridge.set_bridge("torch")

class ImageNorm(object):
    """Apply Normalization to Image Pixels on GPU
    """
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        
    def __call__(self, img):

        if torch.max(img) > 1 and self.mean.max() <= 1:
            img.div_(255.)
        return img.sub_(self.mean).div_(self.std)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def _load_video_from_path_decord(video_path, num_frm, sample, fix_start=None):
    try:
        vr = VideoReader(video_path, num_threads=1)

        vlen = len(vr)
        frame_indices = sample_frames(num_frm, vlen, sample=sample, fix_start=fix_start)
        raw_sample_frms = vr.get_batch(frame_indices)
        raw_sample_frms = raw_sample_frms.float() / 255
    except Exception as e:
        return None

    raw_sample_frms = raw_sample_frms.permute(0, 3, 1, 2) # num_frm,H,W,3 ==> num_frm,3,H,W
    return raw_sample_frms
    
def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs
    
class re_eval_dataset(Dataset):

    def __init__(self, ann_file, transform, video_root, config, max_words=40, img_size=256):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''    
        self.video_root = video_root
        self.annotation = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.img_size = img_size
        
        self.num_frm = config['num_frm']
        self.frm_sampling_strategy = 'uniform'
        self.video_fmt = config['video_fmt']
        self.max_words = max_words
        self.tensor2img = transforms.ToPILImage()
        self.text = [pre_caption(ann['caption'], self.max_words) for ann in self.annotation]
        self.txt2img = [i for i in range(len(self.annotation))]
        self.img2txt = self.txt2img               
            
    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]  
        video_path = os.path.join(self.video_root, ann['video_id'] + self.video_fmt)
        try:
            vid_frm_array = _load_video_from_path_decord(video_path, self.num_frm, self.frm_sampling_strategy)
        except:
            print('bad video: {}'.format(video_path))
            video = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            video = transforms.ToTensor()(imgs).unsqueeze(0)
        video = self.transform(vid_frm_array)
        
        final = torch.zeros([self.num_frm, 3, self.img_size, self.img_size])
        final[:video.shape[0]] = video
        
        return final, ann['video_id']
    
class re_train_dataset(Dataset):

    def __init__(self, ann_file, transform, video_root, rep_config, video_config, max_words=40, img_size=256):
        '''
        image_root (string): Root directory of video
        ann_root (string): directory to store the annotation file
        '''
        self.video_root = video_root
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.tensor2img = transforms.ToPILImage()
        self.img_size = img_size
        
        # video extraction setting
        self.num_frm = video_config['num_frm']
        self.frm_sampling_strategy = video_config['frm_sampling_strategy']
        self.video_fmt = video_config['video_fmt']
        
        self.max_words = max_words
        self.video_ids = {}
        self.tokenizer = BertTokenizer.from_pretrained('weights/bert-base-uncased')
        self.stopwords = json.load(open('data/sw_spacy_choose_word_to_replace.json', 'r'))
        self.stopwords += [',', '.']
        
        if rep_config['func'] == 'mlm':
            self.replace_func = self.replace_mlm
            self.rep_prob = rep_config['prob']
            self.filter_sw = rep_config['filter_sw']
            self.max_rep_num = rep_config['max_rep_num']
            self.min_rep_num = rep_config['min_rep_num']
        else:
            self.replace_func = self.replace_k
            self.rep_num = rep_config['rep_num']
            self.filter_sw = rep_config['filter_sw']

        n = 0
        for ann in self.ann:
            video_id = ann['video_id']
            if video_id not in self.video_ids.keys():
                self.video_ids[video_id] = n
                n += 1             
            
    def __len__(self):
        return len(self.ann)
    
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

    def __getitem__(self, index):
        ann = self.ann[index]
        video_path = os.path.join(self.video_root, ann['video_id'] + self.video_fmt) 
        try:
            vid_frm_array = _load_video_from_path_decord(video_path, self.num_frm, self.frm_sampling_strategy)
        except:
            print('bad video: {}'.format(video_path))
            video = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            video = transforms.ToTensor()(imgs).unsqueeze(0)
        video = self.transform(vid_frm_array)
        
        final = torch.zeros([self.num_frm, 3, self.img_size, self.img_size])
        final[:video.shape[0]] = video
        
        raw_caption = pre_caption_keep_comma_period(ann['caption'])
        caption = pre_caption(ann['caption'], self.max_words)
        rep_idx = self.replace_func(raw_caption)
        
        return final, raw_caption, caption, rep_idx, self.video_ids[ann['video_id']]


