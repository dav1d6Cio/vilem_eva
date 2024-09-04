import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.caption_dataset import re_train_dataset, re_eval_dataset, pretrain_dataset, pretrain_dataset_replace, \
    pretrain_dataset_two_augs, pretrain_dataset_phrase, re_phrase_train_dataset, re_phrase_eval_dataset, \
    re_train_dataset_two_augs, pretrain_dataset_attribute, re_attribute_eval_dataset, re_train_dataset_replace, \
    pretrain_dataset_tree, pretrain_dataset_beit, pretrain_dataset_cutmix, pretrain_dataset_cutmix_weak_aug, \
    re_train_dataset_cutmix, pretrain_data_multi_replace
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset
from dataset.video_dataset import re_eval_dataset as video_eval_dataset
from dataset.video_dataset import re_train_dataset as video_train_dataset
from dataset.xvlm_caption_dataset import pretrain_dataset_region

from dataset.randaugment import RandomAugment
from dataset.utils import GaussianBlur


def create_dataset(dataset, config, tokenizer=None, debug=False):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    pretrain_tcl_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    # jinyu: add augmentation
    train_tcl_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])

    video_train_tcl_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        normalize,
    ])

    box_transform = transforms.Compose([
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness']),
        transforms.ToTensor(),
        normalize,
    ])
    # 用CC3M預訓練，并用MSRVTT進行測試時使用
    video_train_norm_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0., saturation=0., hue=0.),
        transforms.ToTensor(),
        normalize,
    ])

    video_train_tensor_norm_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0., saturation=0., hue=0.),
        normalize,
    ])

    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.2, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    video_test_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        normalize,
    ])

    video_test_norm_transform = transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(288),
        transforms.Resize(256),
        normalize,
    ])

    common_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        #             transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_tcl_transform, debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_video_test':
        dataset = pretrain_dataset_replace(config['train_file'], video_train_norm_transform, config['rep_sampler'],
                                           debug=debug)
        test_dataset = video_eval_dataset(config['test_file'], video_test_norm_transform, config['image_root'],
                                          config['video_param'])
        return dataset, test_dataset

    if dataset == 'pretrain_cutmix':
        dataset = pretrain_dataset_cutmix(config['train_file'], pretrain_tcl_transform, debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_cutmix_weak':
        dataset = pretrain_dataset_cutmix_weak_aug(config['train_file'], pretrain_tcl_transform, debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_simple_aug':
        dataset = pretrain_dataset(config['train_file'], common_transform)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_tec':
        dataset = pretrain_dataset_replace(config['train_file'], pretrain_tcl_transform, config['rep_sampler'],
                                           debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset
    
    if dataset == 'pretrain_xvlm_tec':
        general_dataset = pretrain_dataset_replace(config['train_file'], pretrain_tcl_transform, config['rep_sampler'],
                                           debug=debug)
        region_dataset = pretrain_dataset_region(config['train_file_region'], box_transform, config, debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return general_dataset, region_dataset, test_dataset

    if dataset == 'pretrain_tec_debias':
        train_files = config['train_file']
        pre_train_files = []
        cc3m_idx = None
        # cc3m分了train和val两个文件，先预处理一下
        for i, file in enumerate(train_files):
            if 'cc3m' in file and cc3m_idx is None:
                pre_train_files.append([file])
                cc3m_idx = i
            elif 'cc3m' in file and cc3m_idx is not None:
                pre_train_files[i-1].append(file)
            else:   
                pre_train_files.append([file])
        datasets = []
        for file in pre_train_files:
            dataset = pretrain_dataset_replace(file, pretrain_tcl_transform, config['rep_sampler'])
            datasets.append(dataset)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return datasets, test_dataset

    if dataset == 'pretrain_multi_tec':
        dataset = pretrain_data_multi_replace(config['train_file'], pretrain_tcl_transform, config['rep_sampler'],
                                              debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_beit':
        dataset = pretrain_dataset_beit(config['train_file'], pretrain_tcl_transform, debug=debug)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_tree':
        dataset = pretrain_dataset_tree(config['train_file'], pretrain_tcl_transform)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_slip':
        dataset = pretrain_dataset_two_augs(config['train_file'], pretrain_transform)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_simclr':
        dataset = pretrain_dataset_two_augs(config['train_file'], pretrain_tcl_transform)
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return dataset, test_dataset

    if dataset == 'pretrain_phrase':
        dataset = pretrain_dataset_phrase(config['train_file'], pretrain_tcl_transform, tokenizer)
        test_dataset = re_phrase_eval_dataset(config['test_file'], test_transform, config['image_root'], tokenizer)
        return dataset, test_dataset

    if dataset == 'pretrain_attribute':
        dataset = pretrain_dataset_attribute(config['train_file'], pretrain_tcl_transform, tokenizer)
        test_dataset = re_attribute_eval_dataset(config['test_file'], test_transform, config['image_root'], tokenizer)
        return dataset, test_dataset

    elif dataset == 're':
        train_dataset = re_train_dataset(config['train_file'], train_tcl_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 're_tec':
        train_dataset = re_train_dataset_replace(config['train_file'], train_tcl_transform, config['image_root'],
                                                 config['rep_sampler'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'video_tec':
        train_dataset = video_train_dataset(config['train_file'], video_train_tensor_norm_transform,
                                            config['image_root'],
                                            config['rep_sampler'], config['video_param'], img_size=config['image_res'])
        val_dataset = video_eval_dataset(config['val_file'], video_test_transform, config['image_root'],
                                         config['video_param'], img_size=config['image_res'])
        test_dataset = video_eval_dataset(config['test_file'], video_test_transform, config['image_root'],
                                          config['video_param'], img_size=config['image_res'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 're_cutmix':
        train_dataset = re_train_dataset_cutmix(config['train_file'], train_tcl_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 're_phrase':
        train_dataset = re_phrase_train_dataset(config['train_file'], train_tcl_transform, config['image_root'],
                                                tokenizer)
        val_dataset = re_phrase_eval_dataset(config['val_file'], test_transform, config['image_root'], tokenizer)
        test_dataset = re_phrase_eval_dataset(config['test_file'], test_transform, config['image_root'], tokenizer)
        return train_dataset, val_dataset, test_dataset

    elif dataset == 're_two_augs':
        train_dataset = re_train_dataset_two_augs(config['train_file'], train_tcl_transform, config['image_root'])
        val_dataset = re_eval_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = re_eval_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'],
                                    split='train')
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'],
                                       split='test', answer_list=config['answer_list'])
        return train_dataset, vqa_test_dataset

    elif dataset == 'nlvr':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 've':
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset == 'grounding':
        train_transform = transforms.Compose([
            transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        return train_dataset, test_dataset


def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list, dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders