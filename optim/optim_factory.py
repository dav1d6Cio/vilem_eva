""" Optimizer Factory w/ Custom Weight Decay
Hacked together by / Copyright 2020 Ross Wightman
"""
import torch
from torch import optim as optim
import pdb

from .adafactor import Adafactor
from .adahessian import Adahessian
from .adamp import AdamP
from .lookahead import Lookahead
from .nadam import Nadam
from .novograd import NovoGrad
from .nvnovograd import NvNovoGrad
from .radam import RAdam
from .rmsprop_tf import RMSpropTF
from .sgdp import SGDP

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = True
except ImportError:
    has_apex = False


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def adjust_lr_weight_decay(model, weight_decay=1e-5, skip_list=(), coeff=0.3, lr=3e-4):
    decay = []
    no_decay = []
    bert_decay = []
    bert_no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'bert.' not in name:
                no_decay.append(param)
            else:
                bert_no_decay.append(param)
        else:
            if 'bert.' not in name:
                decay.append(param)
            else:
                bert_decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': bert_no_decay, 'weight_decay': 0., 'lr': lr*coeff},
        {'params': bert_decay, 'weight_decay': weight_decay, 'lr': lr*coeff},
    ]

def adjust_lr_weight_decay_clip(model, weight_decay=1e-5, skip_list=(), vit_coeff=1., coeff=0.3, lr=3e-4):
    decay = []
    no_decay = []
    bert_decay = []
    bert_no_decay = []
    vit_decay = []
    vit_no_decay = []
    if hasattr(model.visual_encoder, 'no_weight_decay'):
        vit_no_decay_name = {'visual_encoder.' + item for item in model.visual_encoder.no_weight_decay()}
        skip_list = set.union(skip_list, vit_no_decay_name)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            if 'bert.' in name:
                bert_no_decay.append(param)
            elif 'visual_encoder.' in name:
                vit_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if 'bert.' in name:
                bert_decay.append(param)
            elif 'visual_encoder.' in name:
                vit_decay.append(param)
            else:
                decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': bert_no_decay, 'weight_decay': 0., 'lr': lr*coeff},
        {'params': bert_decay, 'weight_decay': weight_decay, 'lr': lr*coeff},
        {'params': vit_no_decay, 'weight_decay': 0., 'lr': lr*vit_coeff},
        {'params': vit_decay, 'weight_decay': weight_decay, 'lr': lr*vit_coeff},
    ]

def adjust_lr_weight_decay_xvlm(model, wd=1e-5, skip_list=(), lr_mult=1, lr=1e-4):
    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult}
    ]

    no_decay = {"bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight"}

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("### model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            if n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
        else:  # decay
            if n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)
    return optimizer_grouped_parameters

def create_optimizer(args, model, filter_bias_and_bn=True):
    opt_lower = args.opt.lower()
    weight_decay = args.weight_decay

    if weight_decay and filter_bias_and_bn:
        skip = set()
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        # parameters = add_weight_decay(model, weight_decay, skip)
        if 'lr_mult' not in args.keys():
            args.lr_mult = 1.
        if 'use_xvlm' not in args.keys():
            if 'vit_lr_coeff' in args.keys():
                parameters = adjust_lr_weight_decay_clip(model, weight_decay, skip, args.vit_lr_coeff, args.lr_mult, args.lr)
            else:
                parameters = adjust_lr_weight_decay(model, weight_decay, skip, args.lr_mult, args.lr)
        else:
            parameters = adjust_lr_weight_decay_xvlm(model, weight_decay, skip, args.lr_mult, args.lr)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if 'fused' in opt_lower:
        assert has_apex and torch.cuda.is_available(), 'APEX and CUDA required for fused optimizers'

    opt_args = dict(lr=args.lr, weight_decay=weight_decay)
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        opt_args['eps'] = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas is not None:
        opt_args['betas'] = args.opt_betas
    if hasattr(args, 'opt_args') and args.opt_args is not None:
        opt_args.update(args.opt_args)

    opt_split = opt_lower.split('_')
    opt_lower = opt_split[-1]
    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'momentum':
        opt_args.pop('eps', None)
        optimizer = optim.SGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == 'nadam':
        optimizer = Nadam(parameters, **opt_args)
    elif opt_lower == 'radam':
        optimizer = RAdam(parameters, **opt_args)
    elif opt_lower == 'adamp':        
        optimizer = AdamP(parameters, wd_ratio=0.01, nesterov=True, **opt_args)
    elif opt_lower == 'sgdp':        
        optimizer = SGDP(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == 'adafactor':
        if not args.lr:
            opt_args['lr'] = None
        optimizer = Adafactor(parameters, **opt_args)
    elif opt_lower == 'adahessian':
        optimizer = Adahessian(parameters, **opt_args)
    elif opt_lower == 'rmsprop':
        optimizer = optim.RMSprop(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'rmsproptf':
        optimizer = RMSpropTF(parameters, alpha=0.9, momentum=args.momentum, **opt_args)
    elif opt_lower == 'novograd':
        optimizer = NovoGrad(parameters, **opt_args)
    elif opt_lower == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, **opt_args)
    elif opt_lower == 'fusedsgd':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=True, **opt_args)
    elif opt_lower == 'fusedmomentum':
        opt_args.pop('eps', None)
        optimizer = FusedSGD(parameters, momentum=args.momentum, nesterov=False, **opt_args)
    elif opt_lower == 'fusedadam':
        optimizer = FusedAdam(parameters, adam_w_mode=False, **opt_args)
    elif opt_lower == 'fusedadamw':
        optimizer = FusedAdam(parameters, adam_w_mode=True, **opt_args)
    elif opt_lower == 'fusedlamb':
        optimizer = FusedLAMB(parameters, **opt_args)
    elif opt_lower == 'fusednovograd':
        opt_args.setdefault('betas', (0.95, 0.98))
        optimizer = FusedNovoGrad(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)

    return optimizer
