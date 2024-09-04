import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime
import sys
import traceback
import shutil
import random
import pdb
import re

import torch
import torch.distributed as dist
import torch.autograd as autograd

from PIL import ImageFilter

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, output_dir, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.output_dir = output_dir

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def update_from_dict(self, loss_dict):
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def add_meter_from_list(self, names, meters):
        if names[0] not in self.meters.keys():
            for i, name in enumerate(names):
                self.meters[name] = meters[i]

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                    with open('{}/log.txt'.format(self.output_dir), 'a') as f:
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB), file=f)
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def save_model_file(model, output_dir):
    file_path = '/'.join(model.split('.')[:-1]) + '.py'
    shutil.copy2(file_path, output_dir)


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def l2_normalize(a):
    """L2 normalization along the last dimension.

    Args:
        a: [..., C] tensor to normalize.

    Returns:
        A new tensor containing normalized rows.
    """
    norm = torch.norm(a, dim=-1, keepdim=True)
    return a / norm.clamp_min(1e-10)


def cos_pairwise(a, b=None):
    """Cosine between all pairs of entries in two tensors.

    Args:
        a: [*N, C] tensor, where ``*N`` can be any number of leading dimensions.
        b: [*M, C] tensor, where ``*M`` can be any number of leading dimensions.
            Defaults to ``a`` if missing.

    Returns:
        [*N, *M] tensor of cosine values.
    """
    a = l2_normalize(a)
    b = a if b is None else l2_normalize(b)
    N = a.shape[:-1]
    M = b.shape[:-1]
    a = a.flatten(end_dim=-2)
    b = b.flatten(end_dim=-2)
    cos = torch.einsum("nc,mc->nm", a, b)
    return cos.reshape(N + M)

@torch.no_grad()
def recompute_replace_pos_idx(right_text_ids, wrong_text_ids):
    return right_text_ids != wrong_text_ids

@torch.no_grad()
def construct_wrong_captions(generator, std_tokenizer, text, rep_pos_ids, device):
    text_input = std_tokenizer(text, padding=True, return_tensors="pt").to(device)
    # generate replace captions
    num_replace = rep_pos_ids.shape[1]
    rep_token_ids = text_input.input_ids.clone()
    with torch.no_grad():
        for rep_index in range(num_replace):
            rep_ids_i = rep_pos_ids[:, rep_index]
            rep_token_ids = generator(text_input, rep_ids_i, rep_token_ids)
            # 如果要考虑i-1次的replace，则使用下面的语句
            # rep_token_ids = generator(rep_token_ids, rep_ids_i)
            rep_token_ids = rep_token_ids.detach()
        # rep_token_ids的CLS可能会被换掉，因此这里要换上CLS
        rep_token_ids[:, 0] = std_tokenizer.cls_token_id
        # 去掉SEP
        rep_token_ids[rep_token_ids==std_tokenizer.sep_token_id] = std_tokenizer.pad_token_id
        # 去掉逗号和句号
        rep_token_ids = remove_comma_period(rep_token_ids, std_tokenizer)
    # wrong_text = batch_convert_ids_to_strings(rep_token_ids, std_tokenizer)
    # return wrong_text
    return rep_token_ids

def remove_comma_period(text_ids, tokenizer):
    result = torch.zeros((text_ids.shape[0], 30), device=text_ids.device).long()
    for i, ids in enumerate(text_ids):
        if tokenizer.convert_tokens_to_ids(',') in ids:
            ids = ids[ids!=tokenizer.convert_tokens_to_ids(',')]
        if tokenizer.convert_tokens_to_ids('.') in ids:
            ids = ids[ids != tokenizer.convert_tokens_to_ids('.')]
        ids = ids[:30]
        result[i, :len(ids)] = ids
    return result

import json
stopwords = json.load(open('data/sw_spacy_keep_preposition.json', 'r'))

def get_sw_mask(input_ids, tokenizer):
    sw_mask = torch.ones_like(input_ids)
    for i, ids in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for j, token in enumerate(tokens):
            if token in stopwords:
                sw_mask[i, j] = 0
    return sw_mask

@torch.no_grad()
def batch_convert_ids_to_strings(text_ids, tokenizer):
    strings = []
    for ids in text_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        strings.append(tokens_to_sentence(tokens))
    return strings

@torch.no_grad()
def tokens_to_sentence(tokens):
    clean_tokens = []
    for token in tokens:
        # 去掉逗号和句号
        if token in ['[CLS]', '[SEP]', '[PAD]', ',', '.']:
            continue

        if len(token) >= 2 and token[:2] == '##':
            clean_tokens[-1] += token[2:]
        else:
            clean_tokens.append(token)

    return ' '.join(clean_tokens)

def pc_backward(objectives, optimizer, primary=True):
    '''
    calculate the gradient of the parameters
    input:
    - objectives: a list of objectives
    '''

    grads, shapes, has_grads = _pack_grad(objectives, optimizer)
    with torch.no_grad():
        if primary:
            pc_grad = _project_conflicting_primary(grads, has_grads)
        else:
            pc_grad = _project_conflicting(grads, has_grads)
        pc_grad = _unflatten_grad(pc_grad, shapes[0])
        _set_grad(pc_grad, optimizer)
    return

import copy
def _project_conflicting(grads, has_grads, shapes=None, reduction='sum'):
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= (g_i_g_j) * g_j / (g_j.norm() ** 2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    if reduction == 'mean':
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
    elif reduction == 'sum':
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
    else:
        exit('invalid reduction method')

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)
    return merged_grad

def _project_conflicting_primary(grads, has_grads, shapes=None, reduction='sum'):
    # primary_grads is 0-index
    shared = torch.stack(has_grads).prod(0).bool()
    pc_grad, num_task = copy.deepcopy(grads), len(grads)
    g_0 = pc_grad[0]  # primary gradient
    for g_j in pc_grad[1:]:
        g_0_g_j = torch.dot(g_0, g_j)
        if g_0_g_j < 0:
            g_j -= (g_0_g_j) * g_0 / (g_0.norm() ** 2)
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    if reduction == 'mean':
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
    elif reduction == 'sum':
        merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
    else:
        exit('invalid reduction method')

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)
    return merged_grad

def _set_grad(grads, optimizer):
    '''
    set the modified gradients to the network
    '''

    idx = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            # if p.grad is None: continue
            p.grad = grads[idx]
            idx += 1
    return

def _pack_grad(objectives, optimizer):
    '''
    pack the gradient of the parameters of the network for each objective
    primary objective should be 0-index
    output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient
    '''
    grads, shapes, has_grads = [], [], []
    num_obj = len(objectives)
    for i, obj in enumerate(objectives):
        optimizer.zero_grad(set_to_none=True)
        if i == (num_obj-1):
            obj.backward(retain_graph=True)            
        else:
            obj.backward(retain_graph=True)
        with torch.no_grad():
            grad, shape, has_grad = _retrieve_grad(optimizer)
            grads.append(_flatten_grad(grad, shape))
            has_grads.append(_flatten_grad(has_grad, shape))
            shapes.append(shape)
        if i == (num_obj-1):
            # release the entire computation graph
            sum(objectives).backward()
    return grads, shapes, has_grads

def _unflatten_grad(grads, shapes):
    unflatten_grad, idx = [], 0
    for shape in shapes:
        length = np.prod(shape, dtype=int)
        unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
        idx += length
    return unflatten_grad

def _flatten_grad(grads, shapes):
    flatten_grad = torch.cat([g.flatten() for g in grads])
    return flatten_grad

def _retrieve_grad(optimizer):
    '''
    get the gradient of the parameters of the network with specific
    objective

    output:
    - grad: a list of the gradient of the parameters
    - shape: a list of the shape of the parameters
    - has_grad: a list of mask represent whether the parameter has gradient
    '''

    grad, shape, has_grad = [], [], []
    for group in optimizer.param_groups:
        for p in group['params']:
            # if p.grad is None: continue
            # tackle the multi-head scenario
            if p.grad is None:
                shape.append(p.shape)
                grad.append(torch.zeros_like(p).to(p.device))
                has_grad.append(torch.zeros_like(p).to(p.device))
                continue
            shape.append(p.grad.shape)
            grad.append(p.grad.clone())
            has_grad.append(torch.ones_like(p).to(p.device))
    return grad, shape, has_grad

def read_json(rpath):
    with open(rpath, 'r') as f:
        return json.load(f)
    
def calc_topk_accuracy(output, target, topk=(1,)):
    '''
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels,
    calculate top-k accuracies.
    '''
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.div_(torch.sum(target >= 0)))
    return res

def get_mixup_kwargs(args, mixup_kwargs):
    # Sample [alpha] for mixup strategy and [gamma] for coin flipping mixup
    alpha = args.random_seed.beta(args.mixup.beta, args.mixup.beta)
    image_alpha, text_alpha = alpha, alpha
    gamma = args.random_seed.random()

    # if gamma > 0.5, carrying out only image mixup
    # elif gamma <= 0.5, carrying out only text mixup
    if gamma > 0.5:
        mixup_kwargs['image_alpha'] = image_alpha
    else:
        mixup_kwargs['text_alpha'] = text_alpha
    return mixup_kwargs

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

@torch.no_grad()
def compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in parameters]), 2)
    return total_norm


@torch.no_grad()
def recompute_replace_pos_idx(right_text_ids, wrong_text_ids):
    return right_text_ids != wrong_text_ids


@torch.no_grad()
def construct_wrong_captions(generator, std_tokenizer, text, rep_pos_ids, device, inplace=False):
    text_input = std_tokenizer(text, padding=True, return_tensors="pt").to(device)
    # generate replace captions
    num_replace = rep_pos_ids.shape[1]
    rep_token_ids = text_input.input_ids.clone()
    with torch.no_grad():
        for rep_index in range(num_replace):
            rep_ids_i = rep_pos_ids[:, rep_index]
            if not inplace:
                rep_token_ids = generator(text_input.input_ids, text_input.attention_mask, rep_ids_i,
                                          rep_token_ids)  # .input_ids, text_input.attention_mask
            # 如果要考虑i-1次的replace，则使用下面的语句
            else:
                rep_token_ids = generator(rep_token_ids, text_input.attention_mask, rep_ids_i)
            rep_token_ids = rep_token_ids.detach()
        # rep_token_ids的CLS可能会被换掉，因此这里要换上CLS
        rep_token_ids[:, 0] = std_tokenizer.cls_token_id
        # 去掉SEP
        rep_token_ids[rep_token_ids == std_tokenizer.sep_token_id] = std_tokenizer.pad_token_id
        # 去掉逗号和句号
        rep_token_ids = remove_comma_period(rep_token_ids, std_tokenizer)
    # wrong_text = batch_convert_ids_to_strings(rep_token_ids, std_tokenizer)
    # return wrong_text
    return rep_token_ids


def remove_comma_period(text_ids, tokenizer):
    result = torch.zeros((text_ids.shape[0], 30), device=text_ids.device).long()
    for i, ids in enumerate(text_ids):
        if tokenizer.convert_tokens_to_ids(',') in ids:
            ids = ids[ids != tokenizer.convert_tokens_to_ids(',')]
        if tokenizer.convert_tokens_to_ids('.') in ids:
            ids = ids[ids != tokenizer.convert_tokens_to_ids('.')]
        ids = ids[:30]
        result[i, :len(ids)] = ids
    return result


def get_sw_mask(input_ids, tokenizer):
    sw_mask = torch.ones_like(input_ids)
    for i, ids in enumerate(input_ids):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        for j, token in enumerate(tokens):
            if token in stopwords:
                sw_mask[i, j] = 0
    return sw_mask


@torch.no_grad()
def batch_convert_ids_to_strings(text_ids, tokenizer):
    strings = []
    for ids in text_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids)
        strings.append(tokens_to_sentence(tokens))
    return strings


@torch.no_grad()
def tokens_to_sentence(tokens):
    clean_tokens = []
    for token in tokens:
        # 去掉逗号和句号
        if token in ['[CLS]', '[SEP]', '[PAD]', ',', '.']:
            continue

        if len(token) >= 2 and token[:2] == '##' and len(clean_tokens) > 0:
            clean_tokens[-1] += token[2:]
        else:
            clean_tokens.append(token)

    return ' '.join(clean_tokens)

def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False

def parse_opts(opts):
    if opts is None:
        opts = []

    if len(opts) == 0:
        return opts

    has_equal = opts[0].find("=") != -1

    if has_equal:
        return opts

    return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]

def merge_config(config, opts):
    opts = parse_opts(opts)
    for opt in opts:
        k, v = opt.split("=")
        if v.lower() == "true":
            config[k] = True
        elif v.lower() == "false":
            config[k] = False
        elif is_number(v):
            config[k] = int(v) if '.' not in v and 'e' not in v else float(v)
        else:
            config[k] = v
    return config

def save_model_file(model, output_dir):
    file_path = '/'.join(model.split('.')[:-1]) + '.py'
    shutil.copy2(file_path, output_dir)

def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url