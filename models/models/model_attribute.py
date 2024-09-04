'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from functools import partial
from models.vit_attribute import ViT, interpolate_pos_embed

import torch
import torch.nn.functional as F
from torch import nn

from loss.loss_multicls import AsymmetricLoss
import numpy as np
import random
import pdb

class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder=None,
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_deit=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.visual_encoder = ViT(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=len(tokenizer))

        if init_deit:
            checkpoint = torch.load('weights/deit_base_patch16_224-b5f2ef4d.pth')
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            # delete head.weight and head.bias
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print(msg)

        self.loss = AsymmetricLoss(gamma_neg=1, gamma_pos=0)
    def forward(self, image, label):
        logits = self.visual_encoder(image)
        loss = self.loss(logits, label)
        return {'loss': loss}