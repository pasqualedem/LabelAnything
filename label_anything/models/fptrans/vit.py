"""Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in:
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929
`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270
The official jax code is released and available at https://github.com/google-research/vision_transformer
DeiT model defs and weights from https://github.com/facebookresearch/deit,
paper `DeiT: Data-efficient Image Transformers` - https://arxiv.org/abs/2012.12877
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2021 Ross Wightman

# ===========================================================================================
# Feature-Proxy Transformer (FPTrans) in PyTorch
#
# This code file is copied and modified from
#     https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
# for the development of FPTrans.
#
# By Jian-Wei Zhang (zjw.math@qq.com).
# 2022-09
#
# ===========================================================================================

"""
import math
import logging
from functools import partial
from collections import OrderedDict
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from label_anything.models.fptrans import vit_utils as utils

_logger = logging.getLogger(name=Path(__file__).parents[1].stem)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = utils.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = utils.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 pretrained="",
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=utils.PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='',
                 opt=None, logger=None, original=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.opt = opt
        self.logger = logger
        self.allow_mod = not original
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # Patch embedding
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride=opt.vit_stride)
        num_patches = self.patch_embed.num_patches

        # prompt tokens
        self.prompt_tokens = None
        if self.allow_mod:
            ncls = 15 if opt.dataset == "PASCAL" else 60
            if opt.coco2pascal:
                ncls = 60
            divider = 1 + opt.bg_num * opt.shot
            self.prompt_tokens = nn.Parameter(
                torch.zeros(ncls * divider, opt.num_prompt // divider, embed_dim))  # [bank_size, G, embed_dim]
            nn.init.normal_(self.prompt_tokens.permute(2, 0, 1), std=opt.pt_std)
            self.sampler = np.random.RandomState(1234)

        # Class token, Distillation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        if pretrained == "":
            self.init_weights(weight_init)
        else:
            if str(pretrained).endswith('.pth'):
                _load_weights_pth(logger, self, pretrained)
            elif str(pretrained).endswith('.npz'):
                _load_weights_npz(self, pretrained)
            else:
                raise ValueError(f'Not recognized file {pretrained}. [.pth|.npz]')

            if logger is not None:
                logger.info(' ' * 5 + f'==> {opt.backbone} initialized from {pretrained}')

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        utils.trunc_normal_(self.pos_embed, std=.02)
        if self.dist_token is not None:
            utils.trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            utils.named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            utils.trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def forward_original(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, N ,C]
        cls_token = expand_to_batch(self.cls_token, x.shape[0])  # [B, 1, C]
        if self.dist_token is None:
            x = cat_token(cls_token, x)     # [B, N+1, C]
        else:
            dist_token = expand_to_batch(self.dist_token, x.shape[0])  # [B, 1, C]
            x = cat_token(cls_token, dist_token, x)     # [B, N+2, C]
        x = x + self.pos_embed  # [B, N + 1, C]
        x = self.pos_drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
        x = self.norm(x)

        x = x[:, self.num_tokens:, :]
        b, n, c = x.shape
        hh = int(math.sqrt(n))
        x = x.view(b, hh, hh, c).permute(0, 3, 1, 2).contiguous()

        return dict(out=x)

    def forward(self, x):
        if not self.allow_mod:
            return self.forward_original(x)

        # x: [B, C, H, W]
        # fg_token: [B, 1, C]
        # bg_token: [B, k, C]
        x, (fg_token, bg_token), shot = x
        bank_size, G, embed_dim = self.prompt_tokens.shape
        B = x.shape[0] // (1 + shot)
        divider = 1 + self.opt.bg_num * shot
        prompts = self.prompt_tokens[self.sampler.choice(bank_size, size=B * divider, replace=False)] \
            .reshape(B, divider * G, embed_dim)    # [B, (1+bg_num*shot)*G, embed_dim]
        try:
            tokens = {
                'fg': prompts[:, :G] + fg_token,
                'bg': prompts[:, G:] + bg_token.unsqueeze(2).expand(-1, -1, G, -1).reshape(
                    B, (divider - 1) * G, embed_dim)
            }
        except:
            print("ciao")

        x = self.patch_embed(x)  # [B, N ,C]
        cls_token = expand_to_batch(self.cls_token, x.shape[0])  # [B, 1, C]
        if self.dist_token is None:
            x = cat_token(cls_token, x)     # [B, N+1, C]
        else:
            dist_token = expand_to_batch(self.dist_token, x.shape[0])  # [B, 1, C]
            x = cat_token(cls_token, dist_token, x)     # [B, N+2, C]
        x = x + self.pos_embed  # [B, N + 1, C']
        x = self.pos_drop(x)

        S = shot
        B = x.shape[0] // (S + 1)

        # Concat image and tokens
        _, N, C = x.shape
        n1 = tokens['fg'].shape[1]
        n2 = tokens['bg'].shape[1]
        fg_token = tokens['fg'].view(B, 1, n1, C).expand(-1, S + 1, -1, -1).reshape(B*(S+1), n1, C)
        bg_token = tokens['bg'].view(B, 1, n2, C).expand(-1, S + 1, -1, -1).reshape(B*(S+1), n2, C)
        x = cat_token(x, fg_token, bg_token)

        # Forward transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.reduce_and_expand(x, num=S + 1, start=self.num_tokens, end=n1 + n2)

        # Split outputs
        x, fg_token, bg_token = x.split([N, n1, n2], dim=1)
        tokens['fg'] = fg_token.view(B, S + 1, n1, C)[:, 0]
        tokens['bg'] = bg_token.view(B, S + 1, n2, C)[:, 0]

        x = self.norm(x)

        x = x[:, self.num_tokens:, :]
        b, n, c = x.shape
        hh = int(math.sqrt(n))
        x = x.view(b, hh, hh, c).permute(0, 3, 1, 2).contiguous()

        tokens['fg'] = tokens['fg'].mean(1)[:, :, None, None]    # [B, C, 1, 1]
        if self.opt.bg_num == 1:
            tokens['bg'] = tokens['bg'].mean(1)[:, :, None, None]    # [B, C, 1, 1]
        else:
            _B, _N, _C = tokens['bg'].shape
            bg_num = self.opt.bg_num
            tokens['bg'] = tokens['bg'].reshape(
                _B * bg_num, _N // bg_num, _C).mean(1)[:, :, None, None]    # [Bk, C, 1, 1]

        return dict(out=x, tokens=tokens)

    @staticmethod
    def reduce_and_expand(x, num, start=1, end=0):
        B, N, C = x.shape
        x = x.view(B // num, num, N, C)
        if start > 0:
            mean_token = x[:, :, :start, :].mean(1, keepdim=True).expand(-1, num, -1, -1)
            x[:, :, :start, :] = mean_token
        if end > 0:
            mean_token = x[:, :, -end:, :].mean(1, keepdim=True).expand(-1, num, -1, -1)
            x[:, :, -end:, :] = mean_token
        return x.view(B, N, C)


def expand_to_batch(prompt, batch_size, stack=False):
    if stack:
        prompt = prompt.unsqueeze(0)
    return prompt.expand(batch_size, *[-1 for _ in prompt.shape[1:]])

def cat_token(*args):
    return torch.cat(args, dim=1)

def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            utils.lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                utils.trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        utils.lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights_npz(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(utils.adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = utils.adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
        model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
        model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


@torch.no_grad()
def _load_weights_pth(logger, model: VisionTransformer, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')['model']
    state_dict = model.state_dict()
    if ckpt['pos_embed'].shape != state_dict['pos_embed'].shape:
        ckpt['pos_embed'] = resize_pos_embed(
            ckpt['pos_embed'], state_dict['pos_embed'], model.num_tokens, model.patch_embed.grid_size)

    counter = 0
    for k in state_dict.keys():
        if k in ckpt:
            state_dict[k] = ckpt[k]
            counter += 1

    logger.info(' ' * 5 + f"==> {counter} parameters loaded.")
    model.load_state_dict(state_dict, strict=True)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info(' ' * 5 + '==> Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(' ' * 5 + '==> Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


vit_factory = {
    'ViT-Ti/16':         {'patch_size': 16, 'embed_dim':  192, 'depth': 12, 'num_heads':  3, 'distilled': False},
    'ViT-S/32':          {'patch_size': 32, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-S/16':          {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-S/16-i21k':     {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads':  6, 'distilled': False},
    'ViT-B/32':          {'patch_size': 32, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16':          {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-384':      {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-i21k':     {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/16-i21k-384': {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-B/8':           {'patch_size':  8, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': False},
    'ViT-L/32':          {'patch_size': 32, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},
    'ViT-L/16':          {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},
    'ViT-L/16-384':      {'patch_size': 16, 'embed_dim': 1024, 'depth': 24, 'num_heads': 16, 'distilled': False},

    'DeiT-T/16':         {'patch_size': 16, 'embed_dim':  192, 'depth': 12, 'num_heads': 3, 'distilled': True},
    'DeiT-S/16':         {'patch_size': 16, 'embed_dim':  384, 'depth': 12, 'num_heads': 6, 'distilled': True},
    'DeiT-B/16':         {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': True},
    'DeiT-B/16-384':     {'patch_size': 16, 'embed_dim':  768, 'depth': 12, 'num_heads': 12, 'distilled': True},
}


def vit_model(model_type,
              image_size,
              pretrained="",
              init_channels=3,
              num_classes=1000,
              opt=None,
              logger=None,
              original=False,
              depth=None):
    return VisionTransformer(img_size=image_size,
                             patch_size=vit_factory[model_type]['patch_size'],
                             in_chans=init_channels,
                             num_classes=num_classes,
                             embed_dim=vit_factory[model_type]['embed_dim'],
                             depth=depth or opt.vit_depth or vit_factory[model_type]['depth'],
                             num_heads=vit_factory[model_type]['num_heads'],
                             pretrained=pretrained,
                             distilled=vit_factory[model_type]['distilled'],
                             opt=opt,
                             logger=logger,
                             original=original)


