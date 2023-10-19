# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from . import ImageEncoderViT, MaskDecoderLam, PromptImageEncoder, Lam, TwoWayTransformer
from .build_vit import build_vit_b, build_vit_h, build_vit_l

def build_lam_vit_h(checkpoint=None):
    return _build_lam(
        build_vit_h,
        checkpoint=checkpoint,
    )


build_sam = build_lam_vit_h


def build_lam_vit_l(checkpoint=None):
    return _build_lam(
        build_vit_l,
        checkpoint=checkpoint,
    )


def build_lam_vit_b(checkpoint=None):
    return _build_lam(
        build_vit_b,
        checkpoint=checkpoint,
    )



def build_lam_no_vit(checkpoint=None):
    return _build_lam(
        encoder_embed_dim=None,
        encoder_depth=None,
        encoder_num_heads=None,
        encoder_global_attn_indexes=None,
        checkpoint=checkpoint,
        use_vit=False,
    )


def _build_lam(
    build_vit,
    checkpoint=None,
    use_vit=True,
):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    vit = build_vit() if use_vit else None
    sam = Lam(
        image_encoder=vit,
        prompt_encoder=PromptImageEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
        ),
        mask_decoder=MaskDecoderLam(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
        ),
    )
    sam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(state_dict)
    return sam