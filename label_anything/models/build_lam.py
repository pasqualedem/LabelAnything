# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from label_anything.models.common import LayerNorm2d
from label_anything.models.common import SAM_EMBED_DIM

from . import (
    ImageEncoderViT,
    MaskDecoderLam,
    PromptImageEncoder,
    Lam,
    BinaryLam,
    OneWayTransformer,
    TwoWayTransformer,
    RandomMatrixEncoder,
)
from .build_vit import build_vit_b, build_vit_h, build_vit_l, build_vit_b_mae


def build_lam_vit_h(**kwargs):
    return _build_lam(
        build_vit_h,
        **kwargs,
    )


def build_lam_vit_l(**kwargs):
    return _build_lam(
        build_vit_l,
        **kwargs,
    )


def build_lam_vit_b(**kwargs):
    return _build_lam(
        build_vit_b,
        **kwargs,
    )
    
    
def build_lam_vit_mae_b(**kwargs):
    return _build_lam(
        build_vit_b_mae,
        **kwargs,
    )


def build_lam_no_vit(**kwargs):
    return _build_lam(
        build_vit=None,
        use_vit=False,
        **kwargs,
    )


def _build_lam(
    build_vit,
    checkpoint=None,
    use_sam_checkpoint=False,
    use_vit_sam_neck=True,
    use_vit=True,
    image_embed_dim=SAM_EMBED_DIM,
    embed_dim=SAM_EMBED_DIM,
    image_size=1024,
    vit_patch_size=16,
    class_attention=False,
    example_attention=False,
    example_class_attention=True,
    spatial_convs=None,
    encoder_attention_downsample_rate: int = 2,
    decoder_attention_downsample_rate: int = 2,
    classification_layer_downsample_rate: int = 8,
    use_broken_no_mask=False,
    use_background_embedding=False,
    fusion_transformer="TwoWayTransformer",
    class_encoder=None,
    segment_example_logits=False,
    dropout: float = 0.0,
    binary=False,
):

    image_embedding_size = image_size // vit_patch_size

    vit = build_vit(project_last_hidden=use_vit_sam_neck) if use_vit else None
    
    fusion_transformer = globals()[fusion_transformer](
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
                attention_downsample_rate=decoder_attention_downsample_rate,
            )
    if class_encoder is not None:
        cls = globals()[class_encoder['name']]
        params = {k: v for k, v in class_encoder.items() if k != 'name'}
        class_encoder = cls(**params)
    else:
        class_encoder = nn.Identity()
    
    neck = None if image_embed_dim == embed_dim else nn.Sequential(
            nn.Conv2d(
                image_embed_dim,
                embed_dim,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(embed_dim),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(embed_dim),
        )
    lam_class = BinaryLam if binary else Lam
    
    lam = lam_class(
        image_size=image_size,
        image_encoder=vit,
        neck=neck,
        prompt_encoder=PromptImageEncoder(
            embed_dim=embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
            class_attention=class_attention,
            example_attention=example_attention,
            example_class_attention=example_class_attention,
            dropout=dropout,
            use_broken_no_mask=use_broken_no_mask,
            use_background_embedding=use_background_embedding,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                attention_downsample_rate=encoder_attention_downsample_rate,
                num_heads=8,
                dropout=dropout,
            ),
            class_encoder=class_encoder,
        ),
        mask_decoder=MaskDecoderLam(
            transformer_dim=embed_dim,
            spatial_convs=spatial_convs,
            transformer=fusion_transformer,
            segment_example_logits=segment_example_logits,
            classification_layer_downsample_rate=classification_layer_downsample_rate,
            dropout=dropout,
        ),
    )
    lam.eval()
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)

        if use_sam_checkpoint:
            lam.init_pretrained_weights(state_dict)
        else:
            lam.load_state_dict(state_dict)
    return lam


build_lam = _build_lam
