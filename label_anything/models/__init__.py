# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple

from .sam import Sam, AdaptedSam
from .lam import Lam, BinaryLam
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder, MaskDecoderLam
from .prompt_encoder import PromptEncoder, PromptImageEncoder, RandomMatrixEncoder
from .transformer import IdentityTransformer, OneWayTransformer, TwoWayTransformer
from .build_sam import build_sam_vit_b, build_sam_vit_h, build_sam_vit_l, build_asam_vit_b
from .build_lam import build_lam_vit_b, build_lam_vit_h, build_lam_vit_l, build_lam, build_lam_no_vit, build_lam_vit_mae_b, build_multilevel_lam, build_lam_vit_b_imagenet_i21k, build_lam_dino_b8, LabelAnything, LabelAnythingConfig
from .build_encoder import ENCODERS, build_vit_b, build_vit_h, build_vit_l
from .samfew import SAMFewShotModel
from .dcama import build_dcama
from .fptrans import build_fptrans
from .panet import build_panet
from .ppnet import build_ppnet
# from .bam import build_bam
# from.hdmnet import build_hdmnet
from .denet import build_denet
from .dummy import build_dummy
from .similarity import build_similarity


ComposedOutput = namedtuple("ComposedOutput", ["main", "aux"])

model_registry = {
    "lam": build_lam,
    "lam_no_vit": build_lam_no_vit,
    "lam_h": build_lam_vit_h,
    "lam_l": build_lam_vit_l,
    "lam_b": build_lam_vit_b,
    "lam_mae_b": build_lam_vit_mae_b,
    "lam_dino_b8": build_lam_dino_b8,
    "lam_b_imagenet_i21k": build_lam_vit_b_imagenet_i21k,
    "multilevel_lam": build_multilevel_lam,
    "sam": build_sam_vit_h,
    "sam_h": build_sam_vit_h,
    "sam_l": build_sam_vit_l,
    "sam_b": build_sam_vit_b,
    "asam_b": build_asam_vit_b,
    "dcama": build_dcama,
    "fptrans": build_fptrans,
    "panet": build_panet,
    "ppnet": build_ppnet,
    "denet": build_denet,
    # "hdmnet": build_hdmnet,
    # "bam": build_bam,
    # "hdmnet": build_hdmnet,
    "dummy": build_dummy,
    "similarity": build_similarity,
    # Encoders only
    **ENCODERS
}


def build_samfew(
    sam_model="vit_b",
    sam_params=None,
    fewshot_model="dcama",
    fewshot_params=None,
    custom_preprocess=True,
):
    sam = model_registry[sam_model](**sam_params)
    fewshot = model_registry[fewshot_model](**fewshot_params)
    return SAMFewShotModel(sam, fewshot)


model_registry["samfew"] = build_samfew