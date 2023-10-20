import torch
from functools import partial

from .image_encoder import ImageEncoderViT


vit_configs = dict(
    vit_h = dict(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31]
    ),
    vit_l = dict(   
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
    ),
    vit_b = dict(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11]
    )
)


def build_vit_h(checkpoint=None, use_sam_checkpoint=False):
    return _build_vit(
        **vit_configs["vit_h"],
        use_sam_checkpoint=use_sam_checkpoint,
        checkpoint=checkpoint,
    )


def build_vit_l(checkpoint=None, use_sam_checkpoint=False):
    return _build_vit(
        **vit_configs["vit_l"],
        use_sam_checkpoint=use_sam_checkpoint,
        checkpoint=checkpoint,
    )


def build_vit_b(checkpoint=None, use_sam_checkpoint=False):
    return _build_vit(
        **vit_configs["vit_b"],
        use_sam_checkpoint=use_sam_checkpoint,
        checkpoint=checkpoint,
    )


def _build_vit(
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
        checkpoint=None,
        use_sam_checkpoint=False
    ):
        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        
        vit = ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )
        if checkpoint is not None:
            weights = torch.load(checkpoint, map_location="cpu")
            if use_sam_checkpoint:
                    weights = {k[len("image_encoder."):]: v for k, v in weights.items() 
                            if k[:len("image_encoder")] == "image_encoder"}
            vit.load_state_dict(weights)
        return vit