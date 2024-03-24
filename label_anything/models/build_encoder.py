import copy
from functools import partial

import torch
from einops import rearrange
from transformers import AutoModel, ViTModel

from .image_encoder import ImageEncoderViT

vit_configs = dict(
    vit_h=dict(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
    ),
    vit_l=dict(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
    ),
    vit_b=dict(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    ),
)


def build_vit_h(**kwargs):
    return _build_vit(**vit_configs["vit_h"], **kwargs)


def build_vit_l(**kwargs):
    return _build_vit(**vit_configs["vit_l"], **kwargs)


def build_vit_b(**kwargs):
    return _build_vit(**vit_configs["vit_b"], **kwargs)


def _build_vit(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
    use_sam_checkpoint=False,
    project_last_hidden=True,
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
        project_last_hidden=project_last_hidden,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    if checkpoint is not None:
        weights = torch.load(checkpoint, map_location="cpu")
        if use_sam_checkpoint:
            weights = {
                k[len("image_encoder.") :]: v
                for k, v in weights.items()
                if k[: len("image_encoder")] == "image_encoder"
            }
        vit.load_state_dict(weights)
    return vit


class ViTModelWrapper(ViTModel):
    def forward(self, x):
        h, w = x.shape[-2:]
        output = super().forward(x, interpolate_pos_encoding=True)
        hs = output.last_hidden_state[:, 1:, :]
        out = rearrange(hs, "b (h w) c -> b c h w", h=h // 16).contiguous()
        return out


def delete_encoder_layers(
    model: ViTModelWrapper | ImageEncoderViT, num_layers_to_keep: int
):  
    assert num_layers_to_keep > 0
    # must pass in the full bert model
    if isinstance(model, ViTModelWrapper):
        old_module_list = model.encoder.layer
        new_module_list = torch.nn.ModuleList()

        # Now iterate over all layers, only keepign only the relevant layers.
        for i in range(0, len(num_layers_to_keep)):
            new_module_list.append(old_module_list[i])

        # create a copy of the model, modify it with the new list, and return
        copy_of_model = copy.deepcopy(model)
        copy_of_model.encoder.layer = new_module_list
        del model

        return copy_of_model
    elif isinstance(model, ImageEncoderViT):
        model.blocks = model.blocks[:num_layers_to_keep]
        return model


def build_vit_b_mae(project_last_hidden=False):
    vit_mae = ViTModelWrapper.from_pretrained("facebook/vit-mae-base")
    return vit_mae


def build_encoder(name, **kwargs):
    if name in ENCODERS:
        return ENCODERS[name](**kwargs)
    return AutoModel.from_pretrained(name)


ENCODERS = {
    "vit_h": build_vit_h,
    "vit_l": build_vit_l,
    "vit_b": build_vit_b,
    "vit_b_mae": build_vit_b_mae,
}
