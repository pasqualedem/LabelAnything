from einops import rearrange
import torch
from functools import partial

from transformers import ViTModel, AutoModel, AutoBackbone

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
    def forward(self, pixel_values):
        """We edit the forward method to return the last hidden state of the model

        Differently from HF implementation, we preserve spatial information of the patches
        and remove the [CLS] token from the output.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Last hidden state of the model
        """
        h, w = pixel_values.shape[-2:]
        output = super().forward(pixel_values, interpolate_pos_encoding=True)
        hs = output.last_hidden_state[:, 1:, :]
        out = rearrange(hs, "b (h w) c -> b c h w", h=h // 16).contiguous()
        return out


def build_vit_b_mae(project_last_hidden=False):
    vit_mae = ViTModelWrapper.from_pretrained("facebook/vit-mae-base")
    return vit_mae


def build_vit_b_imagenet_i21k(project_last_hidden=False):
    vit_b_imagenet_i21k = ViTModelWrapper.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )
    return vit_b_imagenet_i21k


def build_vit_dino_b8(project_last_hidden=False):
    vit_dino_b8 = ViTModelWrapper.from_pretrained("facebook/dino-vitb8")
    return vit_dino_b8


def build_resnet50(
    project_last_hidden=False, out_features=["stage2", "stage3", "stage4"]
):
    resnet50 = AutoBackbone.from_pretrained(
        "microsoft/resnet-50", out_features=out_features
    )
    return resnet50


def build_swin_b(
    project_last_hidden=False, out_features=["stage2", "stage3", "stage4"]
):
    swin_b = AutoBackbone.from_pretrained(
        "microsoft/swin-base-patch4-window12-384", out_features=out_features
    )
    return swin_b


def build_encoder(name, **kwargs):
    if name in ENCODERS:
        return ENCODERS[name](**kwargs)
    return AutoModel.from_pretrained(name)


ENCODERS = {
    "vit_h": build_vit_h,
    "vit_l": build_vit_l,
    "vit_b": build_vit_b,
    "vit_b_mae": build_vit_b_mae,
    "vit_dino_b8": build_vit_dino_b8,
    "resnet50": build_resnet50,
    "swin_b": build_swin_b,
}
