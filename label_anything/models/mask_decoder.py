# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from einops import rearrange
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from einops import rearrange, repeat

from label_anything.utils.utils import ResultDict

from .common import LayerNorm2d, MLPBlock


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class MaskDecoderLam(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        spatial_convs=None,
        activation: Type[nn.Module] = nn.GELU,
        segment_example_logits: bool = False,
        classification_layer_downsample_rate: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          trasnformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.attention_dim = transformer_dim
        self.segment_example_logits = segment_example_logits

        first_layer_downsample_rate = (
            classification_layer_downsample_rate // 2
            if classification_layer_downsample_rate > 1
            else 1
        )

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim,
                transformer_dim // first_layer_downsample_rate,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(transformer_dim // first_layer_downsample_rate),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // first_layer_downsample_rate,
                transformer_dim // classification_layer_downsample_rate,
                kernel_size=2,
                stride=2,
            ),
        )
        self.transformer = transformer
        self.spatial_convs = None
        if spatial_convs is not None:
            module_list = []
            for i in range(spatial_convs):
                module_list.append(
                    nn.Conv2d(
                        transformer_dim // classification_layer_downsample_rate,
                        transformer_dim // classification_layer_downsample_rate,
                        kernel_size=3,
                        padding=1,
                    )
                )
                if i < spatial_convs - 1:
                    module_list.append(
                        LayerNorm2d(
                            transformer_dim // classification_layer_downsample_rate
                        )
                    )
                    module_list.append(activation())
            self.spatial_convs = nn.Sequential(*module_list)
        self.class_mlp = MLP(
            transformer_dim,
            transformer_dim,
            transformer_dim // classification_layer_downsample_rate,
            3,
            dropout=dropout,
        )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_embeddings: torch.Tensor,
        flag_examples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          class_embeddings ResultDict: the embeddings of each example and class

        Returns:
          torch.Tensor: batched predicted segmentations
        """
        b, d, h, w = query_embeddings.shape
        _, c, _ = class_embeddings[ResultDict.CLASS_EMBS].shape
        if self.segment_example_logits:
            class_embeddings = rearrange(
                class_embeddings[ResultDict.EXAMPLES_CLASS_EMBS], "b n c d -> b (n c) d"
            )
        else:
            class_embeddings = class_embeddings[ResultDict.CLASS_EMBS]
        class_embeddings, query_embeddings = self.transformer(
            query_embeddings, image_pe, class_embeddings
        )
        query_embeddings = rearrange(query_embeddings, "b (h w) c -> b c h w", h=h)

        upscaled_embeddings = self.output_upscaling(query_embeddings)
        if self.spatial_convs is not None:
            upscaled_embeddings = self.spatial_convs(upscaled_embeddings)
        b, d, h, w = upscaled_embeddings.shape

        class_embeddings = self.class_mlp(class_embeddings)
        seg = (class_embeddings @ upscaled_embeddings.view(b, d, h * w)).view(
            b, -1, h, w
        )
        if self.segment_example_logits:
            seg = rearrange(seg, "b (n c) h w -> b n c h w", c=c)
            seg = seg.max(dim=1).values
        return seg


class AffinityDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        spatial_convs=None,
        activation: Type[nn.Module] = nn.GELU,
        classification_layer_downsample_rate: int = 8,
        transformer_feature_size: int = None,
        class_fusion: str = "sum",
        prototype_merge: bool = False,
        transformer_keys_are_images: bool = True,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          trasnformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
        """
        super().__init__()
        self.attention_dim = transformer_dim
        self.transformer_feature_size = None
        self.class_fusion = class_fusion
        self.transformer_keys_are_images = transformer_keys_are_images
        if transformer_feature_size is not None:
            self.transformer_feature_size = (
                transformer_feature_size,
                transformer_feature_size,
            )

        first_layer_depth = transformer_dim // (
            classification_layer_downsample_rate * 4
        )
        second_layer_depth = transformer_dim // (
            classification_layer_downsample_rate * 2
        )
        third_layer_depth = transformer_dim // classification_layer_downsample_rate

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim,
                first_layer_depth,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(first_layer_depth),
            activation(),
            nn.ConvTranspose2d(
                first_layer_depth,
                second_layer_depth,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(second_layer_depth),
            activation(),
            nn.ConvTranspose2d(
                second_layer_depth,
                third_layer_depth,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(third_layer_depth),
            activation(),
            nn.Conv2d(
                third_layer_depth,
                1,
                kernel_size=1,
            ),
        )
        self.class_embedding_mlp = None
        if prototype_merge:
            self.class_embedding_mlp = MLP(
                transformer_dim,
                transformer_dim,
                third_layer_depth,
                3,
                dropout=0.0,
            )
        self.transformer = transformer
        self.transformer_feature_size = transformer_feature_size
        self.spatial_convs = None
        if spatial_convs is not None:
            module_list = []
            for i in range(spatial_convs):
                module_list.append(
                    nn.Conv2d(
                        transformer_dim,
                        transformer_dim,
                        kernel_size=3,
                        padding=1,
                    )
                )
                if i < spatial_convs - 1:
                    module_list.append(LayerNorm2d(transformer_dim))
                    module_list.append(activation())
            self.spatial_convs = nn.Sequential(*module_list)

    def rescale_for_transformer(self, query, support=None, mask=None, size=None):
        if self.transformer_feature_size is not None:
            size = size if size is not None else self.transformer_feature_size
            query = F.interpolate(query, size=size, mode="bilinear")
            if support is not None:
                b, n, d, h, w = support.shape
                support = rearrange(support, "b n d h w -> (b n) d h w")
                support = F.interpolate(support, size=size, mode="bilinear")
                support = rearrange(support, "(b n) d h w -> b n d h w", b=b)
            if mask is not None:
                b, n, c, d, h, w = mask.shape
                mask = rearrange(mask, "b n c d h w -> (b n c) d h w")
                mask = F.interpolate(mask, size=size, mode="bilinear")
                mask = rearrange(mask, "(b n c) d h w -> b n c d h w", b=b, n=n)
        return query, support, mask

    def _apply_classes_to_features(self, features, classes):
        if self.class_fusion == "sum":
            classes = rearrange(classes, "b n c d -> b n c d () ()")
            return features + classes
        elif self.class_fusion == "mul":
            classes = rearrange(classes, "b n c d -> b n c d () ()")
            return features * classes
        elif self.class_fusion == "softmax":
            m = classes.shape[1]
            classes = rearrange(classes, "b m c d -> b (m c) d")
            classes = F.softmax(classes, dim=1)
            classes = rearrange(classes, "b (m c) d -> b m c d () ()", m=m)
            return features * classes
        elif self.class_fusion == "sigmoid":
            classes = F.sigmoid(classes)
            classes = rearrange(classes, "b n c d -> b n c d () ()")
            return features * classes

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        class_embeddings: torch.Tensor,
        flag_examples: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          class_embeddings ResultDict: the embeddings of each example and class

        Returns:
          torch.Tensor: batched predicted segmentations
        """
        b, n, d, h, w = support_embeddings.shape

        support_masks = class_embeddings[ResultDict.EXAMPLES_CLASS_SRC]  # (b n c) h w
        support_masks = rearrange(
            support_masks, "(b n c) d (h w) -> b n c d h w", b=b, n=n, h=h
        )
        c = support_masks.shape[2]
        class_examples_embeddings = class_embeddings[
            ResultDict.EXAMPLES_CLASS_EMBS
        ]  # b n c d
        support_masks = self._apply_classes_to_features(
            support_masks, class_examples_embeddings
        )
        if not self.transformer_keys_are_images:
            support_embeddings = None

        cur_feature_size = query_embeddings.shape[-2:]
        original_query_embeddings = query_embeddings.clone()
        query_embeddings, support_embeddings, support_masks = (
            self.rescale_for_transformer(
                query_embeddings, support_embeddings, support_masks
            )
        )
        query_embeddings = repeat(query_embeddings, "b d h w -> (b c) (h w) d", c=c)
        support_masks = rearrange(support_masks, "b n c d h w -> (b c) (n h w) d")
        if support_embeddings is not None:
            support_embeddings = repeat(
                support_embeddings, "b n d h w -> (b c) (n h w) d", c=c
            )
        else:
            support_embeddings = support_masks

        # Remove padding classes
        batch_mask = rearrange(flag_examples, "b n c -> (b c) n").any(dim=-1)
        query_embeddings = query_embeddings[batch_mask]
        support_embeddings = support_embeddings[batch_mask]
        support_masks = support_masks[batch_mask]

        query_embeddings = self.transformer(
            query_embeddings,
            support_embeddings,
            support_masks,
            image_pe,
            flag_examples,
            batch_mask,
        )
        query_embeddings = rearrange(query_embeddings, "bc (h w) d -> bc d h w", h=h)
        query_embeddings = self.rescale_for_transformer(
            query_embeddings, None, None, size=cur_feature_size
        )[0]

        if self.spatial_convs is not None:
            query_embeddings = self.spatial_convs(query_embeddings)

        # collapse the depth dimension
        if self.class_embedding_mlp is not None:
            prototypes = class_embeddings[ResultDict.CLASS_EMBS]
            prototypes = self.class_embedding_mlp(prototypes)
            for i in range(len(self.output_upscaling) - 1):
                original_query_embeddings = self.output_upscaling[i](original_query_embeddings)
                query_embeddings = self.output_upscaling[i](query_embeddings)
            d4 = prototypes.shape[2]
            _, _, h8, w8 = query_embeddings.shape
            proto_logits = (
                prototypes @ original_query_embeddings.view(b, d4, h8 * w8)
            ).view(b, -1, h8, w8)
            upscaled_embeddings = self.output_upscaling[-1](query_embeddings)
        else:
            upscaled_embeddings = self.output_upscaling(query_embeddings)
        # Put padding again in the class dimension
        _, _, h8, w8 = upscaled_embeddings.shape
        padded_logits = torch.full(
            (b * c, 1, h8, w8), float("-inf"), device=upscaled_embeddings.device
        )
        padded_logits[batch_mask] = upscaled_embeddings

        logits = rearrange(padded_logits, "(b c) 1 h w -> b c h w", c=c)
        if self.class_embedding_mlp is not None:
            logits = logits + proto_logits
        return logits


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = (
                self.dropout(F.relu(layer(x)))
                if i < self.num_layers - 1
                else self.dropout(layer(x))
            )
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
