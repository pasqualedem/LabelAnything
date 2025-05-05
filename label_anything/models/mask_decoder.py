# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from einops import rearrange, reduce
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms.functional import resize

from typing import List, Tuple, Type
from einops import rearrange, repeat

from label_anything.data.utils import BatchKeys
from label_anything.utils.utils import ResultDict

from .common import AttentionMLPBlock, LayerNorm2d, MLPBlock


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
        conv_upsample_stride: int = 2,
        classification_levels: int = 1,
        dropout: float = 0.0,
        conv_classification: bool = False,
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
        
        self.level_reducer = nn.Conv2d(classification_levels, 1, (3, 3), padding="same") if classification_levels > 1 else None

        if conv_upsample_stride > 1 or classification_layer_downsample_rate > 1:
            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(
                    transformer_dim,
                    transformer_dim // first_layer_downsample_rate,
                    kernel_size=conv_upsample_stride,
                    stride=conv_upsample_stride,
                ),
                LayerNorm2d(transformer_dim // first_layer_downsample_rate),
                activation(),
                nn.ConvTranspose2d(
                    transformer_dim // first_layer_downsample_rate,
                    transformer_dim // classification_layer_downsample_rate,
                    kernel_size=conv_upsample_stride,
                    stride=conv_upsample_stride,
                ),
            )
            self.class_mlp = MLP(
                transformer_dim,
                transformer_dim,
                transformer_dim // classification_layer_downsample_rate,
                3,
                dropout=dropout,
            )
        else:
            self.output_upscaling = nn.Identity()
            self.class_mlp = nn.Identity()

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
        
        self.prototype_tconv = None
        if conv_classification:
            self.prototype_tconv = nn.Sequential(
                *[
                    nn.ConvTranspose2d(
                        in_channels=transformer_dim // classification_layer_downsample_rate,
                        out_channels=transformer_dim // classification_layer_downsample_rate,
                        kernel_size=3,
                        stride=1,
                        padding=0,
                        bias=False,
                    )
                    for _ in range(2)
                ]
            )

    def _get_pe_result(self, pe_result, flag_examples):
        flag_examples = (
            flag_examples
            if BatchKeys.FLAG_EXAMPLES not in pe_result
            else pe_result[BatchKeys.FLAG_EXAMPLES]
        )
        if self.segment_example_logits:
            class_embeddings = rearrange(
                pe_result[ResultDict.EXAMPLES_CLASS_EMBS], "b n c d -> b (n c) d"
            )
            embedding_mask = rearrange(flag_examples, "b m c -> b (m c)")
        else:
            class_embeddings = pe_result[ResultDict.CLASS_EMBS]
            embedding_mask = flag_examples.sum(dim=1).bool().int()
        return class_embeddings, flag_examples, embedding_mask

    def _upscale(self, query_embeddings, class_embeddings):
        class_embeddings = self.class_mlp(class_embeddings)
        upscaled_embeddings = self.output_upscaling(query_embeddings)
        return upscaled_embeddings, class_embeddings

    def _spatial_convs(self, query_embeddings):
        if self.spatial_convs is not None:
            query_embeddings = self.spatial_convs(query_embeddings)
        return query_embeddings

    def _classify(self, query_embeddings, class_embeddings, flag_examples):
        b, d, h, w = query_embeddings.shape
        b, n, c = flag_examples.shape
        if self.prototype_tconv is not None:
            class_embeddings = rearrange(class_embeddings, "b c d -> (b c) d 1 1")
            class_embeddings = self.prototype_tconv(class_embeddings)
            convs = list(rearrange(class_embeddings, "(b c) d h w -> b c d h w", c=c))
            queries = [q.unsqueeze(0) for q in query_embeddings]
            seg = torch.cat([F.conv2d(q, c, padding=2) for q, c in zip(queries, convs)])
        else:
            seg = (class_embeddings @ query_embeddings.view(b, d, h * w)).view(b, -1, h, w)
        if self.segment_example_logits:
            seg = rearrange(seg, "b (n c) h w -> b n c h w", c=c)
            seg[flag_examples.logical_not()] = float("-inf")
            seg = seg.max(dim=1).values
        return seg

    def forward(
        self,
        query_embeddings: torch.Tensor,
        support_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        pe_result: dict,
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
        class_embeddings, flag_examples, embedding_mask = self._get_pe_result(
            pe_result, flag_examples
        )

        class_embeddings, query_embeddings = self.transformer(
            query_embeddings, image_pe, class_embeddings, embedding_mask
        )
        query_embeddings = rearrange(query_embeddings, "b (h w) c -> b c h w", h=h)

        if self.level_reducer:
            cls1 = self._classify(query_embeddings, class_embeddings, flag_examples)

        upscaled_embeddings, class_embeddings = self._upscale(
            query_embeddings, class_embeddings
        )
        upscaled_embeddings = self._spatial_convs(upscaled_embeddings)
        
        cls0 = self._classify(upscaled_embeddings, class_embeddings, flag_examples)
        
        if not self.level_reducer:
            return cls0
        
        h0, w0 = cls0.shape[-2:]
        cls1 = resize(cls1, (h0, w0))
        seg = rearrange(torch.stack([cls0, cls1]), "l b c h w -> (b c) l h w")
        seg = self.level_reducer(seg)
        seg = rearrange(seg, "(b c) 1 h w -> b c h w", b=b)
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
            classification_layer_downsample_rate // 4
        )
        second_layer_depth = transformer_dim // (
            classification_layer_downsample_rate // 2
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
                second_layer_depth,
                3,
                dropout=0.0,
            )
            self.attn_token_to_image = AttentionMLPBlock(
                embed_dim=transformer_dim,
                downsample_rate=1,
                mlp_dim=2048,
                num_heads=8,
                act=activation,
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

    def prototype_transformer(
        self,
        query_embeddings,
        prototypes,
        image_pe,
        batch_mask,
    ):
        bc, d, h, w = query_embeddings.shape
        b, c = prototypes.shape[:2]
        affinity_embeddings = torch.full(
            (b * c, d, h, w), float("-inf"), device=query_embeddings.device
        )
        affinity_embeddings[batch_mask] = query_embeddings
        reduce_embeddings = reduce(
            affinity_embeddings, "(b c) d h w -> b d h w", "max", b=b, c=c
        )

        keys = reduce_embeddings + image_pe
        keys = rearrange(keys, "b d h w -> b (h w) d")
        prototypes = self.attn_token_to_image(prototypes, keys, keys)
        prototypes = self.class_embedding_mlp(prototypes)
        for i in range(len(self.output_upscaling) - 3):
            affinity_embeddings = self.output_upscaling[i](affinity_embeddings)
            reduce_embeddings = self.output_upscaling[i](reduce_embeddings)
        d4 = prototypes.shape[2]
        _, _, h8, w8 = reduce_embeddings.shape
        heads = 32
        proto_logits = rearrange(
            prototypes, "b d (c heads) -> (b heads) d c", heads=8
        ) @ rearrange(
            affinity_embeddings, "b (d heads) h w -> (b heads) d (h w)", heads=heads
        )
        proto_logits = rearrange(
            proto_logits,
            "(b heads) c (h w) -> (b c) heads h w",
            heads=heads,
            h=h8,
            w=w8,
        )
        for i in range(len(self.output_upscaling) - 3, len(self.output_upscaling) - 1):
            proto_logits = self.output_upscaling[i](proto_logits)
            affinity_embeddings = self.output_upscaling[i](affinity_embeddings)
        class_features = torch.cat([affinity_embeddings, proto_logits], dim=1)
        logits = self.output_upscaling[-1](class_features)
        return logits

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
            proto_logits = self.prototype_transformer(
                query_embeddings,
                prototypes,
                image_pe,
                batch_mask,
            )
        else:
            upscaled_embeddings = self.output_upscaling(query_embeddings)
            # Put padding again in the class dimension
            _, _, h8, w8 = upscaled_embeddings.shape
            padded_logits = torch.full(
                (b * c, 1, h8, w8), float("-inf"), device=upscaled_embeddings.device
            )
            padded_logits[batch_mask] = upscaled_embeddings

            logits = rearrange(padded_logits, "(b c) 1 h w -> b c h w", c=c)
        return logits


class MultiLevelMaskDecoder(nn.Module):
    def __init__(
        self,
        mask_decoders: List[MaskDecoderLam],
        embed_dims: List[int],
        segment_example_logits: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoders = mask_decoders
        self.segment_example_logits = segment_example_logits
        max_embed_dim = max(embed_dims)

        self.feature_maps_projectors = nn.ModuleList(
            [
                nn.Conv2d(
                    embed_dim,
                    max_embed_dim,
                    kernel_size=1,
                )
                for embed_dim in embed_dims
            ]
        )
        self.class_embeddings_projectors = nn.ModuleList(
            [
                nn.Linear(
                    embed_dim,
                    max_embed_dim,
                )
                for embed_dim in embed_dims
            ]
        )

    def _classify(self, query_embeddings, class_embeddings, flag_examples):
        b, d, h, w = query_embeddings.shape
        _, c, _ = class_embeddings.shape
        seg = (class_embeddings @ query_embeddings.view(b, d, h * w)).view(b, -1, h, w)
        if self.segment_example_logits:
            seg = rearrange(seg, "b (n c) h w -> b n c h w", c=c)
            seg[flag_examples.logical_not()] = float("-inf")
            seg = seg.max(dim=1).values
        return seg

    def forward(
        self,
        query_embeddings: list[torch.Tensor],
        support_embeddings: list[torch.Tensor],
        image_pe: list[torch.Tensor],
        class_embeddings: dict[torch.Tensor],
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
        decoder_results = []
        # Turn dict of lists into list of dicts
        class_embeddings = [
            {k: v[i] for k, v in class_embeddings.items()}
            for i in range(len(query_embeddings))
        ]
        for (
            lv_query_embeddings,
            lv_support_embeddings,
            lv_image_pe,
            lv_class_embeddings,
            mask_decoder,
        ) in zip(
            query_embeddings,
            support_embeddings,
            image_pe,
            class_embeddings,
            self.mask_decoders,
        ):
            b, d, h, w = lv_query_embeddings.shape
            lv_class_embeddings = mask_decoder._get_pe_result(
                lv_class_embeddings
            )

            lv_class_embeddings, lv_query_embeddings = mask_decoder.transformer(
                lv_query_embeddings, lv_image_pe, lv_class_embeddings
            )
            lv_query_embeddings = rearrange(
                lv_query_embeddings, "b (h w) c -> b c h w", h=h
            )

            upscaled_embeddings, lv_class_embeddings = mask_decoder._upscale(
                lv_query_embeddings, lv_class_embeddings
            )
            upscaled_embeddings = mask_decoder._spatial_convs(upscaled_embeddings)
            decoder_results.append((upscaled_embeddings, lv_class_embeddings))
        b, d, h, w = decoder_results[0][0].shape  # Get max h, w
        feature_maps, lv_class_embeddings = zip(*decoder_results)
        # Upscale feature maps
        feature_maps = [
            F.interpolate(fm, size=(h, w), mode="bilinear") for fm in feature_maps
        ]
        # Project feature maps and class embeddings to the same dimension
        feature_maps = [
            projector(fm)
            for fm, projector in zip(feature_maps, self.feature_maps_projectors)
        ]
        lv_class_embeddings = [
            projector(ce)
            for ce, projector in zip(
                lv_class_embeddings, self.class_embeddings_projectors
            )
        ]

        # Sum feature maps and class embeddings
        feature_maps = sum(feature_maps)
        lv_class_embeddings = sum(lv_class_embeddings)

        # Classify
        return self._classify(feature_maps, lv_class_embeddings, flag_examples)


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
