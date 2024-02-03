# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from einops import rearrange

from typing import Any, Optional, Tuple, Type

from .common import Attention, LayerNorm2d, MLPBlock, AttentionMLPBlock
from .transformer import TwoWayTransformer

from label_anything.data.utils import Label


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [
            nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)
        ]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # For when no masks in input

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )
        point_embedding[labels == Label.NULL] = 0.0
        point_embedding[labels == Label.NULL] += self.not_a_point_embed.weight
        point_embedding[labels == Label.NEGATIVE] += self.point_embeddings[0].weight
        point_embedding[labels == Label.POSITIVE] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = coords.type(
            self.positional_encoding_gaussian_matrix.dtype
        )  # Ensure same type
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords)  # B x N x C


class PromptImageEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        transformer: nn.Module,
        class_example_attention: bool = True,
        class_attention: bool = False,
        example_attention: bool = False,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to LAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__(
            embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation
        )

        num_heads: int = 8
        attention_downsample_rate: int = 2
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        mlp_dim: int = 2048

        self.transformer = transformer

        self.sparse_embedding_attention = AttentionMLPBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            downsample_rate=attention_downsample_rate,
            mlp_dim=mlp_dim,
            act=activation,
        )
        self.no_sparse_embedding = nn.Embedding(
            1, embed_dim
        )  # For when no sparse embeddings in input
        
        self.example_attention = None
        if example_attention:
            self.example_attention = AttentionMLPBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                downsample_rate=attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
            )
        self.class_attention = None
        if class_attention:
            self.class_attention = AttentionMLPBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                downsample_rate=attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
            )

        if class_example_attention:
            self.class_example_attention = AttentionMLPBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                downsample_rate=attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
            )

        self.not_a_mask_embed = nn.Embedding(
            1, embed_dim // 4
        )  # For classes/examples with missing masks

    def _embed_masks(
        self, masks: torch.Tensor, masks_flags: torch.Tensor, chunk_size=None
    ) -> torch.Tensor:
        """Embeds mask inputs. (B, C, H, W)"""
        B, M, C, H, W = masks.shape
        masks = rearrange(masks, "b m c h w -> (b m c) 1 h w")
        if chunk_size is None:
            mask_embedding = self.mask_downscaling(masks)
        else:
            for i in range(0, masks.shape[0], chunk_size):
                mask_embedding = torch.zeros(
                    B * M * C, self.embed_dim, H // 4, W // 4, device=self._get_device()
                )
                mask_embedding[i : i + chunk_size] = self.mask_downscaling(
                    masks[i : i + chunk_size]
                )
        mask_embedding = rearrange(
            mask_embedding, "(b m c) d h w -> b m c d h w", b=B, m=M
        )
        H, W = mask_embedding.shape[-2:]
        mask_embedding[masks_flags == Label.NULL] = 0.0
        mask_embedding[masks_flags == Label.NULL] += self.not_a_mask_embed.weight
        return mask_embedding

    def _get_batch_examples_class_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size and the number classes of the output
        given the batch size and the number of classes of the input prompts.
        """
        if points is not None:
            points = points[0]
            return points.shape[0], points.shape[1], points.shape[2]
        elif boxes is not None:
            boxes = boxes[0]
            return boxes.shape[0], boxes.shape[1], boxes.shape[2]
        elif masks is not None:
            masks = masks[0]
            return masks.shape[0], masks.shape[1], masks.shape[2]
        else:
            raise ValueError("No prompts provided")

    def embed_points_masks(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[Tuple[torch.Tensor, torch.Tensor]],
        masks: Optional[Tuple[torch.Tensor, torch.Tensor]],
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (tuple(torch.Tensor, torch.Tensor) or none): boxes to embed and padding
          masks (tuple(torch.Tensor, torch.Tensor) or none): masks to embed and padding

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        B, n_examples, n_classes = self._get_batch_examples_class_size(
            points, boxes, masks
        )
        bs = B * n_examples * n_classes

        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            device=self._get_device(),
            dtype=self.no_sparse_embedding.weight.dtype,
        )
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            boxes, flags = boxes
            box_embeddings = self._embed_boxes(boxes, flags)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        if boxes is None and points is None:
            sparse_embeddings = torch.zeros(
                (bs, 1, self.embed_dim), device=self._get_device()
            )
            sparse_embeddings += self.no_sparse_embedding.weight

        # Attention over sparse embeddings
        sparse_embeddings = rearrange(
            sparse_embeddings,
            "(b m c) n d -> (b m) (c n) d",
            b=B,
            m=n_examples,
            c=n_classes,
        )
        sparse_embeddings = self.sparse_embedding_attention(sparse_embeddings)
        sparse_embeddings = rearrange(
            sparse_embeddings,
            "(b m) (c n) d -> b m c n d",
            b=B,
            m=n_examples,
            c=n_classes,
        )

        if masks is not None:
            mask_inputs, mask_flags = masks
            dense_embeddings = self._embed_masks(mask_inputs, mask_flags, chunk_size)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, 1, 1, -1, 1, 1
            ).expand(
                B,
                n_examples,
                n_classes,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings

    def _embed_points(
        self, points: torch.Tensor, labels: torch.Tensor, pad: bool
    ) -> torch.Tensor:
        B = points.shape[0]
        points = rearrange(points, "b m c n xy -> (b m c) n xy")
        labels = rearrange(labels, "b m c n -> (b m c) n")
        return super()._embed_points(points, labels, pad)

    def _embed_boxes(self, boxes: torch.Tensor, padding) -> torch.Tensor:
        b, m, c, n = boxes.shape[:4]
        boxes = rearrange(boxes, "b m c n xy -> (b m c) n xy")
        box_embeddings = super()._embed_boxes(boxes)
        box_embeddings = rearrange(
            box_embeddings, "(b m c n) xy d -> b m c (n xy) d", b=b, c=c, m=m, n=n
        )
        two_points_padding = padding.repeat(1, 1, 1, 2)
        box_embeddings[two_points_padding == Label.NULL] = 0.0
        box_embeddings[
            two_points_padding == Label.NULL
        ] += self.not_a_point_embed.weight
        box_embeddings = rearrange(box_embeddings, "b m c n d-> (b m c) n d")
        return box_embeddings

    def get_unattended_prompts(
        self,
        image_embeddings: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning unattended prompts

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed (B, M, C, 2)
          boxes (torch.Tensor or none): boxes to embed (B, M, C, 2, 2)
          masks (torch.Tensor or none): masks to embed (B, M, C, H, W)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """

    def sparse_dense_fusion(self, src, pos_src, sparse_embeddings, chunk_size=None):
        _, _, h, w = src.shape
        if chunk_size is None:
            return rearrange(
                self.transformer(src, pos_src, sparse_embeddings)[1],
                "b (h w) d  -> b d h w",
                h=h,
            )  # src: (BMC, HW, D)
        for i in range(0, src.shape[1], chunk_size):
            _, attn_out = self.transformer(
                src[i : i + chunk_size],
                pos_src[i : i + chunk_size],
                sparse_embeddings[i : i + chunk_size],
            )
            src[i : i + chunk_size] = rearrange(attn_out, "b (h w) d  -> b d h w", h=h)
        return src

    def forward(
        self,
        image_embeddings: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        chunk_size=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning class embeddings

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed (B, M, C, 2)
          boxes (torch.Tensor or none): boxes to embed (B, M, C, 2, 2)
          masks (torch.Tensor or none): masks to embed (B, M, C, H, W)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        sparse_embeddings, dense_embeddings = self.embed_points_masks(
            points, boxes, masks, chunk_size=chunk_size
        )
        sparse_embeddings = rearrange(sparse_embeddings, "b m c n d -> (b m c) n d")

        b, m, c, d, h, w = dense_embeddings.shape
        dense_embeddings = rearrange(dense_embeddings, "b m c d h w -> (b m c) d h w")

        src = rearrange(image_embeddings, "b m d h w -> b m 1 d h w").repeat(
            1, 1, c, 1, 1, 1
        )
        src = rearrange(src, "b m c d h w -> (b m c) d h w")
        src = src + dense_embeddings
        pos_src = torch.repeat_interleave(
            self.get_dense_pe(), sparse_embeddings.shape[0], dim=0
        )

        # Run the transformer to fuse the dense embeddings and sparse embeddings
        src = self.sparse_dense_fusion(
            src, pos_src, sparse_embeddings, chunk_size=chunk_size
        )
        src = rearrange(src, "b d h w -> b d (h w)")
        src = nn.functional.adaptive_avg_pool1d(src, (1)).squeeze(2)  # (BMC, D)
        src = rearrange(src, "(b m c) d -> b m c d", b=b, m=m, c=c)
        
        if self.class_example_attention is not None:
            src = rearrange(src, "b m c d -> b (m c) d", c=c)
            src = self.class_example_attention(src)
            src = rearrange(src, "b (m c) d -> b m c d", c=c)

        if self.example_attention is not None:
            src = rearrange(src, "b m c d -> (b c) m d", c=c)
            src = self.example_attention(src)
            src = rearrange(src, "(b c) m d -> b m c d", c=c)
        if self.class_attention is not None:
            src = rearrange(src, "b m c d -> (b m) c d", c=c)
            src = self.class_attention(src)
            src = rearrange(src, "(b m) c d -> b m c d", m=m)

        # Average over examples
        src = torch.mean(src, dim=1)  # (B, C, D)
        return src


class PromptMaskImageEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        transformer: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to LAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__(
            embed_dim, image_embedding_size, input_image_size, mask_in_chans, activation
        )

        num_heads: int = 8
        attention_downsample_rate: int = 2
        mlp_dim: int = 2048

        self.example_attention = Attention(
            embed_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_example_attention = nn.LayerNorm(embed_dim)
        self.example_mlp = MLPBlock(embed_dim, mlp_dim)
        self.norm_example_mlp = nn.LayerNorm(embed_dim)

        self.not_a_mask_embed = nn.Embedding(
            1, embed_dim // 4
        )  # For classes/examples with missing masks

    def _embed_masks(
        self, masks: torch.Tensor, masks_flags: torch.Tensor
    ) -> torch.Tensor:
        """Embeds mask inputs. (B, C, H, W)"""
        B, M, C, _, _ = masks.shape
        masks = rearrange(masks, "b m c h w -> (b m c) 1 h w")
        mask_embedding = self.mask_downscaling(masks)
        mask_embedding = rearrange(
            mask_embedding, "(b m c) d h w -> b m c d h w", b=B, m=M
        )
        H, W = mask_embedding.shape[-2:]
        mask_embedding[masks_flags == Label.NULL] = 0.0
        mask_embedding[masks_flags == Label.NULL] += self.not_a_mask_embed.weight
        return mask_embedding

    def _get_batch_examples_class_size(
        self,
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size and the number classes of the output
        given the batch size and the number of classes of the input prompts.
        """
        if masks is not None:
            masks = masks[0]
            return masks.shape[0], masks.shape[1], masks.shape[2]
        else:
            return 1

    def forward(
        self,
        image_embeddings: torch.Tensor,
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds masks, returning class embeddings

        Arguments:
          masks (torch.Tensor or none): masks to embed (B, M, C, H, W)

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        dense_embeddings = self._embed_masks(masks)
        b, m, c, d, h, w = dense_embeddings.shape
        dense_embeddings = rearrange(dense_embeddings, "b m c d h w -> (b m c) d h w")

        src = rearrange(image_embeddings, "b m d h w -> b m 1 d h w").repeat(
            1, 1, c, 1, 1, 1
        )
        src = rearrange(src, "b m c d h w -> (b m c) d h w")
        src = src + dense_embeddings
        src = rearrange(src, "(b m c) d h w -> b (m c) d", b=b, m=m, c=c)

        src = self.example_attention(src, src, src)
        src = self.norm_example_attention(src)
        src = self.example_mlp(src) + src
        src = self.norm_example_mlp(src)

        # Average over examples
        src = rearrange(src, "b (m c) d -> b m c d", c=c)
        src = torch.mean(src, dim=1)  # (B, C, D)
        return src
