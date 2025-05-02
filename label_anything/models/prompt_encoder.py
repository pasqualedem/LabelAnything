# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from einops import rearrange, reduce, repeat

from typing import Any, Optional, Tuple, Type

from .common import Attention, LayerNorm2d, MLPBlock, AttentionMLPBlock
from .transformer import TwoWayTransformer, OneWayAttentionBlock

from label_anything.data.utils import BatchKeys, Label
from label_anything.utils.utils import ResultDict


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


class RandomMatrixEncoder(nn.Module):
    def __init__(self, bank_size, embed_dim):
        super().__init__()
        seed = 42
        self.bank_size = bank_size
        self.embed_dim = embed_dim
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, bank_size, embed_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

    def sample_rows(self, C, device):
        fg_rows = torch.randperm(self.bank_size - 1, device=device)[: C - 1] + 1
        bg_rows = torch.zeros(1, device=device, dtype=torch.long)
        return torch.cat([bg_rows, fg_rows])

    def forward_with_rows(self, dense_embeddings, sparse_embeddings, selected_rows):
        B, M, C, N, D = sparse_embeddings.shape
        class_encoding = self.pos_embedding[:, :, selected_rows]  # 1 x 1 x C x D
        sparse_class_encoding = repeat(
            class_encoding, "1 1 c d -> b m c n d", b=B, m=M, n=N
        )
        sparse_embeddings = sparse_embeddings + sparse_class_encoding

        B, M, C, D, H, W = dense_embeddings.shape
        dense_class_encoding = repeat(
            class_encoding, "1 1 c d -> b m c d h w", b=B, m=M, h=H, w=W
        )
        dense_embeddings = dense_embeddings + dense_class_encoding

        return dense_embeddings, sparse_embeddings

    def forward(self, dense_embeddings, sparse_embeddings):
        """Adds random class embedding

        Args:
            sparse_embeddings (torch.Tensor): Sparse embeddings with shape B x M x C x N x D
            dense_embeddings (torch.Tensor): Dense embeddings with shape B x M x C x D x H x W
        """
        B, M, C, N, D = sparse_embeddings.shape
        selected_rows = self.sample_rows(C, device=sparse_embeddings.device)
        return self.forward_with_rows(
            dense_embeddings, sparse_embeddings, selected_rows
        )
        

class EmbeddingTransformer(nn.Module):
    def __init__(self, emb_dim, num_embeddings, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            OneWayAttentionBlock(embedding_dim=emb_dim, num_heads=8) for _ in range(num_layers)
        ])
        self.embeddings = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=emb_dim)
        self.embedding_dropout = 0.2
        
    def forward(self, src, image_pe, flag_examples):
        b, m, c = flag_examples.shape
        h, w = src.shape[-2:]
        n = self.embeddings.weight.shape[0]
        embeddings = repeat(self.embeddings.weight, "n d -> (b c) n d", b=b, c=c)
        key_mask = repeat(flag_examples, "b m c -> (b c) (m h w)", h=h, w=w)
        src = rearrange(src, "(b m c) d h w -> (b c) (m h w) d", b=b, m=m)
        for layer in self.layers:
            embeddings = layer(embeddings, src, key_mask=key_mask, query_pe=torch.zeros_like(embeddings))
            
        flag_embeddings = flag_examples.sum(dim=1).bool().int()
        flag_embeddings = repeat(flag_embeddings, " b c -> b n c", n=n)
        if self.training:
            included = torch.rand(n).to(flag_embeddings.device) > self.embedding_dropout
            if not included.any():  # Check if all are False
                random_index = torch.randint(0, n, (1,)).item()  # Randomly select an index
                included[random_index] = True  # Set it to True
            included = rearrange(included, "n -> 1 n 1")
            flag_embeddings = flag_embeddings * included
        embeddings = rearrange(embeddings, "(b c) n d -> b n c d", c=c)

        return {
            ResultDict.EXAMPLES_CLASS_EMBS: embeddings,
            BatchKeys.FLAG_EXAMPLES: flag_embeddings,
        }
        
class GuidedPooler(nn.Module):
    def __init__(self, emb_dim, num_embeddings):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.attention = nn.MultiheadAttention(emb_dim, 8)
        
        self.fg_chooser = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim // 2, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 2, emb_dim // 4, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 4, emb_dim // 8, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 8, num_embeddings+1, (1, 1), stride=1, padding=0)
        )

        self.bg_chooser = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim // 2, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 2, emb_dim // 4, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 4, emb_dim // 8, (1, 1), stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(emb_dim // 8, num_embeddings+1, (1, 1), stride=1, padding=0)
        )

    
    def act(self, x):
        tau = 0.5
        return torch.nn.functional.gumbel_softmax(x, tau=tau, hard=False)
        
    def forward(self, src, image_pe, flag_examples):
        fg_flag_examples = flag_examples.clone()
        b, m, c  = fg_flag_examples.shape
        h, w = src.shape[-2:]
        
        # Remove bg
        fg_flag_examples = fg_flag_examples[:, :, 1:]
        src = src + image_pe
        src = rearrange(src, "(b m c) d ... -> b m c d ...", b=b, m=m)
        fg_src = src[:, :, 1:]
        bg_src = fg_src.mean(dim=2).unsqueeze(2)
        bg_flag_examples = fg_flag_examples.sum(dim=2).bool().int().unsqueeze(2)
        fg_src = rearrange(fg_src, "b m c d h w -> (b m c) (h w) d", b=b, m=m)
        bg_src = rearrange(bg_src, "b m c d h w -> (b m c) (h w) d", b=b, m=m, c=1)
        
        fg_src, _ = self.attention(fg_src, fg_src, fg_src)
        bg_src, _ = self.attention(bg_src, bg_src, bg_src)
        
        fg_src = rearrange(fg_src, "(b m c) (h w) d -> (b m c) d h w", b=b, m=m, h=h)
        bg_src = rearrange(bg_src, "(b m c) (h w) d -> (b m c) d h w", b=b, m=m, c=1, h=h)
        
        fg_mask = self.fg_chooser(fg_src)
        bg_mask = self.bg_chooser(bg_src)
        
        fg_mask = rearrange(self.act(fg_mask), "bmc n ... -> n bmc 1 ...")[1:]
        bg_mask = rearrange(self.act(bg_mask), "bmc n ... -> n bmc 1 ...")[1:]
                
        fg_src = repeat(fg_src, "... -> n ...", n=self.num_embeddings)
        bg_src = repeat(bg_src, "... -> n ...", n=self.num_embeddings)
        
        fg = fg_mask * fg_src
        bg = bg_mask * bg_src
        
        fg = nn.functional.adaptive_avg_pool2d(fg, (1, 1))
        bg = nn.functional.adaptive_avg_pool2d(bg, (1, 1))
        flag_examples = torch.cat([bg_flag_examples, fg_flag_examples], dim=2)
        flag_examples = repeat(flag_examples, "b m c -> b (n m) c", n=self.num_embeddings)
        
        fg = rearrange(fg, "n (b m c) d 1 1 -> b (n m) c d", b=b, m=m, c=(c - 1))
        bg = rearrange(bg, "n (b m c) d 1 1 -> b (n m) c d", b=b, m=m, c=1)
        
        embeddings = torch.cat([bg, fg], dim=2)
        
        return {
            ResultDict.EXAMPLES_CLASS_EMBS: embeddings,
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            ResultDict.MASK_EMBEDDINGS: (bg_mask, fg_mask)
        }


class PromptImageEncoder(PromptEncoder):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        transformer: nn.Module,
        class_encoder: nn.Module,
        example_class_attention: bool = True,
        class_attention: bool = False,
        class_embedding_dim: int = None,
        example_attention: bool = False,
        activation: Type[nn.Module] = nn.GELU,
        use_support_features: bool = True,
        embeddings_per_example: int = 1,
        embedding_extraction: str = None,
        dropout: float = 0.0,
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
        self.embeddings_per_example = embeddings_per_example
        mlp_dim: int = 2048

        self.transformer = transformer
        self.class_encoder = class_encoder
        self.use_support_features = use_support_features
        if embedding_extraction == "pooler":
            self.embedding_extraction = GuidedPooler(emb_dim=embed_dim, num_embeddings=embeddings_per_example)
        elif embedding_extraction == "cross_attention":
            self.embedding_extraction = EmbeddingTransformer(emb_dim=embed_dim, num_embeddings=embeddings_per_example, num_layers=2)
        else:
            self.embedding_extraction = None            

        self.sparse_embedding_attention = AttentionMLPBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            downsample_rate=1,
            mlp_dim=mlp_dim,
            act=activation,
            dropout=dropout,
        )
        self.no_sparse_embedding = nn.Embedding(
            1, embed_dim
        )  # For when no sparse embeddings in input

        if class_embedding_dim is not None:
            class_attention_downsample_rate = 1
            self.class_projector_in = nn.Linear(embed_dim, class_embedding_dim)
            self.class_projector_out = nn.Linear(class_embedding_dim, embed_dim)
        else:
            class_embedding_dim = embed_dim
            self.class_projector_in = nn.Identity()
            self.class_projector_out = nn.Identity()
            class_attention_downsample_rate = attention_downsample_rate

        self.class_attention = None
        if class_attention:
            self.class_attention = AttentionMLPBlock(
                embed_dim=class_embedding_dim,
                num_heads=num_heads,
                downsample_rate=class_attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
                dropout=dropout,
            )

        self.class_example_attention = None
        if example_class_attention:
            self.class_example_attention = AttentionMLPBlock(
                embed_dim=class_embedding_dim,
                num_heads=num_heads,
                downsample_rate=class_attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
                dropout=dropout,
            )

        self.example_attention = None
        if example_attention:
            self.example_attention = AttentionMLPBlock(
                embed_dim=class_embedding_dim,
                num_heads=num_heads,
                downsample_rate=class_attention_downsample_rate,
                mlp_dim=mlp_dim,
                act=activation,
                dropout=dropout,
            )
            
        if not self.use_support_features:
            self.proto_chooser = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 8, (1, 1), stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(embed_dim // 8, 1, (1, 1), stride=1, padding=0),
                nn.Sigmoid(),
            )
                 
        self.not_a_mask_embed = nn.Embedding(
            1, embed_dim
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
        mask_embedding[masks_flags == Label.NULL] += rearrange(
            self.not_a_mask_embed.weight, "1 d -> 1 d 1 1"
        )
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

        sparse_embeddings = rearrange(
            sparse_embeddings,
            "(b m c) n d -> (b m) (c n) d",
            b=B,
            m=n_examples,
            c=n_classes,
        )

        # Attention over sparse embeddings
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

    def sparse_dense_fusion(self, src, pos_src, sparse_embeddings, chunk_size=None):
        src, sparse_embeddings = self.class_encoder(
            src, sparse_embeddings
        )  # Inject class awareness
        return self.apply_transformer(src, pos_src, sparse_embeddings, chunk_size)

    def apply_transformer(self, src, pos_src, sparse_embeddings, chunk_size=None):
        b, m, c, d, h, w = src.shape
        src = rearrange(src, "b m c d h w -> (b m c) d h w")
        sparse_embeddings = rearrange(sparse_embeddings, "b m c n d -> (b m c) n d")
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

    def prompt_class_information_merge(self, embeddings, flag_examples):
        b, m, c, _ = embeddings.shape
        embeddings = self.class_projector_in(embeddings)
        if self.class_attention is not None:
            embeddings = rearrange(embeddings, "b m c d -> (b m) c d", c=c)
            key_mask = rearrange(flag_examples, "b m c -> (b m) c")
            embeddings = self.class_attention(embeddings, key_mask=key_mask)
            embeddings = rearrange(embeddings, "(b m) c d -> b m c d", m=m)

        if self.example_attention is not None:
            embeddings = rearrange(embeddings, "b m c d -> (b c) m d", c=c)
            key_mask = rearrange(flag_examples, "b m c -> (b c) m")
            embeddings = self.example_attention(embeddings, key_mask=key_mask)
            embeddings = rearrange(embeddings, "(b c) m d -> b m c d", c=c)

        if self.class_example_attention is not None:
            embeddings = rearrange(embeddings, "b m c d -> b (m c) d", c=c)
            key_mask = rearrange(flag_examples, "b m c -> b (m c)")
            embeddings = self.class_example_attention(embeddings, key_mask=key_mask)
            embeddings = rearrange(embeddings, "b (m c) d -> b m c d", c=c)
        embeddings = self.class_projector_out(embeddings)
        return embeddings
    
    def _obtain_embeddings(self, src, image_pe, flag_examples):
        _, d, h, w = src.shape
        b, m, c = flag_examples.shape
        if self.embedding_extraction:
            class_embeddings = None
            return self.embedding_extraction(src, image_pe, flag_examples)
        
        if self.embeddings_per_example and self.embeddings_per_example > 1:
            num_embeddings = int(torch.sqrt(torch.tensor(self.embeddings_per_example)))
            embeddings = nn.functional.adaptive_avg_pool2d(src, (num_embeddings, num_embeddings)) # (BMC, D, num_embeddings, num_embeddings)
            embeddings = rearrange(embeddings, "(b m c) d h w -> b (m h w) c d", b=b, m=m, c=c, d=d)
            m = m * num_embeddings * num_embeddings
            flag_examples = repeat(flag_examples, "b m c -> b (m h w) c", h=num_embeddings, w=num_embeddings)
        else:
            src = rearrange(src, "b d h w -> b d (h w)")
            embeddings = nn.functional.adaptive_avg_pool1d(src, (1)).squeeze(2)  # (BMC, D)
            embeddings = rearrange(embeddings, "(b m c) d -> b m c d", b=b, m=m, c=c)
        embeddings = self.prompt_class_information_merge(embeddings, flag_examples)

        # Average over examples removing padding embeddings
        masked_embeddings = embeddings * flag_examples.unsqueeze(-1)
        normalizer = flag_examples.clone().unsqueeze(-1).sum(dim=1).float()
        normalizer[normalizer == 0] = (
            1  # Put 1 in padding to avoid division by 0 (logits will be put to -inf)
        )

        class_embeddings = masked_embeddings.sum(dim=1) / normalizer
        return {
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            ResultDict.CLASS_EMBS: class_embeddings,
            ResultDict.EXAMPLES_CLASS_EMBS: embeddings
        }

    def forward(
        self,
        image_embeddings: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        flag_examples: Optional[torch.Tensor],
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
          flag_examples (torch.Tensor or none): flags to indicate which examples (B, M, C)

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

        if image_embeddings.shape[-2:] != dense_embeddings.shape[-2:]:
            dense_embeddings = nn.functional.interpolate(
                dense_embeddings,
                size=image_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        if self.use_support_features:
            src = rearrange(image_embeddings, "b m d h w -> b m 1 d h w").repeat(
                1, 1, c, 1, 1, 1
            )
            src = rearrange(src, "b m c d h w -> (b m c) d h w")
            src = src + dense_embeddings
        else:
            src = dense_embeddings
        pos_src = torch.repeat_interleave(
            self.get_dense_pe(), sparse_embeddings.shape[0], dim=0
        )

        # Run the transformer to fuse the dense embeddings and sparse embeddings
        src = rearrange(src, "(b m c) d h w -> b m c d h w", b=b, m=m, c=c)
        sparse_embeddings = rearrange(
            sparse_embeddings, "(b m c) n d -> b m c n d", b=b, m=m, c=c
        )
        src = self.sparse_dense_fusion(
            src, pos_src, sparse_embeddings, chunk_size=chunk_size
        )
        if not self.use_support_features:
            mask = self.proto_chooser(src)
            # mask = rearrange(mask, "(b m c) d h w -> b m c d h w", b=b, m=m, c=c)
            src = repeat(image_embeddings, "b m d h w -> (b m c) d h w", c=c)
            src = src * mask
            
        embeddings_dict = self._obtain_embeddings(
            src, pos_src, flag_examples
        )
        return {
            **embeddings_dict,
            ResultDict.EXAMPLES_CLASS_SRC: src,
        }


class PromptImagePoolEncoder(PromptImageEncoder):
    def _obtain_embeddings(self, embeddings, flag_examples):
        embeddings = self.prompt_class_information_merge(embeddings, flag_examples)

        # Average over examples removing padding embeddings
        masked_embeddings = embeddings * flag_examples.unsqueeze(-1)
        normalizer = flag_examples.clone().unsqueeze(-1).sum(dim=1).float()
        normalizer[normalizer == 0] = (
            1  # Put 1 in padding to avoid division by 0 (logits will be put to -inf)
        )

        class_embeddings = masked_embeddings.sum(dim=1) / normalizer
        return {
            BatchKeys.FLAG_EXAMPLES: flag_examples,
            ResultDict.CLASS_EMBS: class_embeddings,
            ResultDict.EXAMPLES_CLASS_EMBS: embeddings
        }
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        flag_examples: Optional[torch.Tensor],
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
          flag_examples (torch.Tensor or none): flags to indicate which examples (B, M, C)

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

        if image_embeddings.shape[-2:] != dense_embeddings.shape[-2:]:
            dense_embeddings = nn.functional.interpolate(
                dense_embeddings,
                size=image_embeddings.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        dense_embeddings = rearrange(dense_embeddings, "(b m c) d h w -> b m c d h w", b=b, m=m, c=c)
        sparse_embeddings = rearrange(sparse_embeddings, "(b m c) n d -> b m c n d", b=b, m=m, c=c)

        if self.use_support_features:
            dense_embeddings, sparse_embeddings = self.class_encoder(dense_embeddings, sparse_embeddings)
            dense_embeddings = dense_embeddings.sum(dim=2)
            src = image_embeddings + dense_embeddings
        else:
            raise NotImplementedError(
                "PromptImagePoolEncoder does not support use_support_features=False"
            )

        # Run the transformer to fuse the dense embeddings and sparse embeddings
        src = rearrange(src, "b m d h w -> (b m) d h w")
        sparse_embeddings = rearrange(sparse_embeddings, "b m c n d -> (b m) (c n) d")
        pos_src = torch.repeat_interleave(
            self.get_dense_pe(), src.shape[0], dim=0
        )
        embeddings, src = self.transformer(src, pos_src, sparse_embeddings)
        embeddings = reduce(embeddings, "(b m) (c n) d -> b m c d", b=b, c=c, reduction="mean")
            
        embeddings_dict = self._obtain_embeddings(embeddings, flag_examples)
        return {
            **embeddings_dict,
            ResultDict.EXAMPLES_CLASS_SRC: src,
        }



class MultiLevelPromptEncoder(nn.Module):
    def __init__(
        self,
        prompt_encoders: nn.ModuleList,
    ) -> None:
        super().__init__()
        self.prompt_encoders = prompt_encoders

    def forward(
        self,
        image_embeddings: torch.Tensor,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        flag_examples: Optional[torch.Tensor],
        chunk_size=None,
    ):
        result_dict = {
            result_key: []
            for result_key in [
                ResultDict.CLASS_EMBS,
                ResultDict.EXAMPLES_CLASS_SRC,
                ResultDict.EXAMPLES_CLASS_EMBS,
            ]
        }
        for prompt_encoder, level_image_embeddings in zip(
            self.prompt_encoders, image_embeddings
        ):
            result = prompt_encoder(
                level_image_embeddings,
                points,
                boxes,
                masks,
                flag_examples,
                chunk_size=chunk_size,
            )

            result_dict[ResultDict.CLASS_EMBS].append(result[ResultDict.CLASS_EMBS])
            result_dict[ResultDict.EXAMPLES_CLASS_SRC].append(
                result[ResultDict.EXAMPLES_CLASS_SRC]
            )
            result_dict[ResultDict.EXAMPLES_CLASS_EMBS].append(
                result[ResultDict.EXAMPLES_CLASS_EMBS]
            )
        return result_dict

    def get_dense_pe(self):
        return [
            prompt_encoder.get_dense_pe() for prompt_encoder in self.prompt_encoders
        ]
