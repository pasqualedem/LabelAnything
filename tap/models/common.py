# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from einops import repeat
import torch
import torch.nn as nn

from typing import Type


SAM_EMBED_DIM = 256


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.drop(self.act(self.lin1(x))))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        if dropout > 0.0:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = nn.Identity()
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        key_mask=None,
        attn_mask=None,
    ) -> torch.Tensor:
        bsz, src_len, _ = q.shape
        
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Masks
        _, _, _, c_per_head = q.shape
        score_mask = None
        if key_mask is not None:
            key_mask = repeat(key_mask, 'b n -> b h s n', h=self.num_heads, s=src_len)
            score_mask = torch.zeros_like(
                key_mask, device=key_mask.device, dtype=torch.bool
            )
        if attn_mask is not None:
            # Put 0 in attn_mask where is 1 and -inf where is 0
            mask = torch.zeros_like(
                attn_mask, device=attn_mask.device, dtype=torch.bool
            )
            if score_mask is None:
                score_mask = mask
            else:
                score_mask = score_mask.bitwise_or(mask).logical_not()
            
        # Attention
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        if score_mask is not None:
            attn[score_mask] = float("-inf")
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class AttentionMLPBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        downsample_rate: int,
        mlp_dim: int,
        num_heads: int,
        act: Type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim, act, dropout=dropout)
        self.attn = Attention(
            embed_dim,
            num_heads=num_heads,
            downsample_rate=downsample_rate,
            dropout=dropout,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        key_mask=None,
        attn_mask=None,
    ) -> torch.Tensor:
        if k is None:
            k = q
        if v is None:
            v = q
        attn_out = self.norm(self.attn(q, k, v, key_mask, attn_mask) + q)
        return self.norm(self.mlp(attn_out) + attn_out)
