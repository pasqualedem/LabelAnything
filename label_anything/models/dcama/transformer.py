
import inspect
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.attn_fn = attention
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, return_attn=False):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attn_fn(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = torch.mean(x, -3)

        return (x, self.attn) if return_attn else x


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None, aggregation='sum', **kwargs):
    "Compute 'Scaled Dot Product Attention' with customizable aggregation and hyperparameters"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax for attention weights
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    if aggregation == 'sum':
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'max':
        # No hyperparameters needed for max
        max_scores, _ = torch.max(scores, dim=-1, keepdim=True)
        p_attn = (scores == max_scores).float()  # Hard attention
        p_attn = F.normalize(p_attn, p=1, dim=-1)  # Normalize for weighted aggregation
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'threshold':
        # Get threshold from kwargs, default is 0.5
        threshold = kwargs.get('threshold', 0.5)
        p_attn = p_attn * (scores > threshold).float()  # Mask values below the threshold
        p_attn = F.normalize(p_attn, p=1, dim=-1)  # Normalize again
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'power':
        # Get gamma from kwargs, default is 2
        gamma = kwargs.get('gamma', 2)
        p_attn = p_attn**gamma
        p_attn = F.normalize(p_attn, p=1, dim=-1)  # Normalize for proper weighting
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'lse':
        # Get lambda_param from kwargs, default is 1.0
        lambda_param = kwargs.get('lambda_param', 1.0)
        lse_scores = (1 / lambda_param) * torch.logsumexp(lambda_param * scores, dim=-1, keepdim=True)
        p_attn = torch.exp(scores - lse_scores)  # Recompute normalized probabilities
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'sigmoid':
        # Get tau (threshold) and k (steepness) from kwargs
        tau = kwargs.get('tau', 0.5)
        k = kwargs.get('k', 10)
        sigmoid_weights = torch.sigmoid(k * (scores - tau))
        p_attn = p_attn * sigmoid_weights
        p_attn = F.normalize(p_attn, p=1, dim=-1)  # Normalize again
        return torch.matmul(p_attn, value), p_attn

    if aggregation == 'hard':
        # No hyperparameters needed for hard attention
        _, max_indices = torch.max(scores, dim=-1, keepdim=True)
        p_attn = torch.zeros_like(scores).scatter_(-1, max_indices, 1.0)  # One-hot encoding for max
        return torch.matmul(p_attn, value), p_attn

    raise ValueError(f"Unknown aggregation type: {aggregation}")


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def get_attn_fn(**params):

    def attn_fn_with_args(*args, **kwargs):
        return attention(*args, **kwargs, **params)
    
    return attn_fn_with_args