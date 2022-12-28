# -*- coding: utf-8 -*-
import torch
from torch import nn

from lightning_ie.nn import INF, SinusoidalEmbedding
from lightning_ie.util import apply_RoPE_position_embeddings


class GlobalPointer(nn.Module):
    """ GlobalPointer
        https://spaces.ac.cn/archives/8373
    """

    def __init__(self, hidden_size, head_hidden_size, num_heads, mask_tril=False, use_pe=True, max_pe_len=512):

        super().__init__()
        self.num_heads = num_heads
        self.head_hidden_size = head_hidden_size
        self.hidden_size = hidden_size

        self.mask_tril = mask_tril
        self.use_pe = use_pe
        self.max_pe_len = max_pe_len

        self.linear = nn.Linear(self.hidden_size, self.num_heads * self.head_hidden_size * 2)

        if self.use_pe:
            self.position_embed = SinusoidalEmbedding(self.max_pe_len, self.head_hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.LongTensor = None):
        x = self.linear(hidden)
        x = torch.split(x, self.head_hidden_size * 2, dim=-1)  # [B, L, D*2] * K
        x = torch.stack(x, dim=-2)  # B, L, K, D*2
        qw, kw = x[..., :self.head_hidden_size], x[..., self.head_hidden_size:]  # B, L, K, D

        if self.use_pe:
            # B, L, K, D
            pos = self.position_embed(torch.arange(x.size(1), dtype=torch.long, device=hidden.device)[None])
            qw, kw = apply_RoPE_position_embeddings(pos, qw, kw)

        x = torch.einsum('bmkd,bnkd->bkmn', qw, kw) / self.head_hidden_size ** 0.5  # scale

        if attention_mask is not None:
            mask = torch.einsum('bm,bn->bmn', attention_mask, attention_mask)[:, None, :, :]
            x = x * mask - INF * (1 - mask)

        if self.mask_tril:
            mask = torch.tril(torch.ones_like(x), diagonal=-1)
            x = x * (1 - mask) - INF * mask

        return x


class EfficientGlobalPointer(nn.Module):
    """ EfficientGlobalPointer
        https://spaces.ac.cn/archives/8877
    """

    def __init__(self, hidden_size, head_hidden_size, num_heads, mask_tril=False, use_pe=True, max_pe_len=512):

        super().__init__()
        self.num_heads = num_heads
        self.head_hidden_size = head_hidden_size
        self.hidden_size = hidden_size

        self.mask_tril = mask_tril
        self.use_pe = use_pe
        self.max_pe_len = max_pe_len

        self.linear_p = nn.Linear(self.hidden_size, self.head_hidden_size * 2)
        self.linear_q = nn.Linear(self.head_hidden_size * 2, self.num_heads * 2)

        if self.use_pe:
            self.position_embed = SinusoidalEmbedding(self.max_pe_len, self.head_hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.LongTensor = None):
        x = self.linear_p(hidden)
        qw, kw = x[..., :self.head_hidden_size], x[..., self.head_hidden_size:]

        if self.use_pe:
            # B, L, K, D
            pos = self.position_embed(torch.arange(x.size(1), dtype=torch.long, device=hidden.device)[None])
            qw, kw = apply_RoPE_position_embeddings(pos, qw, kw)

        logits = torch.einsum('bmd,bnd->bmn', qw, kw) / self.head_hidden_size ** 0.5
        bias = torch.einsum('bnk->bkn', self.linear_q(x)) / 2
        x = logits[:, None] + bias[:, :self.num_heads, None] + bias[:, self.num_heads:, :, None]

        if attention_mask is not None:
            mask = torch.einsum('bm,bn->bmn', attention_mask, attention_mask)[:, None, :, :]
            x = x * mask - INF * (1 - mask)

        if self.mask_tril:
            mask = torch.tril(torch.ones_like(x), diagonal=-1)
            x = x * (1 - mask) - INF * mask

        return x
