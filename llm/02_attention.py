"""
02 - Scaled Dot-Product Attention
==================================
Attention is the core operation in every Transformer / LLM.
Given queries Q, keys K, and values V it computes:

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

This file builds up from the bare formula to multi-head attention.

Run:
    python llm/02_attention.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1. Single-head scaled dot-product attention (from scratch)
# ------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        Q: (batch, seq_q, d_k)
        K: (batch, seq_k, d_k)
        V: (batch, seq_k, d_v)
        mask: optional boolean mask — True positions are MASKED OUT (set to -inf)
    Returns:
        output: (batch, seq_q, d_v)
        weights: (batch, seq_q, seq_k)  — attention probabilities
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (batch, seq_q, seq_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    output  = weights @ V                                 # (batch, seq_q, d_v)
    return output, weights


batch, seq_len, d_k, d_v = 2, 5, 16, 32
Q = torch.randn(batch, seq_len, d_k)
K = torch.randn(batch, seq_len, d_k)
V = torch.randn(batch, seq_len, d_v)

out, attn_weights = scaled_dot_product_attention(Q, K, V)
print("Single-head attention output:", out.shape)    # (2, 5, 32)
print("Attention weights:           ", attn_weights.shape)  # (2, 5, 5)

# ------------------------------------------------------------------
# 2. Causal (autoregressive) mask
# ------------------------------------------------------------------
# In decoder-only LLMs (GPT-style), each token can only attend to
# tokens at the SAME or EARLIER positions.

def causal_mask(seq_len: int) -> torch.Tensor:
    """Upper-triangular True mask (positions that should be blocked)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)

mask = causal_mask(seq_len)
print(f"\nCausal mask ({seq_len}x{seq_len}):\n{mask.int()}")

out_causal, _ = scaled_dot_product_attention(Q, K, V, mask=mask)
print("Causal attention output:", out_causal.shape)

# ------------------------------------------------------------------
# 3. Multi-Head Attention
# ------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Split d_model into `n_heads` smaller heads, run attention in
    parallel, then project back.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_head)"""
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = q.size(0)

        Q = self._split_heads(self.W_q(q))   # (batch, n_heads, seq, d_head)
        K = self._split_heads(self.W_k(k))
        V = self._split_heads(self.W_v(v))

        if mask is not None:
            # broadcast over batch and heads: (seq_q, seq_k) -> (1, 1, seq_q, seq_k)
            mask = mask.unsqueeze(0).unsqueeze(0)

        out, _ = scaled_dot_product_attention(Q, K, V, mask)   # (batch, heads, seq, d_head)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous()                 # (batch, seq, heads, d_head)
        out = out.view(batch, -1, self.d_model)                # (batch, seq, d_model)
        return self.W_o(out)


d_model, n_heads = 64, 4
mha = MultiHeadAttention(d_model, n_heads)
x   = torch.randn(2, 10, d_model)   # batch=2, seq=10
out = mha(x, x, x)                  # self-attention
print(f"\nMulti-head attention output: {out.shape}")   # (2, 10, 64)

# With causal mask
mask = causal_mask(10)
out_masked = mha(x, x, x, mask=mask)
print(f"Causal multi-head output:    {out_masked.shape}")
