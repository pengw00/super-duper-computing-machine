"""
03 - Minimal Transformer Block
================================
A Transformer encoder block (used in BERT, ViT, …) consists of:

    1. Multi-Head Self-Attention  (with residual + LayerNorm)
    2. Feed-Forward Network       (with residual + LayerNorm)

A decoder-only block (used in GPT) additionally uses a causal mask so
that each position can only attend to earlier positions.

This file builds both and stacks them into a small language model that
can generate text character by character as a demonstration.

Run:
    python llm/03_transformer.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# Re-use the attention helpers from 02_attention.py inline so
# this file is self-contained.
# ---------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _split(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.n_heads, self.d_head).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        b = q.size(0)
        Q, K, V = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        if mask is not None:
            mask = mask.unsqueeze(0).unsqueeze(0)
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(b, -1, self.d_model)
        return self.W_o(out)


# ------------------------------------------------------------------
# 1. Encoder block (bidirectional, no causal mask)
# ------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """Pre-LN Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn   = MultiHeadAttention(d_model, n_heads)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention sub-layer
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        # Feed-forward sub-layer
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ------------------------------------------------------------------
# 2. Decoder block (causal, GPT-style)
# ------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Pre-LN Transformer decoder block with causal self-attention."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadAttention(d_model, n_heads)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        seq = x.size(1)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=1)
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ------------------------------------------------------------------
# 3. Tiny character-level GPT
# ------------------------------------------------------------------

class TinyGPT(nn.Module):
    """
    A minimal decoder-only language model that operates on characters.
    Useful for illustrating end-to-end training without large datasets.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.blocks    = nn.Sequential(*[DecoderBlock(d_model, n_heads, d_model * 4, dropout)
                                         for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (batch, seq_len)  integer token ids
        b, t = idx.shape
        positions = torch.arange(t, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)  # (b, t, d_model)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x)   # (b, t, vocab_size) — logits


# ------------------------------------------------------------------
# 4. Quick smoke test
# ------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Encoder block ---
    block = EncoderBlock(d_model=64, n_heads=4, ffn_dim=256)
    x = torch.randn(2, 10, 64)
    out = block(x)
    print(f"EncoderBlock output: {out.shape}")   # (2, 10, 64)

    # --- Decoder block ---
    dec = DecoderBlock(d_model=64, n_heads=4, ffn_dim=256)
    out2 = dec(x)
    print(f"DecoderBlock output: {out2.shape}")  # (2, 10, 64)

    # --- TinyGPT forward pass ---
    VOCAB = 65   # 'a'-'z', 'A'-'Z', digits, punctuation (~Shakespeare)
    model = TinyGPT(vocab_size=VOCAB).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"TinyGPT params: {params:,}")

    ids  = torch.randint(0, VOCAB, (4, 16)).to(device)  # batch=4, seq=16
    logits = model(ids)
    print(f"TinyGPT logits: {logits.shape}")   # (4, 16, 65)

    # --- One gradient update ---
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    targets   = ids[:, 1:].contiguous()          # predict next token
    preds     = logits[:, :-1, :].contiguous()
    loss = F.cross_entropy(preds.view(-1, VOCAB), targets.view(-1))
    loss.backward()
    optimiser.step()
    print(f"Loss after one step: {loss.item():.4f}")

    # --- Greedy generation ---
    model.eval()
    context = torch.zeros(1, 1, dtype=torch.long).to(device)  # start token = 0
    generated = []
    with torch.no_grad():
        for _ in range(20):
            logits = model(context[:, -16:])      # cap context to max_len
            next_id = logits[:, -1, :].argmax(-1)
            context = torch.cat([context, next_id.unsqueeze(0)], dim=1)
            generated.append(next_id.item())
    print("Generated token ids (first 20):", generated)


if __name__ == "__main__":
    main()
