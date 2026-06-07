"""
01 - Token Embeddings
=====================
Large language models represent tokens (words / sub-words) as dense
vectors called *embeddings*.  PyTorch provides `nn.Embedding` — a
learnable lookup table indexed by integer token IDs.

Run:
    python llm/01_embeddings.py
"""

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# 1. nn.Embedding basics
# ------------------------------------------------------------------
VOCAB_SIZE  = 1000   # number of unique tokens
EMBED_DIM   = 64     # dimension of each embedding vector

embedding = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
print(f"Embedding table shape: {embedding.weight.shape}")  # (1000, 64)

# Look up embeddings for a batch of token IDs
token_ids = torch.tensor([[1, 5, 3, 2],    # sentence 1
                           [7, 0, 4, 9]])   # sentence 2
embedded = embedding(token_ids)
print(f"Input token_ids shape:  {token_ids.shape}")   # (2, 4)
print(f"Embedded output shape:  {embedded.shape}")    # (2, 4, 64)

# ------------------------------------------------------------------
# 2. Positional encoding (sinusoidal, as in "Attention is All You Need")
# ------------------------------------------------------------------

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """
    Returns a (1, seq_len, d_model) tensor of positional encodings.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pos = torch.arange(seq_len).unsqueeze(1)          # (seq_len, 1)
    i   = torch.arange(0, d_model, 2)                  # (d_model/2,)
    div = torch.pow(10000.0, i / d_model)               # denominator

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos / div)
    pe[:, 1::2] = torch.cos(pos / div)
    return pe.unsqueeze(0)   # (1, seq_len, d_model)


SEQ_LEN = 20
D_MODEL = 64
pe = sinusoidal_positional_encoding(SEQ_LEN, D_MODEL)
print(f"\nPositional encoding shape: {pe.shape}")   # (1, 20, 64)

# ------------------------------------------------------------------
# 3. Combining token embeddings + positional encodings
# ------------------------------------------------------------------

class TokenEmbedder(nn.Module):
    """Token embedding + sinusoidal positional encoding."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer(
            "pe", sinusoidal_positional_encoding(max_len, d_model)
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)
        seq_len = token_ids.size(1)
        x = self.token_emb(token_ids)   # (batch, seq_len, d_model)
        x = x + self.pe[:, :seq_len, :] # add positional info
        return x


embedder = TokenEmbedder(vocab_size=1000, d_model=64)
ids = torch.randint(0, 1000, (2, 10))   # batch=2, seq_len=10
out = embedder(ids)
print(f"\nTokenEmbedder output shape: {out.shape}")  # (2, 10, 64)

# ------------------------------------------------------------------
# 4. Similarity between embeddings (cosine similarity)
# ------------------------------------------------------------------
v1 = embedding(torch.tensor(42))
v2 = embedding(torch.tensor(42))   # same token → identical
v3 = embedding(torch.tensor(7))    # different token

cos = nn.CosineSimilarity(dim=0)
print(f"\ncos(v1, v2) same token:      {cos(v1, v2).item():.4f}")   # ≈ 1.0
print(f"cos(v1, v3) different token: {cos(v1, v3).item():.4f}")    # random init → small
