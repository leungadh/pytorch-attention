"""
EXERCISE 5 — MY TURN

Task: change n_head to 4 (keep C=8, so head_dim=2).
Does the output shape change? What about the weights shape?
This mirrors going from 4-head to 8-head attention in real models.

We run both configurations side-by-side so every shape difference
(and non-difference) is visible at a glance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 5 — MY TURN: 2-Head vs. 4-Head Attention")
print("=" * 60)

torch.manual_seed(42)

B = 1
T = 3
C = 8   # fixed — must be divisible by n_head

X_in = torch.randn(B, T, C)


def multi_head_attention(X, n_head):
    B, T, C = X.shape
    head_dim = C // n_head

    c_attn = nn.Linear(C, 3 * C, bias=False)
    c_proj = nn.Linear(C, C,     bias=False)

    # QKV projection + split
    qkv    = c_attn(X)
    Q, K, V = qkv.split(C, dim=-1)

    # Reshape into heads: (B, T, C) → (B, n_head, T, head_dim)
    Q = Q.view(B, T, n_head, head_dim).transpose(1, 2)
    K = K.view(B, T, n_head, head_dim).transpose(1, 2)
    V = V.view(B, T, n_head, head_dim).transpose(1, 2)

    # Scaled dot-product + causal mask
    scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
    mask   = torch.tril(torch.ones(T, T))
    scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)

    # Aggregate + merge heads
    out = weights @ V
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    return weights, out, c_proj(out)


for n_head in [2, 4]:
    head_dim = C // n_head
    torch.manual_seed(42)   # same seed so weights are comparable
    weights, out_merged, final = multi_head_attention(X_in, n_head)

    print(f"\n{'─' * 40}")
    print(f"n_head={n_head}  →  head_dim = C // n_head = {C} // {n_head} = {head_dim}")
    print(f"{'─' * 40}")
    print(f"  Q/K/V per head shape:  (B={B}, n_head={n_head}, T={T}, head_dim={head_dim})")
    print(f"  Attention weights:     {list(weights.shape)}")
    print(f"  After merging heads:   {list(out_merged.shape)}")
    print(f"  Final output:          {list(final.shape)}")

print("""
>>> WHAT TO NOTICE:
    The OUTPUT shape never changes — it is always (B, T, C) = (1, 3, 8).
    This is what makes transformers stackable: the block's interface is
    fixed regardless of how many heads are used internally.

    What DOES change is the attention weights shape:
      2 heads → (1, 2, 3, 3)  — 2 separate 3×3 attention matrices
      4 heads → (1, 4, 3, 3)  — 4 separate 3×3 attention matrices

    And the per-head capacity:
      2 heads → head_dim=4  — each head works in a 4-dimensional subspace
      4 heads → head_dim=2  — each head works in a 2-dimensional subspace

    More heads = more parallel "conversations", but each conversation
    operates in a smaller space. Real models balance these:
      GPT-2 small:  12 heads, head_dim=64  (C=768)
      GPT-3:        96 heads, head_dim=128 (C=12288)

    The total compute stays roughly the same — you're slicing C into
    more pieces, not adding more dimensions.
""")
