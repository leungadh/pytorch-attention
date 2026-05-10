"""
BONUS — MY TURN

Task: stack two CausalSelfAttention blocks — just like a real transformer.
The output shape of block 1 feeds directly into block 2.

We go further than the minimal hint: stack up to 6 blocks and track
how the token representations change at each layer, then ground it
in real transformer depths (GPT-2, GPT-3).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head  = n_head
        self.head_dim = n_embed // n_head
        self.c_attn  = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.c_proj  = nn.Linear(n_embed, n_embed,     bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(C, dim=-1)

        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
        scores  = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask    = torch.tril(torch.ones(T, T, device=x.device))
        scores  = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out     = weights @ V
        out     = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


print("=" * 60)
print("BONUS — MY TURN: Stacking Attention Blocks")
print("=" * 60)

torch.manual_seed(0)

N_EMBED = 8
N_HEAD  = 2
N_LAYERS = 6

x = torch.randn(1, 3, N_EMBED)   # (batch=1, tokens=3, embed=8)

print(f"\nInput shape: {list(x.shape)}")
print(f"Stacking {N_LAYERS} CausalSelfAttention blocks...\n")

# ── Stack N_LAYERS blocks and track how representations drift ─
prev = x.clone()
for i in range(1, N_LAYERS + 1):
    torch.manual_seed(i)   # different weights per layer
    block = CausalSelfAttention(n_embed=N_EMBED, n_head=N_HEAD)
    x = block(x)

    # L2 distance from the previous layer's output — how much did it change?
    delta = (x - prev).norm().item()
    print(f"  Layer {i}: output shape = {list(x.shape)}  |  change from prev layer = {delta:.4f}")
    prev = x.clone()

print(f"\nFinal output shape: {list(x.shape)}  ← identical to input shape")

print("""
>>> WHAT TO NOTICE:
    The shape never changes across all 6 layers — always (1, 3, 8).
    This is the design contract that makes stacking possible:
    every block takes (B, T, C) and returns (B, T, C).

    The "change from prev layer" shows that each block is genuinely
    transforming the representations — not just passing them through.
    By layer 6 the vectors are far from where they started, even
    though the shape looks identical.

    This is how real transformers scale:
      GPT-2 small : 12 layers,  C=768,   12 heads
      GPT-2 large : 36 layers,  C=1280,  20 heads
      GPT-3       : 96 layers,  C=12288, 96 heads

    Same block blueprint, stacked deeper. The depth is where the
    representational power comes from.
""")
