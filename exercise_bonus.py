"""
BONUS — The Full CausalSelfAttention Module (nn.Module)

This is the "production-ready code" Zachary showed at the end —
the code that looked like a nightmare before the video, but
now should make complete sense.

Everything you built across exercises 1–5 is in here:
  - c_attn        → the fused QKV projection       (Exercise 5 Step 1)
  - view + transpose → split into heads             (Exercise 5 Step 1b)
  - scaled dot-product → Q @ Kᵀ / √dk              (Exercise 3)
  - causal mask   → masked_fill + softmax           (Exercise 4)
  - weights @ V   → aggregate values                (Exercise 2 Step 3)
  - transpose + view + c_proj → merge and project   (Exercise 5 Steps 5–6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("BONUS — Full CausalSelfAttention Module")
print("=" * 60)


class CausalSelfAttention(nn.Module):
    """
    Minimal causal multi-head self-attention.
    Every line maps to something built in exercises 1–5.
    """
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

        self.n_head  = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        # Exercise 5 Step 1: single projection for Q, K, V together
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)

        # Exercise 5 Step 6: final output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        # ── Project to Q, K, V ──────────────────────────────
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(C, dim=-1)

        # ── Split into heads (Exercise 5 trick) ─────────────
        def split_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # ── Scaled dot-product attention (Exercise 3) ────────
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)

        # ── Causal mask (Exercise 4) ─────────────────────────
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        # ── Aggregate values (Exercise 2 Step 3) ─────────────
        out = weights @ V

        # ── Merge heads back ─────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # ── Final projection ──────────────────────────────────
        return self.c_proj(out)


# ── Test it ──────────────────────────────────────────────────
torch.manual_seed(0)
model = CausalSelfAttention(n_embed=8, n_head=2)
x_test = torch.randn(1, 3, 8)   # (batch=1, tokens=3, embed=8)
y_test = model(x_test)

print(f"\nInput shape:  {list(x_test.shape)}")
print(f"Output shape: {list(y_test.shape)}")
print("\nInput (first token): ", x_test[0, 0].detach().round(decimals=3).tolist())
print("Output (first token):", y_test[0, 0].detach().round(decimals=3).tolist())

print("""
>>> WHAT TO NOTICE:
    Input and output shapes are identical.
    But the output values are different — each token has absorbed
    information from its causal context (past tokens only).

    This is exactly the module Zachary showed at the very end.
    The "nightmare" code from the beginning of the video.
    Now it's just a map of things you built yourself, step by step.

Congrats — attention has clicked!
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Try stacking two CausalSelfAttention blocks — just like a real transformer.
# The output shape of block 1 feeds directly into block 2.
#
# block1 = CausalSelfAttention(n_embed=8, n_head=2)
# block2 = CausalSelfAttention(n_embed=8, n_head=2)
# x = torch.randn(1, 3, 8)
# x = block1(x)
# x = block2(x)
# print(x.shape)  # still (1, 3, 8) — stackable by design
# ───────────────────────────────────────────────────────────
