"""
EXERCISE 5 — Multi-Head Attention (the parallel conversations)

Zachary's insight: one "conversation" (one attention head) isn't
enough to capture all the richness of language.

Multi-head attention runs MANY conversations at once:
  Head 1 → might track verb-object relationships
  Head 2 → might resolve pronoun references
  Head 3 → might handle long-distance dependencies

The clever trick: we don't run separate Q, K, V for each head.
Instead, we RESHAPE the embedding dimension (C) into (n_heads, head_dim)
using .view() and .transpose(), then compute ALL heads in parallel.

Example from the video (GPT-2 scale):
  C = 768 embedding dimensions
  n_heads = 12
  head_dim = 768 / 12 = 64  (each head gets 64 dimensions)

We'll use a tiny version so the shapes are easy to follow.

Builds directly on Exercise 4's scaled dot-product + causal mask.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 5 — Multi-Head Attention (the parallel conversations)")
print("=" * 60)

torch.manual_seed(42)

B        = 1   # batch size
T        = 3   # sequence length (tokens)
C        = 8   # embedding dimension (8 instead of 768 to keep it readable)
n_head   = 2   # number of attention heads
head_dim = C // n_head  # = 4  (each head gets 4 dimensions)

print(f"\nSetup: B={B}, T={T}, C={C}, n_head={n_head}, head_dim={head_dim}")

# ── Step 1a: Compute Q, K, V with a learned linear layer ────
# In a real model Q, K, V each have their OWN weight matrix.
# Here we combine all three into one big projection (3*C output)
# then split — this is how GPT-2 / nanoGPT does it.
c_attn = nn.Linear(C, 3 * C, bias=False)

X_in = torch.randn(B, T, C)
qkv  = c_attn(X_in)             # shape: (1, 3, 24)
Q, K, V = qkv.split(C, dim=-1)  # each: (1, 3, 8)

print(f"\nAfter linear projection: Q shape = {list(Q.shape)}")

# ── Step 1b: Reshape into heads ──────────────────────────────
# From (B, T, C) → (B, T, n_head, head_dim) → (B, n_head, T, head_dim)
#
# The .transpose(1, 2) is the critical trick Zachary highlighted:
# it brings n_head forward so PyTorch treats it like a batch dimension,
# letting it compute ALL heads in one matrix multiply.

Q = Q.view(B, T, n_head, head_dim).transpose(1, 2)  # (1, 2, 3, 4)
K = K.view(B, T, n_head, head_dim).transpose(1, 2)
V = V.view(B, T, n_head, head_dim).transpose(1, 2)

print(f"After reshape + transpose: Q shape = {list(Q.shape)}")
print("  Meaning: (batch=1, heads=2, tokens=3, head_dim=4)")

# ── Step 2: Scaled dot-product attention (all heads at once) ─
scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)  # (1, 2, 3, 3)

# ── Step 3: Causal mask (Exercise 4) ─────────────────────────
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))

weights = F.softmax(scores, dim=-1)
print(f"\nAttention weights shape: {list(weights.shape)}")
print("  (1 batch × 2 heads × 3×3 attention matrix)")

# ── Step 4: Aggregate values ─────────────────────────────────
out = weights @ V       # (1, 2, 3, 4)
print(f"Output per head shape: {list(out.shape)}")

# ── Step 5: Merge heads back ─────────────────────────────────
# Reverse the transpose, then merge (n_head, head_dim) → C
out = out.transpose(1, 2).contiguous().view(B, T, C)  # (1, 3, 8)
print(f"After merging heads: {list(out.shape)}")

# ── Step 6: Final linear projection ──────────────────────────
c_proj = nn.Linear(C, C, bias=False)
final_output = c_proj(out)   # (1, 3, 8)
print(f"Final output shape: {list(final_output.shape)}")

print(f"""
>>> WHAT TO NOTICE:
    The shape journey for multi-head:
      Input X:                {list(X_in.shape)}
      After Q/K/V projection: {list(Q.shape)}  (B, n_head, T, head_dim)
      Attention weights:      {list(weights.shape)}
      After merging heads:    {list(out.shape)}     ← back to (B, T, C)
      Final output:           {list(final_output.shape)}     ← same as input!

    The input and output shapes match. This is what allows transformers
    to be stacked — the output of one block feeds cleanly into the next.
    GPT-3 does this 96 times. Same block, stacked 96 layers deep.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Try changing n_head to 4 (and keep C=8, so head_dim=2).
# Does the output shape change? What about the weights shape?
# This mirrors going from 4-head to 8-head attention in real models.
# ───────────────────────────────────────────────────────────
