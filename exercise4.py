"""
EXERCISE 4 — The Causal Mask (no cheating!)

Zachary called this the "don't look ahead" problem.
GPT generates one word at a time. When predicting word 3,
it must not be allowed to SEE word 3 — that's cheating.

The fix: set future scores to -infinity BEFORE softmax.
e^(-inf) = 0, so those positions get zero attention weight.

We use torch.tril() (lower triangular matrix) to build the mask:
  1 = "allowed to look"
  0 = "blocked (future)"

Builds directly on Exercise 3's scaled dot-product attention.
"""

import torch
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 4 — The Causal Mask")
print("=" * 60)

torch.manual_seed(0)

# Same input as Exercise 3: (batch=1, tokens=3, channels=2)
X_batched = torch.tensor([[[1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0]]])

B, T, C = X_batched.shape
Q = X_batched
K = X_batched
V = X_batched

dk = C

# Reproduce Exercise 3's scaled scores as our starting point
scaled_scores = (Q @ K.transpose(-2, -1)) / math.sqrt(dk)  # (1, 3, 3)

# ── Step 1: Create the mask ──────────────────────────────────
mask = torch.tril(torch.ones(T, T))
print("\nStep 1 — Lower-triangular mask (1=allowed, 0=blocked):")
print(mask)
print("""
  Token 1 can only see: itself
  Token 2 can see:      itself + token 1
  Token 3 can see:      itself + token 1 + token 2
""")

# ── Step 2: Apply the mask to our scaled scores ──────────────
# We use masked_fill: wherever mask==0, replace score with -inf
masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
print("Step 2 — Scores after masking (upper triangle = -inf):")
print(masked_scores)

# ── Step 3: Softmax (now safe — future = zero weight) ────────
causal_weights = F.softmax(masked_scores, dim=-1)
print("\nStep 3 — Causal attention weights:")
print(causal_weights.round(decimals=3))

# ── Step 4: Aggregate ────────────────────────────────────────
output = causal_weights @ V  # (1, 3, 2)
print("\nStep 4 — Final output (causal context-aware):")
print(output)

print("""
>>> WHAT TO NOTICE:
    Row 0 (token 1): only pays attention to itself — can't see future
    Row 1 (token 2): can attend to token 1 and itself
    Row 2 (token 3): can attend to all tokens (the whole past)

    The upper-right triangle is all zeros. The model can't cheat.
    This is how every GPT-style model works under the hood.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# What if you had a 5-token sentence? Build the mask for T=5
# and print it. Which tokens can token 4 see?
#
# T5 = 5
# mask5 = torch.tril(torch.ones(T5, T5))
# print(mask5)
# ───────────────────────────────────────────────────────────
