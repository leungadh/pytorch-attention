"""
EXERCISE 3 — Scaled Dot-Product Attention (the full formula)

The full formula Zachary showed:

  Attention(Q, K, V) = softmax( Q @ Kᵀ / √dk ) @ V

The only new piece vs. Exercise 2 is dividing by √dk.
WHY? As vectors get longer (larger dk), dot products get bigger,
which makes softmax outputs very extreme (close to 0 or 1).
Dividing by √dk keeps the scores in a stable range.

Zachary's setup: T=3 tokens, C=2 dimensions (same as Exercise 2)
but now we add the scaling step explicitly and introduce the
batch dimension (B, T, C).
"""

import torch
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 3 — Scaled Dot-Product Attention (full formula)")
print("=" * 60)

torch.manual_seed(0)

# Batch of 1 sentence, 3 tokens, 2-dimensional embeddings
# Shape: (batch=1, tokens=3, channels=2)
X_batched = torch.tensor([[[1.0, 0.0],
                            [0.0, 1.0],
                            [1.0, 1.0]]])

B, T, C = X_batched.shape  # 1, 3, 2
print(f"\nInput shape: batch={B}, tokens={T}, channels={C}")
print("Input X:\n", X_batched)

# For now: Q = K = V = X (same as Zachary's from-scratch demo)
Q = X_batched
K = X_batched
V = X_batched

dk = C  # the dimension of the key vectors

# ── Full formula in one readable chain ──────────────────────
# Step A: Raw scores
raw_scores = Q @ K.transpose(-2, -1)         # (1, 3, 3)
print("\nStep A — Raw scores Q @ Kᵀ:")
print(raw_scores)

# Step B: Scale  ← the new piece!
scaled_scores = raw_scores / math.sqrt(dk)   # (1, 3, 3)
print(f"\nStep B — Scaled scores (divided by √{dk} = {math.sqrt(dk):.3f}):")
print(scaled_scores)

# Step C: Softmax
weights = F.softmax(scaled_scores, dim=-1)   # (1, 3, 3)
print("\nStep C — Attention weights after softmax:")
print(weights.round(decimals=3))

# Step D: Aggregate
output = weights @ V                          # (1, 3, 2)
print("\nStep D — Final output (context-aware):")
print(output)

print(f"""
>>> WHAT TO NOTICE:
    The shape journey Zachary described:
      Input X:         {list(X_batched.shape)}
      After Q @ Kᵀ:   {list(raw_scores.shape)}   ← explodes to TxT
      After softmax:   {list(weights.shape)}   ← still TxT
      After @ V:       {list(output.shape)}    ← back to original shape!

    The output has the SAME shape as the input, but each vector has
    absorbed context from its neighbors. That's the magic.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# What happens if dk is very large (e.g., 512)?
# Try making X have 512 dimensions (random) and see how raw_scores
# compare to scaled_scores. Does scaling matter more now?
#
# torch.manual_seed(0)
# X_large = torch.randn(1, 3, 512)
# Q, K, V = X_large, X_large, X_large
# dk_large = 512
# raw = Q @ K.transpose(-2, -1)
# scaled = raw / math.sqrt(dk_large)
# print("Max raw score:", raw.max().item())
# print("Max scaled score:", scaled.max().item())
# ───────────────────────────────────────────────────────────
