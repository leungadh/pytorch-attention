"""
EXERCISE 4 — MY TURN

Task: what if you had a 5-token sentence? Build the mask for T=5
and print it. Which tokens can token 4 see?

We build the mask, annotate exactly what each token can see,
then run the full causal attention pass on a 5-token input
to confirm the upper triangle is truly zeroed out.
"""

import torch
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 4 — MY TURN: Causal Mask for a 5-Token Sentence")
print("=" * 60)

T = 5

# ── Step 1: Build the mask ────────────────────────────────────
mask = torch.tril(torch.ones(T, T))

print("\nStep 1 — Lower-triangular mask for T=5:")
print(mask)

print("\nWhat each token can attend to:")
for i in range(T):
    allowed = [str(j) for j in range(T) if mask[i, j] == 1]
    print(f"  Token {i} can see: tokens {', '.join(allowed)}")

print(f"\n  → Token 4 (the last token) can see ALL {T} tokens: 0, 1, 2, 3, 4")
print(  "    Token 0 (the first token) can only see itself.")

# ── Step 2: Full causal attention on a 5-token input ─────────
torch.manual_seed(0)

X = torch.randn(1, T, 4)   # (batch=1, tokens=5, channels=4)
Q, K, V = X, X, X
dk = X.shape[-1]

raw_scores    = Q @ K.transpose(-2, -1)
scaled_scores = raw_scores / math.sqrt(dk)
masked_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))
weights       = F.softmax(masked_scores, dim=-1)
output        = weights @ V

print("\nStep 2 — Causal attention weights (5×5):")
print(weights.squeeze(0).round(decimals=3))

print("\nStep 3 — Upper triangle is all zeros?", weights[0].triu(diagonal=1).sum().item() == 0.0)

print("""
>>> WHAT TO NOTICE:
    The mask scales naturally to any sequence length — torch.tril()
    always produces the right lower-triangular shape for T tokens.

    Row 0: [1, 0, 0, 0, 0] — token 0 attends only to itself (weight=1.0)
    Row 1: [*, *, 0, 0, 0] — token 1 splits attention across tokens 0–1
    Row 4: [*, *, *, *, *] — token 4 can attend to the full history

    The upper-right triangle stays exactly zero after softmax.
    No future token leaks through, regardless of sequence length.
""")
