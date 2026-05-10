"""
EXERCISE 2 — The QKV Conversation (crane lifted steel)

Zachary's mental model: every word plays THREE roles:

  Q (Query)  = "What am I looking for?" — the word's question
  K (Key)    = "What do I have?"        — the word's label
  V (Value)  = "What will I give you?"  — the word's actual content

The three-step process:
  1. SCORE     — Q from one word dot-products with every K
                 (high score = strong match)
  2. NORMALIZE — softmax turns raw scores into percentages
                 (who do I listen to?)
  3. AGGREGATE — weighted average of all V vectors
                 (absorb the information)

Example: "crane lifted steel"
  Is "crane" a bird or a machine?
  It looks at its neighbors and updates itself to be more machine-like.
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("EXERCISE 2 — The QKV Conversation (crane lifted steel)")
print("=" * 60)

# We'll represent each word as a tiny 2D vector.
# Think of the first number as "bird-like" and the second as "machine-like".
#
#   crane  → ambiguous: [0.7, 0.7]   (could be either!)
#   lifted → machine:   [0.1, 0.9]
#   steel  → machine:   [0.1, 0.9]

# Shape: (num_tokens=3, embedding_dim=2)
X = torch.tensor([
    [0.7, 0.7],   # crane  — ambiguous
    [0.1, 0.9],   # lifted — machine-like
    [0.1, 0.9],   # steel  — machine-like
])

print("\nOriginal word vectors (before attention):")
print(f"  crane  = {X[0].tolist()}")
print(f"  lifted = {X[1].tolist()}")
print(f"  steel  = {X[2].tolist()}")

# For simplicity (just like Zachary did in the video),
# we set Q = K = V = X. In a real model these are learned projections.
Q = X
K = X
V = X

# ── STEP 1: SCORE ──────────────────────────────────────────
# Q @ K.T gives us a matrix of dot-product scores.
# Score[i][j] = "how much should word i pay attention to word j?"
#
# Shape: (3, 2) @ (2, 3) → (3, 3)

scores = Q @ K.T   # equivalent to torch.matmul(Q, K.transpose(-2, -1))

print("\nSTEP 1 — Raw scores (Q @ Kᵀ):")
print(scores)
print("  Each row = one word's scores against all words")
print("  Row 0 (crane): how much crane cares about crane, lifted, steel")

# ── STEP 2: NORMALIZE ──────────────────────────────────────
# softmax converts raw scores to probabilities (they sum to 1).
# High score → high attention weight → "I'll listen to you a lot"

weights = F.softmax(scores, dim=-1)  # dim=-1 means "along the last axis"

print("\nSTEP 2 — Attention weights after softmax:")
print(weights.round(decimals=3))
print("  Each row sums to 1.0 (check below):")
print(" ", weights.sum(dim=-1).tolist())

# ── STEP 3: AGGREGATE ──────────────────────────────────────
# The new vector for each word = weighted average of ALL value vectors.
# crane's new vector = 0.X * V[crane] + 0.X * V[lifted] + 0.X * V[steel]

output = weights @ V   # shape: (3, 3) @ (3, 2) → (3, 2)

print("\nSTEP 3 — Output after attention (context-aware vectors):")
print(f"  crane  (before) = {X[0].tolist()}")
print(f"  crane  (after)  = {output[0].tolist()}")
print(f"\n  lifted (before) = {X[1].tolist()}")
print(f"  lifted (after)  = {output[1].tolist()}")
print(f"\n  steel  (before) = {X[2].tolist()}")
print(f"  steel  (after)  = {output[2].tolist()}")

print("""
>>> WHAT TO NOTICE:
    Look at crane's vector. It started at [0.7, 0.7] — perfectly ambiguous.
    After attention, the second number (machine-like) grew larger.
    crane absorbed information from lifted and steel, which are strongly
    machine-like. The ambiguity is resolved through the conversation!
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Change the vectors so crane is more bird-like: [0.9, 0.1]
# and its neighbors are also bird-like: [0.9, 0.1]
# What happens to crane's output vector? Does it become more bird-like?
#
# X_bird = torch.tensor([
#     [0.9, 0.1],  # crane — bird-like
#     [0.9, 0.1],  # wings — bird-like
#     [0.8, 0.2],  # feathers — bird-like
# ])
# Q, K, V = X_bird, X_bird, X_bird
# scores  = Q @ K.T
# weights = F.softmax(scores, dim=-1)
# output  = weights @ V
# print(f"crane (before) = {X_bird[0].tolist()}")
# print(f"crane (after)  = {output[0].tolist()}")
# ───────────────────────────────────────────────────────────
