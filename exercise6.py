"""
EXERCISE 6 — Learned QKV Projections

In exercises 2 and 3 we set Q = K = V = X directly.
That was a useful shortcut for learning the mechanics, but it has
a fundamental limitation: every word asks the SAME question it
answers and contributes the same content it queries for.

In a real model, Q, K, and V are SEPARATE learned projections:

  Q = X @ Wq.T   "what am I looking for?"   — learned to ask
  K = X @ Wk.T   "what do I have?"          — learned to advertise
  V = X @ Wv.T   "what will I give you?"    — learned to contribute

Each weight matrix (Wq, Wk, Wv) is trained independently, so the
model can learn to ask very different questions than it answers,
and contribute richer content than it queries for.

We use the same "crane lifted steel" setup as Exercise 2 so the
difference between Q=K=V=X and learned projections is clear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 6 — Learned QKV Projections")
print("=" * 60)

torch.manual_seed(42)

# Same input as Exercise 2 — (tokens=3, embed_dim=4)
# We use 4 dims instead of 2 so the projection matrices have
# something interesting to do.
X = torch.tensor([
    [0.7, 0.7, 0.0, 0.0],   # crane  — ambiguous
    [0.1, 0.9, 0.1, 0.0],   # lifted — machine-like
    [0.1, 0.9, 0.0, 0.1],   # steel  — machine-like
])

T, C = X.shape   # 3 tokens, 4 dims
dk   = C         # key dimension (same size for simplicity)

print(f"\nInput X  (shape {list(X.shape)}):")
for name, row in zip(["crane ", "lifted", "steel "], X):
    print(f"  {name} = {row.tolist()}")

# ── PART A: The old way — Q = K = V = X ──────────────────────
print("\n" + "─" * 40)
print("PART A — No projections (Q = K = V = X)")
print("─" * 40)

Q_old = X
K_old = X
V_old = X

scores_old  = Q_old @ K_old.T / math.sqrt(dk)
weights_old = F.softmax(scores_old, dim=-1)
output_old  = weights_old @ V_old

print("\nQ == K == V == X?", torch.equal(Q_old, K_old) and torch.equal(K_old, V_old))
print(f"\nAttention weights:\n{weights_old.round(decimals=3)}")
print(f"\ncrane (before): {X[0].tolist()}")
print(f"crane (after) : {output_old[0].tolist()}")

# ── PART B: Learned projections — separate Wq, Wk, Wv ────────
print("\n" + "─" * 40)
print("PART B — Learned projections (separate Wq, Wk, Wv)")
print("─" * 40)

# Each is an independent nn.Linear (no bias, as in nanoGPT)
# Input dim = C, output dim = dk (here both = 4)
Wq = nn.Linear(C, dk, bias=False)
Wk = nn.Linear(C, dk, bias=False)
Wv = nn.Linear(C, dk, bias=False)

Q_new = Wq(X)   # shape: (3, 4)
K_new = Wk(X)   # shape: (3, 4)
V_new = Wv(X)   # shape: (3, 4)

print("\nAfter projection — are Q, K, V still equal to X?")
print(f"  Q == X? {torch.allclose(Q_new, X)}")
print(f"  K == X? {torch.allclose(K_new, X)}")
print(f"  V == X? {torch.allclose(V_new, X)}")
print(f"  Q == K? {torch.allclose(Q_new, K_new)}")
print(f"  Q == V? {torch.allclose(Q_new, V_new)}")

print(f"\nX  [crane]: {X[0].tolist()}")
print(f"Q  [crane]: {Q_new[0].detach().tolist()}")
print(f"K  [crane]: {K_new[0].detach().tolist()}")
print(f"V  [crane]: {V_new[0].detach().tolist()}")

scores_new  = Q_new @ K_new.T / math.sqrt(dk)
weights_new = F.softmax(scores_new, dim=-1)
output_new  = weights_new @ V_new

print(f"\nAttention weights:\n{weights_new.round(decimals=3)}")
print(f"\ncrane (before):           {X[0].tolist()}")
print(f"crane (after, no proj):   {output_old[0].tolist()}")
print(f"crane (after, learned):   {output_new[0].detach().tolist()}")

print("""
>>> WHAT TO NOTICE:
    PART A: Q, K, V are identical copies of X. Every word both asks
    and answers the same question, and contributes exactly what it
    queries for. The attention weights are symmetric where inputs
    are similar.

    PART B: After projection, Q, K, and V are all different from X
    and from each other — even though they came from the same input.
    Wq, Wk, Wv each apply a different linear transformation, so:
      - Q captures what each word is SEARCHING for
      - K captures what each word is ADVERTISING
      - V captures what each word will CONTRIBUTE if selected

    This separation is what gives the model its expressive power.
    During training, gradients flow back through Wq, Wk, Wv
    independently — they specialise for different roles.
    Q=K=V=X can never learn this distinction.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Inspect the weight matrices Wq.weight, Wk.weight, Wv.weight.
# They start as random — but notice they are already different
# from each other. After training on real data they specialise further.
#
# print("Wq weights:\n", Wq.weight.data.round(decimals=3))
# print("Wk weights:\n", Wk.weight.data.round(decimals=3))
# print("Wv weights:\n", Wv.weight.data.round(decimals=3))
# print("Are Wq and Wk equal?", torch.equal(Wq.weight, Wk.weight))
# ───────────────────────────────────────────────────────────
