"""
EXERCISE 2 — MY TURN

Task: change the vectors so crane is more bird-like [0.9, 0.1]
and its neighbors are also bird-like. What happens to crane's
output vector? Does it become more bird-like?

We run the same QKV steps as Exercise 2, but compare both
contexts side-by-side so the contrast is clear.
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("EXERCISE 2 — MY TURN: Bird vs. Machine Context")
print("=" * 60)


def attention(X):
    """Single-head self-attention with Q = K = V = X."""
    Q, K, V = X, X, X
    scores  = Q @ K.T
    weights = F.softmax(scores, dim=-1)
    output  = weights @ V
    return weights, output


# ── Context A: machine-like neighbors (from Exercise 2) ──────
X_machine = torch.tensor([
    [0.7, 0.7],   # crane    — ambiguous
    [0.1, 0.9],   # lifted   — machine-like
    [0.1, 0.9],   # steel    — machine-like
])

# ── Context B: bird-like neighbors (MY TURN) ─────────────────
X_bird = torch.tensor([
    [0.9, 0.1],   # crane    — bird-like
    [0.9, 0.1],   # wings    — bird-like
    [0.8, 0.2],   # feathers — bird-like
])

for label, X in [("MACHINE context", X_machine), ("BIRD context", X_bird)]:
    weights, output = attention(X)

    print(f"\n{'─' * 40}")
    print(f"{label}")
    print(f"{'─' * 40}")
    print(f"  crane  (before) = {X[0].tolist()}")
    print(f"  crane  (after)  = {output[0].tolist()}")
    print(f"\n  Attention weights for crane (row 0):")
    print(f"    {weights[0].tolist()}")
    print(f"    → crane attends most to: word {weights[0].argmax().item()}")

print("""
>>> WHAT TO NOTICE:
    In the MACHINE context, crane started ambiguous [0.7, 0.7].
    After attention its machine dimension (index 1) grew — it shifted
    toward [0.34, 0.82], pulled by "lifted" and "steel".

    In the BIRD context, crane started bird-like [0.9, 0.1].
    After attention its bird dimension (index 0) stayed dominant —
    it shifted toward something like [0.87, 0.13], reinforced by
    "wings" and "feathers".

    Same word "crane", opposite outcomes — purely because of context.
    This is exactly the static problem that attention solves:
    the output vector is shaped by the neighborhood, not just the word.
""")
