"""
EXERCISE 6 — MY TURN

Task: inspect Wq.weight, Wk.weight, Wv.weight.
They start random — but notice they are already different from each other.
After training on real data they specialise further.

We go further than just printing: we run one gradient step and show
that Wq, Wk, Wv receive DIFFERENT gradient signals and update differently.
This is the concrete proof of why three separate matrices matter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 6 — MY TURN: Inspecting and Training Wq, Wk, Wv")
print("=" * 60)

torch.manual_seed(42)

C  = 4
dk = 4

Wq = nn.Linear(C, dk, bias=False)
Wk = nn.Linear(C, dk, bias=False)
Wv = nn.Linear(C, dk, bias=False)

# ── PART A: Inspect the initial random weights ────────────────
print("\nPART A — Initial weight matrices (random, before training)")
print("─" * 40)
print(f"Wq:\n{Wq.weight.data.round(decimals=3)}")
print(f"\nWk:\n{Wk.weight.data.round(decimals=3)}")
print(f"\nWv:\n{Wv.weight.data.round(decimals=3)}")

print(f"\nAre Wq == Wk? {torch.equal(Wq.weight, Wk.weight)}")
print(f"Are Wq == Wv? {torch.equal(Wq.weight, Wv.weight)}")
print(f"Are Wk == Wv? {torch.equal(Wk.weight, Wv.weight)}")

print("""
>>> WHAT TO NOTICE:
    All three matrices are initialised from the same distribution
    (Kaiming uniform by default in nn.Linear), but each draws
    independently — so they start different and will only diverge
    further during training.
""")

# ── PART B: One gradient step — do the weights update equally? ─
print("PART B — One gradient step: do Wq, Wk, Wv receive equal gradients?")
print("─" * 40)

torch.manual_seed(42)

X = torch.tensor([
    [0.7, 0.7, 0.0, 0.0],
    [0.1, 0.9, 0.1, 0.0],
    [0.1, 0.9, 0.0, 0.1],
])

# Re-initialise with fresh identical seeds so we can compare before/after
Wq = nn.Linear(C, dk, bias=False)
Wk = nn.Linear(C, dk, bias=False)
Wv = nn.Linear(C, dk, bias=False)

# Record weights before the step
Wq_before = Wq.weight.data.clone()
Wk_before = Wk.weight.data.clone()
Wv_before = Wv.weight.data.clone()

# Forward pass: full attention
Q = Wq(X)
K = Wk(X)
V = Wv(X)

scores  = Q @ K.T / math.sqrt(dk)
weights = F.softmax(scores, dim=-1)
output  = weights @ V   # (3, 4)

# Dummy loss: push all output values toward 1.0
loss = (output - torch.ones_like(output)).pow(2).mean()
loss.backward()

# Gradient step (lr=0.1)
with torch.no_grad():
    Wq.weight -= 0.1 * Wq.weight.grad
    Wk.weight -= 0.1 * Wk.weight.grad
    Wv.weight -= 0.1 * Wv.weight.grad

# Measure how much each matrix changed
dq = (Wq.weight.data - Wq_before).abs().mean().item()
dk_ = (Wk.weight.data - Wk_before).abs().mean().item()
dv = (Wv.weight.data - Wv_before).abs().mean().item()

print(f"\nLoss: {loss.item():.4f}")
print(f"\nMean absolute weight change after one step:")
print(f"  ΔWq = {dq:.6f}")
print(f"  ΔWk = {dk_:.6f}")
print(f"  ΔWv = {dv:.6f}")

print(f"\nAre the changes equal?  ΔWq == ΔWk: {abs(dq - dk_) < 1e-6}  |  ΔWq == ΔWv: {abs(dq - dv) < 1e-6}")

print(f"\nWq after one step:\n{Wq.weight.data.round(decimals=3)}")
print(f"\nWk after one step:\n{Wk.weight.data.round(decimals=3)}")
print(f"\nWv after one step:\n{Wv.weight.data.round(decimals=3)}")

print("""
>>> WHAT TO NOTICE:
    ΔWq, ΔWk, ΔWv are all different — each matrix received a unique
    gradient signal from the same forward pass.

    WHY? Backprop flows differently through each matrix:
      - Wv directly controls the VALUES that get aggregated.
        Its gradient comes straight from the output error.
      - Wq and Wk control the ATTENTION PATTERN (the weights).
        Their gradients flow back through softmax and the scores,
        a longer and more indirect path.

    Over thousands of steps these different signals drive the
    three matrices to specialise:
      Wq  → learns what each word should search for
      Wk  → learns how each word should present itself to queries
      Wv  → learns what information to pass along when selected

    A single shared matrix (Q=K=V=X) can never develop these
    three distinct roles — it receives one blended gradient
    that averages away the specialisation.
""")
