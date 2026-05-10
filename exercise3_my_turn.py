"""
EXERCISE 3 — MY TURN

Task: what happens if dk is very large (e.g., 512)?
Compare raw vs. scaled scores and see whether scaling matters more
at high dimensions.

We run the experiment at three sizes — small (2), medium (64),
large (512) — so the trend is clear, not just a single data point.
"""

import torch
import torch.nn.functional as F
import math

print("=" * 60)
print("EXERCISE 3 — MY TURN: Why Scaling Matters at Large dk")
print("=" * 60)

torch.manual_seed(0)

dims = [2, 64, 512]

for dk in dims:
    X = torch.randn(1, 3, dk)
    Q, K, V = X, X, X

    raw    = Q @ K.transpose(-2, -1)           # (1, 3, 3)
    scaled = raw / math.sqrt(dk)               # (1, 3, 3)

    raw_weights    = F.softmax(raw,    dim=-1)
    scaled_weights = F.softmax(scaled, dim=-1)

    # How "peaky" is the distribution? Max weight close to 1.0 = bad.
    raw_max    = raw_weights.max().item()
    scaled_max = scaled_weights.max().item()

    print(f"\n{'─' * 40}")
    print(f"dk = {dk}  (√dk = {math.sqrt(dk):.1f})")
    print(f"{'─' * 40}")
    print(f"  Max raw score:    {raw.max().item():8.2f}")
    print(f"  Max scaled score: {scaled.max().item():8.2f}")
    print(f"  Softmax on raw    — max weight: {raw_max:.4f}  {'← near 1, collapsed!' if raw_max > 0.9 else ''}")
    print(f"  Softmax on scaled — max weight: {scaled_max:.4f}  {'← near 1, collapsed!' if scaled_max > 0.9 else ''}")

print("""
>>> WHAT TO NOTICE:
    At dk=2 (Exercise 3's setup), raw and scaled scores are close —
    scaling makes little practical difference.

    At dk=64 (GPT-2's head_dim), raw scores are already large enough
    that softmax starts concentrating weight on one token. Scaling
    brings it back to a healthy spread.

    At dk=512, raw dot products can reach hundreds. Softmax treats
    this like a near-infinite temperature gap — the highest score
    gets almost all the weight (→ 1.0) and every other token gets
    almost zero. The model effectively stops listening to context.

    Dividing by √dk neutralises this: it keeps scores in roughly the
    same range regardless of dimension, so softmax always produces a
    meaningful distribution rather than a collapsed spike.

    This is why the scaling factor is non-negotiable in real models.
""")
