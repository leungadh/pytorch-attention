"""
=============================================================
  ATTENTION IS ALL YOU NEED — PyTorch Guided Exercises
  Based on Zachary Huang's "Attention Click Forever" video
=============================================================

How to use this file:
  - Run the whole file at once:  python attention_exercises.py
  - Or paste section by section into a Python REPL / Jupyter notebook
  - Read every comment — they're the lesson!

Prerequisites:
  pip install torch

=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────
# EXERCISE 1 — Word Embeddings & The Static Problem
# ─────────────────────────────────────────────────────────────
#
# Zachary's key insight: a neural network only understands
# lists of numbers (vectors). An embedding layer is a giant
# lookup table — every word gets its own row.
#
# But here's the villain: the lookup is STATIC.
# "bank" in "river bank" and "bank" in "withdrew money from the bank"
# return the EXACT SAME vector. The model is deaf to context.
#
# Let's see this for ourselves.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("EXERCISE 1 — Word Embeddings & The Static Problem")
print("=" * 60)

# Tiny vocabulary: 10 words, each represented by a 4-number vector
vocab_size = 10
n_embed = 4

torch.manual_seed(42)  # so results are reproducible
token_embedding_table = nn.Embedding(vocab_size, n_embed)

# Imagine our vocabulary:
#   ID 3 = "cat"
#   ID 7 = "bank"

# Look up "bank" in sentence 1: "I sat on the river bank"
bank_id = torch.tensor([7])
vector_in_river_context = token_embedding_table(bank_id)

# Look up "bank" in sentence 2: "I withdrew money from the bank"
# Same word, same ID → same lookup
vector_in_money_context = token_embedding_table(bank_id)

print("\nVector for 'bank' in 'river bank' context:")
print(vector_in_river_context)

print("\nVector for 'bank' in 'money bank' context:")
print(vector_in_money_context)

print("\nAre they identical?", torch.equal(vector_in_river_context, vector_in_money_context))

print("""
>>> WHAT TO NOTICE:
    Both vectors are exactly the same — even though the meaning is completely
    different. This is the static problem Zachary described.
    The attention mechanism is the solution: it will MODIFY these vectors
    using context from neighboring words.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Try it: create an embedding for a vocab of 50 words with 8 dimensions.
# Look up word ID 0, 1, and 2. Are they different from each other? Why?
#
# token_embedding_table_2 = nn.Embedding(???, ???)
# print(token_embedding_table_2(torch.tensor([0, 1, 2])))
# ───────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# EXERCISE 2 — The QKV Conversation (crane lifted steel)
# ─────────────────────────────────────────────────────────────
#
# Zachary's mental model: every word plays THREE roles:
#
#   Q (Query)  = "What am I looking for?" — the word's question
#   K (Key)    = "What do I have?"        — the word's label
#   V (Value)  = "What will I give you?"  — the word's actual content
#
# The three-step process:
#   1. SCORE     — Q from one word dot-products with every K
#                  (high score = strong match)
#   2. NORMALIZE — softmax turns raw scores into percentages
#                  (who do I listen to?)
#   3. AGGREGATE — weighted average of all V vectors
#                  (absorb the information)
#
# Example: "crane lifted steel"
#   Is "crane" a bird or a machine?
#   It looks at its neighbors and updates itself to be more machine-like.
# ─────────────────────────────────────────────────────────────

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
# ... (run the same steps above)
# ───────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# EXERCISE 3 — Scaled Dot-Product Attention (the full formula)
# ─────────────────────────────────────────────────────────────
#
# The full formula Zachary showed:
#
#   Attention(Q, K, V) = softmax( Q @ Kᵀ / √dk ) @ V
#
# The only new piece vs. Exercise 2 is dividing by √dk.
# WHY? As vectors get longer (larger dk), dot products get bigger,
# which makes softmax outputs very extreme (close to 0 or 1).
# Dividing by √dk keeps the scores in a stable range.
#
# Zachary's setup: T=3 tokens, C=2 dimensions (same as above)
# but now we add the scaling step explicitly.
# ─────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────
# EXERCISE 4 — The Causal Mask (no cheating!)
# ─────────────────────────────────────────────────────────────
#
# Zachary called this the "don't look ahead" problem.
# GPT generates one word at a time. When predicting word 3,
# it must not be allowed to SEE word 3 — that's cheating.
#
# The fix: set future scores to -infinity BEFORE softmax.
# e^(-inf) = 0, so those positions get zero attention weight.
#
# We use torch.tril() (lower triangular matrix) to build the mask:
#   1 = "allowed to look"
#   0 = "blocked (future)"
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("EXERCISE 4 — The Causal Mask")
print("=" * 60)

T = 3  # sentence length

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


# ─────────────────────────────────────────────────────────────
# EXERCISE 5 — Multi-Head Attention
# ─────────────────────────────────────────────────────────────
#
# Zachary's insight: one "conversation" (one attention head) isn't
# enough to capture all the richness of language.
#
# Multi-head attention runs MANY conversations at once:
#   Head 1 → might track verb-object relationships
#   Head 2 → might resolve pronoun references
#   Head 3 → might handle long-distance dependencies
#
# The clever trick: we don't run separate Q, K, V for each head.
# Instead, we RESHAPE the embedding dimension (C) into (n_heads, head_dim)
# using .view() and .transpose(), then compute ALL heads in parallel.
#
# Example from the video (GPT-2 scale):
#   C = 768 embedding dimensions
#   n_heads = 12
#   head_dim = 768 / 12 = 64  (each head gets 64 dimensions)
#
# We'll use a tiny version so the shapes are easy to follow.
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("EXERCISE 5 — Multi-Head Attention (the parallel conversations)")
print("=" * 60)

torch.manual_seed(42)

B      = 1   # batch size
T      = 3   # sequence length (tokens)
C      = 8   # embedding dimension (we use 8 instead of 768 to keep it readable)
n_head = 2   # number of attention heads
head_dim = C // n_head  # = 4  (each head gets 4 dimensions)

print(f"\nSetup: B={B}, T={T}, C={C}, n_head={n_head}, head_dim={head_dim}")

# ── Step 1a: Compute Q, K, V with learned linear layers ─────
# In a real model Q, K, V each have their OWN weight matrix.
# Here we combine all three into one big projection (3*C output)
# then split — this is how GPT2 / nanoGPT does it.
c_attn = nn.Linear(C, 3 * C, bias=False)

X_in = torch.randn(B, T, C)
qkv  = c_attn(X_in)          # shape: (1, 3, 24)
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

# Add causal mask
mask = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(mask == 0, float('-inf'))

weights = F.softmax(scores, dim=-1)
print(f"\nAttention weights shape: {list(weights.shape)}")
print("  (1 batch × 2 heads × 3×3 attention matrix)")

# ── Step 3: Aggregate values ─────────────────────────────────
out = weights @ V       # (1, 2, 3, 4)
print(f"Output per head shape: {list(out.shape)}")

# ── Step 4: Merge heads back ─────────────────────────────────
# Reverse the transpose, then merge (n_head, head_dim) → C
out = out.transpose(1, 2).contiguous().view(B, T, C)  # (1, 3, 8)
print(f"After merging heads: {list(out.shape)}")

# ── Step 5: Final linear projection ──────────────────────────
c_proj = nn.Linear(C, C, bias=False)
final_output = c_proj(out)   # (1, 3, 8)
print(f"Final output shape: {list(final_output.shape)}")

print(f"""
>>> WHAT TO NOTICE:
    The shape journey for multi-head:
      Input X:               {list(X_in.shape)}
      After Q/K/V projection: {list(Q.shape)}  (B, n_head, T, head_dim)
      Attention weights:     {list(weights.shape)}
      After merging heads:   {list(out.shape)}     ← back to (B, T, C)
      Final output:          {list(final_output.shape)}     ← same as input!

    The input and output shapes match. This is what allows transformers
    to be stacked — the output of one block feeds cleanly into the next.
    GPT-3 does this 96 times. Same block, stacked 96 layers deep.
""")

# ── YOUR TURN ──────────────────────────────────────────────
# Try changing n_head to 4 (and keep C=8, so head_dim=2).
# Does the output shape change? What about the weights shape?
# This mirrors going from 4-head to 8-head attention in real models.
# ───────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────
# BONUS — The Full CausalSelfAttention Module (nn.Module)
# ─────────────────────────────────────────────────────────────
#
# This is the "production-ready code" Zachary showed at the end —
# the code that looked like a nightmare before the video, but
# now should make complete sense.
#
# Everything you built above is in here:
#   - c_attn  → the QKV projection
#   - view + transpose → split into heads
#   - scaled dot-product + causal mask → the core formula
#   - transpose + view + c_proj → merge and project
# ─────────────────────────────────────────────────────────────

print("=" * 60)
print("BONUS — Full CausalSelfAttention Module")
print("=" * 60)

class CausalSelfAttention(nn.Module):
    """
    A clean, minimal implementation of causal multi-head self-attention.
    Every line corresponds to something you built in exercises 1–5.
    """
    def __init__(self, n_embed, n_head):
        super().__init__()
        assert n_embed % n_head == 0, "n_embed must be divisible by n_head"

        self.n_head  = n_head
        self.n_embed = n_embed
        self.head_dim = n_embed // n_head

        # Exercise 5 Step 1: single projection for Q, K, V together
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)

        # Exercise 5 Step 5: final output projection
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


# Test it!
torch.manual_seed(0)
model = CausalSelfAttention(n_embed=8, n_head=2)
x_test = torch.randn(1, 3, 8)   # (batch=1, tokens=3, embed=8)
y_test = model(x_test)

print(f"\nInput shape:  {list(x_test.shape)}")
print(f"Output shape: {list(y_test.shape)}")
print("\nInput (first token):", x_test[0, 0].detach().round(decimals=3).tolist())
print("Output(first token):", y_test[0, 0].detach().round(decimals=3).tolist())

print("""
>>> WHAT TO NOTICE:
    Input and output shapes are identical.
    But the output values are different — each token has absorbed
    information from its causal context (past tokens only).

    This is exactly the module Zachary showed at the very end.
    The "nightmare" code from the beginning of the video.
    Now it's just a map of things you built yourself, step by step.

Congrats — attention has clicked. 🎉
""")
