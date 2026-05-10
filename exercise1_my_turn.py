"""
EXERCISE 1 — MY TURN

Task: create an embedding for a vocab of 50 words with 8 dimensions.
Look up word ID 0, 1, and 2. Are they different from each other? Why?
"""

import torch
import torch.nn as nn

print("=" * 60)
print("EXERCISE 1 — MY TURN: Exploring Embedding Lookup")
print("=" * 60)

token_embedding_table = nn.Embedding(50, 8)

vectors = token_embedding_table(torch.tensor([0, 1, 2]))

print("\nVectors for word IDs 0, 1, and 2:")
for word_id, vec in enumerate(vectors):
    print(f"  ID {word_id}: {vec.tolist()}")

print("\nAre ID 0 and ID 1 identical?", torch.equal(vectors[0], vectors[1]))
print("Are ID 0 and ID 2 identical?", torch.equal(vectors[0], vectors[2]))
print("Are ID 1 and ID 2 identical?", torch.equal(vectors[1], vectors[2]))

print("""
>>> WHY ARE THEY DIFFERENT?
    nn.Embedding is just a matrix of shape (vocab_size, n_embed) —
    here (50, 8). Each row is one word's vector, initialised independently
    from a random normal distribution (mean=0, std=1) by default.

    Looking up ID 0, 1, and 2 simply returns rows 0, 1, and 2 of that
    matrix. Since every row is drawn independently at random, they will
    almost certainly be different — just like three random 8-dimensional
    points in space are almost certainly not the same point.

    CONTRAST with Exercise 1's "bank" example:
    Looking up the SAME ID twice always returns the SAME row — that's
    the static problem. Different IDs → different rows → different vectors.
""")

# ── Verify the "same ID = same vector" property still holds ──
v_first  = token_embedding_table(torch.tensor([0]))
v_second = token_embedding_table(torch.tensor([0]))
print("Same ID looked up twice — still identical?", torch.equal(v_first, v_second))
