"""
EXERCISE 1 — Word Embeddings & The Static Problem

Zachary's key insight: a neural network only understands
lists of numbers (vectors). An embedding layer is a giant
lookup table — every word gets its own row.

But here's the villain: the lookup is STATIC.
"bank" in "river bank" and "bank" in "withdrew money from the bank"
return the EXACT SAME vector. The model is deaf to context.
"""

import torch
import torch.nn as nn

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
# token_embedding_table_2 = nn.Embedding(50, 8)
# print(token_embedding_table_2(torch.tensor([0, 1, 2])))
# ───────────────────────────────────────────────────────────
