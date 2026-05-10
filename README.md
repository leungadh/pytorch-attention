# Attention Is All You Need — PyTorch Exercises

A hands-on guided walkthrough of the Transformer attention mechanism, built from absolute scratch in PyTorch. Every exercise is tied to the intuition from [Zachary Huang's "Give me 20 min, I will make Attention click forever"](https://youtu.be/4vye3iUrR-g?si=ZJgClyP517GURfvv).

## What You'll Build

By the end of these exercises, you will have implemented every piece of the attention mechanism yourself — starting from a simple word lookup table and ending with a full `CausalSelfAttention` module, the core building block of GPT.

| Exercise | Topic | Key Concept |
|----------|-------|-------------|
| 1 | Word Embeddings | Why static vectors are the villain |
| 2 | QKV Conversation | Score → Normalize → Aggregate |
| 3 | Scaled Dot-Product Attention | The full formula: `softmax(QKᵀ / √dk) · V` |
| 4 | Causal Mask | Preventing the model from "cheating" |
| 5 | Multi-Head Attention | Parallel conversations via `view` + `transpose` |
| Bonus | `CausalSelfAttention` module | The complete `nn.Module`, demystified |
| 6 | Learned QKV Projections | Why `Q = K = V = X` is a shortcut, and what separate `Wq`, `Wk`, `Wv` matrices unlock |

## Files

Each exercise is available as a standalone file. Every exercise file ends with a **"YOUR TURN"** prompt; the corresponding `_my_turn` file implements and explains it.

### Exercises

| File | Description |
|------|-------------|
| `attention_exercises.py` | All exercises in one linear walkthrough |
| `exercise1.py` | `nn.Embedding` as a lookup table; demonstrates the static problem — "bank" returns the same vector regardless of context |
| `exercise2.py` | Manual Q, K, V attention (`Q = K = V = X`) on "crane lifted steel"; score → softmax → aggregate, showing how crane absorbs its machine-like context |
| `exercise3.py` | Adds the `/ √dk` scaling factor and the batch dimension `(B, T, C)`; shows why large `dk` makes softmax collapse without scaling |
| `exercise4.py` | Builds a `torch.tril` causal mask and applies `masked_fill(..., -inf)` so future tokens contribute zero attention weight |
| `exercise5.py` | Fused QKV projection via `nn.Linear(C, 3*C)`, head splitting with `.view` + `.transpose`, all heads computed in one batched matmul, then merged back |
| `exercise_bonus.py` | Packages exercises 1–5 into a clean `CausalSelfAttention(nn.Module)` matching nanoGPT's structure |
| `exercise6.py` | Side-by-side comparison of `Q = K = V = X` vs. separate `Wq`, `Wk`, `Wv` projections on "crane lifted steel"; shows Q, K, V diverge into distinct vectors and why that separation is essential for learning |

### My Turn

| File | Description |
|------|-------------|
| `exercise1_my_turn.py` | Creates a 50-word, 8-dim embedding; looks up IDs 0, 1, 2 and explains why different IDs produce different vectors while the same ID always returns the same one |
| `exercise2_my_turn.py` | Runs the QKV conversation in both machine context (lifted, steel) and bird context (wings, feathers), comparing crane's output vector side-by-side to show context determines meaning |
| `exercise3_my_turn.py` | Runs scaled vs. unscaled attention at `dk` = 2, 64, and 512; shows raw scores growing linearly with dimension while scaled scores stay bounded |
| `exercise4_my_turn.py` | Builds the causal mask for a 5-token sentence, annotates exactly which tokens each position can attend to, and verifies the upper triangle is provably zero after softmax |
| `exercise5_my_turn.py` | Compares `n_head=2` vs `n_head=4` (with `C=8` fixed) side-by-side; shows that output shape is invariant to head count while the attention weights shape and per-head `head_dim` both change |
| `exercise_bonus_my_turn.py` | Stacks 6 `CausalSelfAttention` blocks and tracks the L2 change in token representations at each layer, confirming shape invariance and grounding the design in GPT-2/GPT-3 depths |

## Prerequisites

- Basic Python knowledge
- Familiarity with what Q, K, V mean conceptually (watch the video above first!)
- No prior PyTorch experience needed

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

**1. Install uv**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. Clone the repo and set up the environment**
```bash
git clone <your-repo-url>
cd <your-repo-name>
uv init
uv add torch
```

**3. Run an exercise**
```bash
uv run attention_exercises.py   # full walkthrough
uv run exercise1.py             # individual exercise
uv run exercise1_my_turn.py     # worked "YOUR TURN" solution
```

> Tested on macOS (Apple Silicon M2). PyTorch's MPS backend is supported automatically — no GPU required.

## How to Use

Each exercise file:
- Explains the **why** before showing any code
- Uses **tiny, readable tensors** (3 tokens, 8 dimensions) so every number is traceable
- Ends with a **"YOUR TURN"** prompt — the `_my_turn` file implements and explains it

Run `attention_exercises.py` for the full linear walkthrough, or run individual files to focus on one concept. You can also paste any section into a Python REPL or Jupyter notebook to step through interactively.

## The Core Formula

Everything in this repo is building toward one equation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

By Exercise 3, every symbol in this formula will be something you've written yourself.

## Concepts Covered

**Word Embeddings** — A neural network can't understand the word "bank". It only understands lists of numbers. An embedding layer is a learned lookup table that maps each word to a vector. The problem: it's static — "bank" (river) and "bank" (money) get the exact same vector.

**Query, Key, Value** — Each word plays three roles simultaneously. The Query is the word's question ("what am I looking for?"). The Key is its label ("what do I have?"). The Value is its content ("what will I give you?"). A query finds the best-matching key, then pulls in that word's value.

**Scaled Dot-Product Attention** — The three-step process encoded as efficient matrix math: score every word against every other word, normalize with softmax, take a weighted average of the values.

**Causal Mask** — GPT generates one token at a time. It must not be allowed to see future tokens. Setting future scores to `-inf` before softmax makes those positions contribute zero attention weight.

**Multi-Head Attention** — One attention calculation isn't rich enough to capture all the structure in language. Running multiple "heads" in parallel — each potentially specializing in different relationships — dramatically increases the model's expressive power.

## Future Exercises

Ideas for extending the series, roughly in order of progression:

| # | Topic | What it adds |
|---|-------|-------------|
| 7 | Positional Encoding | Attention is permutation-invariant — "crane lifted steel" and "steel lifted crane" currently produce the same output. Add sinusoidal or learned positional embeddings to inject token order |
| 8 | Full Transformer Block | Wrap `CausalSelfAttention` with a residual connection, layer norm, and a two-layer MLP (feed-forward) sublayer — the complete GPT building block |
| 9 | Attention Weight Visualisation | Plot the `(T × T)` attention weight matrix as a heatmap with `matplotlib`; seeing which tokens attend to which is the most intuitive way to verify the model's behaviour |
| 10 | KV Cache | Show how inference speeds up by caching past `K` and `V` tensors — each new token then only computes one row of the attention matrix instead of recomputing the full `T × T` |
| 11 | Minimal Trainable GPT | Combine embedding + positional encoding + N transformer blocks + a language model head into a tiny GPT that can be trained on a short text (e.g. Shakespeare) |

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Zachary Huang — "Give me 20 min, I will make Attention click forever"](https://youtu.be/4vye3iUrR-g?si=ZJgClyP517GURfvv) — the video that inspired these exercises
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — the clean GPT implementation these exercises are modeled after
