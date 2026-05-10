# Attention Is All You Need — PyTorch Exercises

A hands-on guided walkthrough of the Transformer attention mechanism, built from absolute scratch in PyTorch. Every exercise is tied to the intuition from [Zachary Huang's "Attention Click Forever" video](https://www.youtube.com/watch?v=iDulhoQ2pro).

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

**3. Run the exercises**
```bash
uv run attention_exercises.py
```

> Tested on macOS (Apple Silicon M2). PyTorch's MPS backend is supported automatically — no GPU required.

## How to Use

The file is structured as a linear walkthrough. Each exercise:
- Explains the **why** before showing any code
- Uses **tiny, readable tensors** (3 tokens, 8 dimensions) so every number is traceable
- Ends with a **"YOUR TURN"** prompt to experiment with

You can run the whole file at once, or paste sections into a Python REPL or Jupyter notebook to step through interactively.

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

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Zachary Huang — "Attention Click Forever"](https://www.youtube.com/@zach_explainss) — the video that inspired these exercises
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT) — the clean GPT implementation these exercises are modeled after
