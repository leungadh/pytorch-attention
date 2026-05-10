# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Code

```bash
uv run attention_exercises.py
```

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. The only dependency is `torch`. There are no tests or linter configs.

## Architecture

The entire project is a single script: `attention_exercises.py`. It is a linear, top-to-bottom walkthrough — not a library. Each exercise prints its own output and flows directly into the next.

**Exercise progression:**

1. `nn.Embedding` — demonstrates the static word vector problem
2. Manual Q/K/V — score → softmax → aggregate, with `Q = K = V = X` (no learned projections)
3. Scaled dot-product — adds `/ math.sqrt(dk)` scaling; introduces batch dimension `(B, T, C)`
4. Causal mask — applies `torch.tril` + `masked_fill(..., float('-inf'))` on top of Exercise 3's scores
5. Multi-head attention — uses `nn.Linear(C, 3*C)` for a single fused QKV projection, then `.view(B, T, n_head, head_dim).transpose(1, 2)` to split heads; all heads computed in one batched matmul
6. `CausalSelfAttention(nn.Module)` — packages exercises 1–5 into a reusable module matching nanoGPT's structure

The `Result` file is a saved copy of a previous run's stdout — useful as a reference for expected output.
