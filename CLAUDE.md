# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Code

```bash
uv run attention_exercises.py   # full walkthrough
uv run exercise1.py             # individual exercise
uv run exercise1_my_turn.py     # worked "YOUR TURN" solution
```

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. The only dependency is `torch`. There are no tests or linter configs.

## Architecture

Each exercise is a standalone, self-contained script — not a library. Every file prints its own output and can be run independently. `attention_exercises.py` is the original monolithic walkthrough containing all exercises in sequence.

**Exercise progression:**

1. `exercise1.py` — `nn.Embedding` as a lookup table; demonstrates the static word vector problem
2. `exercise2.py` — manual Q/K/V with `Q = K = V = X`; score → softmax → aggregate on "crane lifted steel"
3. `exercise3.py` — adds `/ math.sqrt(dk)` scaling and batch dimension `(B, T, C)`
4. `exercise4.py` — causal mask via `torch.tril` + `masked_fill(..., float('-inf'))`
5. `exercise5.py` — fused QKV projection `nn.Linear(C, 3*C)`, head splitting with `.view` + `.transpose`, all heads in one batched matmul
6. `exercise_bonus.py` — `CausalSelfAttention(nn.Module)` packaging exercises 1–5, matching nanoGPT's structure
7. `exercise6.py` — learned QKV projections; side-by-side comparison of `Q = K = V = X` vs. separate `Wq`, `Wk`, `Wv` matrices

**My Turn files** implement the "YOUR TURN" prompt at the end of each exercise:

- `exercise1_my_turn.py` — 50-word, 8-dim embedding; why different IDs produce different vectors
- `exercise2_my_turn.py` — bird vs. machine context comparison for "crane"
- `exercise3_my_turn.py` — scaling comparison at `dk` = 2, 64, 512
- `exercise4_my_turn.py` — causal mask for a 5-token sentence
- `exercise5_my_turn.py` — `n_head=2` vs `n_head=4` shape comparison
- `exercise_bonus_my_turn.py` — stacking 6 `CausalSelfAttention` blocks with per-layer L2 delta tracking
- `exercise6_my_turn.py` — inspects initial `Wq`/`Wk`/`Wv` weights and runs one gradient step to show they receive different gradient signals

**Result files** (`exercise*_result`) are saved stdout from each script — useful as reference for expected output.

## Planned Future Exercises

7. Positional Encoding
8. Full Transformer Block (residual + layer norm + MLP)
9. Attention Weight Visualisation
10. KV Cache
11. Minimal Trainable GPT
