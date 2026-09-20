# Tiny Shakespeare GPT — A Transformer Language Model Built From Scratch

A character/token-level GPT-style transformer implemented from first principles in PyTorch — no `transformers` library, no pretrained weights. Built as a learning and portfolio project to demonstrate a ground-up understanding of how modern LLMs actually work, from tokenization through self-attention to autoregressive generation.

## What this project implements, from scratch

- **Character-level tokenizer** — simple stoi/itos vocabulary over the training corpus
- **Byte-Pair Encoding (BPE) tokenizer + trainer** — implemented from raw bytes, including the merge-counting algorithm, no external tokenizer libraries
- **Causal self-attention** — scaled dot-product attention with query/key/value projections and a triangular causal mask, verified independently before being wrapped into the full model
- **Multi-head attention** — parallel attention heads with output projection
- **Full transformer block** — attention + feedforward MLP, each wrapped in residual connections and pre-layer-norm, stacked into a multi-layer GPT
- **Training loop** — AdamW optimizer, train/val loss tracking to monitor overfitting
- **Sampling strategies** — temperature scaling and top-k filtering for controllable generation

## Architecture

| Component | Value |
|---|---|
| Embedding dim | 64 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Context length (block size) | 32 |
| Parameters (char-level, vocab=65) | ~0.21M |
| Parameters (BPE, vocab=556) | ~0.27M |

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — ~1.1M characters of Shakespeare's plays, the standard toy dataset for from-scratch LLM projects (as used in Karpathy's nanoGPT).

## Results

Both tokenization strategies were trained for 5,000 steps with identical architecture:

| Tokenizer | Vocab size | Final train loss | Final val loss |
|---|---|---|---|
| Character-level | 65 | 1.58 | 1.77 |
| BPE (300 merges) | 556 | 2.94 | 3.37 |

*(Loss values aren't directly comparable across vocab sizes — higher vocab raises the random-guess baseline loss, from ≈4.17 for 65 tokens to ≈6.32 for 556 tokens.)*

**Finding:** At this small dataset/model scale, BPE did not outperform character-level tokenization — if anything, generated text from the char-level model showed cleaner word boundaries and formatting. This makes sense: BPE's efficiency gains come from packing more meaning per token, but that requires enough training data to properly learn embeddings for a much larger vocabulary (556 vs 65 tokens). On ~1MB of text, the char-level model's smaller vocabulary was easier to train well. BPE's advantages are expected to show up more clearly at larger data/model scale — most from-scratch tutorials assume BPE is strictly better without actually testing this tradeoff.

## Sample generation (character-level model, temperature=0.7, top-k=10)

```
Men your greate.

CLARENCE:
Them so a changed-senter.

LEONTES:
The show she hath a lieging her must have my should so,
I making he must of thou done, be silves, if her commans in mercy some hath daughter and the mystate of my hunders...
```

The model has learned the *structural form* of the source text — character names in caps followed by colons, line breaks, archaic contractions — despite having no semantic understanding of the content. This is expected for a model this size (~0.2-0.3M params vs. billions in production LLMs).

## What this doesn't do (by design)

This is a from-scratch educational implementation, not a production system. It does not include: rotary/relative positional embeddings, KV-caching for fast inference, mixed-precision training, distributed training, or any pretraining-scale data. Those would be natural next steps for scaling this up.

## Running it

The full implementation is in `tiny_shakespeare_gpt.ipynb`. Runs on a free Google Colab GPU (T4) in a few minutes end-to-end, including both tokenizer variants.
