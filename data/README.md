# Corpora

Two training corpora ship with nanoLM. They serve different purposes.

## `corpus.txt` — 599 chars, 24-char vocab

Nursery rhymes and repeated simple sentences. Small enough that a full training
run finishes in well under a second, which makes it the right default for
Phases 2–9: you change a hyperparameter and see the result immediately.

Small vocab is also what makes the Phase 2 numbers legible — a 24×64 embedding
plus a 64×24 projection plus 24 biases is 3,096 parameters you can add up by
hand.

## `corpus_large.txt` — 600K chars, 95-char vocab

Two deliberately different kinds of text, interleaved:

| Source | Share | License |
|--------|-------|---------|
| Shakespeare (the "tiny Shakespeare" text used across char-level LM work) | ~67% | Public domain |
| nanoLM's own Python source | ~33% | MIT (this repo) |

Everything is normalized to printable ASCII plus newline, so the vocab has no
long tail of near-unseen characters.

### Why two modes

Mixture-of-Experts (Phase 10) only becomes interesting when the data contains
distinguishable kinds of text. Verse and Python have very different character
statistics — indentation, brackets and underscores against spaces and letters —
so a router has something real to separate. On `corpus.txt` every window looks
alike and the experts have nothing to divide up.

### Why interleaved rather than concatenated

The two sources alternate in 4 KB chunks. Two reasons:

1. `BatchLoader` samples random windows of `block_size` characters. At 4 KB per
   chunk almost every window lands well inside one mode, so a routing decision
   is well defined.
2. `BatchLoader.split()` divides the data contiguously — the first 90% trains
   and the last 10% validates. Concatenating prose then code would put pure
   Python in the validation set and make the validation loss meaningless.

### Regenerating

`corpus_large.txt` is a snapshot: the code half was taken from this repo at the
time it was written, so it drifts as the source changes. That is harmless — it
is training data, not a build artifact.
