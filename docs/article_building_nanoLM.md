# I built a small language model in NumPy, by hand. Here are the numbers.

I wanted to understand transformers without a framework hiding the work. So I wrote one in NumPy and derived every gradient myself. No PyTorch for the first six phases, no autograd. The result is nanoLM: about 4,400 lines that start as a toy bigram and end as a 2025-style transformer.

The constraint I set was simple. Each phase adds one idea and ships something I can run. Nothing goes in until I can compute its gradient by hand.

```
+-------+--------------------------------------+--------------------------------------+
| Phase | What I built                         | Idea it teaches                      |
+-------+--------------------------------------+--------------------------------------+
| 0-1   | Config, logger, tokenizer, loader    | Tracking, tokenization               |
| 2     | NumPy bigram + backprop by hand      | Embeddings, loss, gradient descent   |
| 3-4   | Self-attention, multi-head, N layers | The transformer block; depth & width |
| 5     | AdamW, LR schedule, val split        | Real training recipes                |
| 6     | Greedy, temperature, top-k, top-p    | Sampling knobs                       |
| 7     | PyTorch mirror on Apple Silicon      | Why frameworks exist                 |
| 8     | Instruction fine-tune, DPO sketch    | Base model to assistant              |
| 9     | RoPE, RMSNorm, SwiGLU, GQA, KV-cache | The 2017 to 2025 gap                 |
+-------+--------------------------------------+--------------------------------------+
```

## Phase 2: a model you can actually watch learn

A bigram over a 24-character vocab has 3,096 parameters. The trainer prints exactly where they come from:

```
  Layer     Shape       Params
  embed.E   (24 x 64)    1,536
  proj.W    (64 x 24)    1,536
  proj.b    (24,)           24
  TOTAL                  3,096
```

At init the model is maximally confused: cross-entropy 3.18, perplexity 24.0. That perplexity means "all 24 characters look equally likely." After 300 steps of plain gradient descent it drops to 2.13, perplexity 8.4. So the model went from guessing among 24 options to roughly 8.

The run takes 232 ms. That is about 1,300 gradient steps per second on a laptop CPU, no GPU. When the loop is that fast you stop waiting and start trying things, which is most of the point of building small.

Scaling stops being vague too. Add a real transformer block and you jump from 3K params to 212,352 (1 layer, D=128). Go to 4 layers and 4 heads and it is 805,632. "7B vs 70B" means something once you can print the table and point at where the numbers come from.

## Phase 9: the modern stack, one flag at a time

This was the part I was waiting for. Take the GPT-2-era model and add what models after 2023 actually use, each behind its own switch so I can see the effect in isolation.

RoPE replaces the learned position table with a rotation of the query and key vectors. The dot product then depends on the distance between two tokens, not their absolute positions. No extra parameters. RMSNorm drops the mean and the bias from LayerNorm and works just as well with less compute. SwiGLU swaps ReLU for a gated activation; I size its hidden layer to about 8D/3 rounded to 64, same as LLaMA, so the parameter count stays fair.

GQA lets several query heads share one key/value head. On my 4-layer model, going from 8 KV heads to 2 cut total parameters by 11.4% and shrinks the inference cache. KV-cache stops recomputing attention over the full history every step. The win grows with context: I measured 2.8x at a 32-token window and 4.9x at 128. int8 quantization stores weights as 8-bit integers plus a scale, which compressed the weight matrices 7.9x with an average error around 2e-4.

Stacked together, the biggest config I run is 4.2M parameters and still trains on a CPU. Tiny, but it shares every structural idea with the large ones.

## The useful part: I reviewed my own code and two features were fake

This is the thing I did not expect to learn, and it is not about transformers.

After Phase 9 I went back and reviewed the code properly. Two features I had marked done were quietly doing nothing.

The first was response masking. Instruction fine-tuning is supposed to compute loss only on the assistant's reply, not the prompt. My code built a per-token mask and then collapsed it into one number and scaled every gradient by it. That is the same as training on the full sequence at a lower learning rate. The prompt tokens were still being learned. The feature did not exist.

The second was quantization. My function rounded the weights to the int8 grid and then stored them back as float64. The error was real but the memory savings, the entire point, were zero. I checked the file size before and after. Identical.

Both passed their tests. The tests asked "does loss go down" and "is the error small," not "is this doing what it says." That is the trap with hand-written ML. A wrong sign or a shortcut does not crash. It just makes training a bit worse and you go blame the data.

The fix that changed how I work on this: a gradient-check test. For every backward pass I wrote, it compares my analytic gradient to a finite-difference of my own forward pass. They agree to about 1e-12 across attention, GQA, RoPE, SwiGLU and the norms. Masking now zeros the prompt gradient, and I assert it: perturb a masked token and the loss moves by exactly 0. Quantization now actually shrinks the model. The suite is at 130 tests.

## The numbers, in one place

```
+---------------------------------+--------------------+
| Measurement                     | Result             |
+---------------------------------+--------------------+
| bigram params                   | 3,096              |
| bigram loss (300 steps)         | 3.18 -> 2.13       |
| bigram perplexity               | 24.0 -> 8.4        |
| training speed (CPU)            | ~1,300 steps/sec   |
| params: 1L -> 4L (D=128)        | 212,352 -> 805,632 |
| GQA kv=8 -> kv=2                | -11.4% params      |
| KV-cache speedup (32 / 128 ctx) | 2.8x / 4.9x        |
| int8 weight compression         | 7.9x  (err ~2e-4)  |
| largest config                  | 4.2M params        |
| gradient-check agreement        | ~1e-12             |
| tests                           | 130                |
+---------------------------------+--------------------+
```

## If you try this

Build the smallest version that still contains the real mechanism. A 24-character vocab teaches the same cross-entropy gradient as a 100K BPE vocab and fits in your head. Print one weight and watch it move. And when something "works," write the test that would catch you fooling yourself. For ML that test is almost always a numerical gradient check, not an accuracy number.

The code is on GitHub: [github.com/Sandesh-raut/nanoLM-from-scratch](https://github.com/Sandesh-raut/nanoLM-from-scratch). Clone it, run pytest, change a number, see what happens.
