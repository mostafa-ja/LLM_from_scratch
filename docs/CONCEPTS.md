# Concepts behind the code

This document is a map, not a replacement for the notebooks. For each idea it answers three questions:

1. **What is it?**
2. **Why does GPT need it?**
3. **Where can I see it in this repository?**

If you can explain every section below without looking at the code, you understand the core of a GPT-style language model.

---

## 0. The whole pipeline

```text
text
  -> tokenizer -> token ids                         (B, T)
  -> token embedding + position embedding           (B, T, C)
  -> N Transformer blocks                           (B, T, C)
  -> final LayerNorm                                (B, T, C)
  -> LM head                                        (B, T, vocab_size)
  -> cross-entropy loss (training) / sampling (inference)
```

| Symbol | Meaning |
|--------|---------|
| `B` | batch size |
| `T` | sequence length (at most `block_size`) |
| `C` | embedding width (`n_embd`) |
| `vocab_size` | number of possible tokens |

Training and inference use the same forward pass, but differently:

- **Training** feeds a whole window at once and compares every position with the next token. The causal mask prevents position `t` from seeing position `t+1`.
- **Inference** starts with a prompt and generates one token at a time. The newly generated token is appended and the model runs again.

The small model in this repository is in [`src/llm_from_scratch/model.py`](../src/llm_from_scratch/model.py). The training loop is in [`src/llm_from_scratch/train.py`](../src/llm_from_scratch/train.py).

---

## 1. Tokenization

### What it is

A tokenizer turns a string into a sequence of integer ids. GPT-2 uses **byte-level Byte Pair Encoding (BPE)**.

BPE starts with 256 possible byte tokens. It repeatedly finds the most frequent adjacent pair of tokens and replaces that pair with a new token. Each merge is recorded. Encoding replays the merges in the same order.

### Why GPT needs it

Neural networks operate on vectors, not characters. Tokenization defines:

- the size of the embedding matrix,
- how long a sentence is in tokens,
- how much text fits in the context window,
- how well the model handles rare words, code, numbers, and non-English text.

A larger vocabulary shortens sequences but makes the embedding and output matrices larger. GPT-2 chose 50,257 tokens. This repository pads the output vocabulary to 50,304 for GPU alignment (see section 5).

### The split pattern matters

Text is split into chunks before BPE is applied, so merges never cross word or punctuation boundaries. GPT-2 uses:

```text
's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

The leading space is part of the token. This is why `"hello"` and `" hello"` can tokenize differently.

### Where to see it

- Notebook: [`01_tokenizer_from_scratch.ipynb`](../notebooks/01_tokenizer_from_scratch.ipynb)
- Code: [`BPETokenizer`](../src/llm_from_scratch/tokenizer.py)
- Test: [`tests/test_tokenizer.py`](../tests/test_tokenizer.py)

### Exercise

Train the tokenizer with `vocab_size=300` and `vocab_size=600`. Encode the same sentence and compare the number of tokens. Which vocabulary is better? The answer depends on the corpus and the compute budget.

---

## 2. Embeddings and positions

### What it is

Each token id indexes a row of the token embedding matrix:

```text
wte: (vocab_size, C)
```

Each position indexes a row of the position embedding matrix:

```text
wpe: (block_size, C)
```

The model adds them:

```python
x = wte(idx) + wpe(positions)
```

### Why GPT needs it

The token embedding says **what** the token is. The position embedding says **where** it is. Without position information, self-attention would treat a bag of tokens as unordered.

GPT-2 uses learned position embeddings. Some modern models use rotary position embeddings (RoPE), but the idea is the same: inject order into the representation.

### Where to see it

- Notebook: [`03_gpt2_architecture_and_optimizations.ipynb`](../notebooks/03_gpt2_architecture_and_optimizations.ipynb)
- Code: [`GPT.forward`](../src/llm_from_scratch/model.py)

---

## 3. Self-attention

### What it is

Every token produces three vectors:

- a **query** `q`: what am I looking for?
- a **key** `k`: what information do I contain?
- a **value** `v`: what do I pass on if I am attended to?

For one attention head:

$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

- `QK^T` scores every query against every key.
- Dividing by `sqrt(d_k)` keeps the scores from growing with the head dimension.
- `M` is a causal mask: `M[i,j] = 0` for `j <= i`, and `-inf` for `j > i`. After softmax, future positions get zero weight.
- Multiplying by `V` produces a weighted average of previous values.

### Multi-head attention

GPT-2 splits the embedding dimension into `n_head` heads. Each head has its own `Q`, `K`, `V` projection and can learn a different relationship. The heads are concatenated and projected back to `C` dimensions.

```text
(B, T, C) -> (B, T, 3C) -> three tensors of (B, T, C)
          -> (B, n_head, T, head_dim)
          -> attention per head
          -> (B, T, C)
```

### Why GPT needs it

Attention is the only operation that moves information **between positions**. The MLP operates on each position independently. Without attention, the model would be a fancy n-gram model.

### FlashAttention and `scaled_dot_product_attention`

Older implementations materialise the full `T x T` attention matrix. FlashAttention computes attention in chunks and avoids writing that matrix to memory, which is faster and uses less memory for long sequences.

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The single call handles scaling, masking, softmax, and the value aggregation. PyTorch dispatches to a fused kernel when the hardware supports it, and falls back otherwise.

### Where to see it

- Notebook: [`02_bigram_to_self_attention.ipynb`](../notebooks/02_bigram_to_self_attention.ipynb)
- Code: [`CausalSelfAttention`](../src/llm_from_scratch/model.py)
- Test: `tests/test_model.py::test_causal_masking`

### Exercise

Replace the `F.scaled_dot_product_attention` call with the manual formula. Run the model in eval mode and compare outputs. This is the single best way to demystify attention.

---

## 4. The Transformer block

### What it is

A GPT block has two sub-layers:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

This is the **pre-norm** layout: normalize the input, run the sub-layer, add the residual.

The MLP is a position-wise feed-forward network:

```text
Linear(C, 4C) -> GELU -> Linear(4C, C)
```

### Why GPT needs it

- **LayerNorm** stabilizes activation scale and makes deep networks trainable.
- **Residual connections** give gradients a direct path to earlier layers. They also let each block learn a small update instead of a full replacement.
- **Pre-norm** is more stable than post-norm for deep Transformers, which is why GPT-2 uses it.
- **GELU** is a smooth activation that works well in Transformers.

### Where to see it

- Code: [`Block`](../src/llm_from_scratch/model.py) and [`MLP`](../src/llm_from_scratch/model.py)
- Notebook: [`03_gpt2_architecture_and_optimizations.ipynb`](../notebooks/03_gpt2_architecture_and_optimizations.ipynb)

---

## 5. Output head, weight tying, and vocabulary padding

### Output head

The final hidden state is projected to the vocabulary size:

```text
lm_head: Linear(C, vocab_size, bias=False)
logits: (B, T, vocab_size)
```

### Weight tying

The token embedding matrix has shape `(vocab_size, C)`. The LM head weight has shape `(vocab_size, C)` as well. GPT-2 uses the **same tensor** for both:

```python
self.transformer.wte.weight = self.lm_head.weight
```

This is called weight tying. It saves `vocab_size * C` parameters and encourages the input and output spaces to share structure.

### Why 50,304 instead of 50,257?

50,257 is a prime-ish number that is not aligned to GPU-friendly sizes. 50,304 is a multiple of 64 and close to 50,257. The extra rows are unused padding; the tokenizer still produces ids up to 50,256. Aligning the output dimension helps GPU kernels and Tensor Cores.

### Where to see it

- Code: [`GPT.__init__`](../src/llm_from_scratch/model.py)
- Test: `tests/test_model.py::test_weight_tying`

---

## 6. Loss and perplexity

### Next-token cross-entropy

For a sequence of length `T`, the model makes `T` predictions. The loss is the average negative log-probability of the correct next token:

$$
L = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_{t+1}\mid x_{\le t})
$$

In code, the targets are the inputs shifted by one position. That is why `DataLoaderLite` returns `x = tokens[start:start+T]` and `y = tokens[start+1:start+T+1]`.

### Perplexity

Perplexity is `exp(loss)`. It can be read as "the model is as uncertain as if it were choosing uniformly among this many tokens." Lower is better.

A randomly initialized model has a loss near `ln(vocab_size)`. For GPT-2's 50,257-token vocabulary, that is about `10.8`.

### Where to see it

- Code: [`GPT.forward`](../src/llm_from_scratch/model.py) and [`evaluate`](../src/llm_from_scratch/train.py)
- Notebook: [`07_guided_tour.ipynb`](../notebooks/07_guided_tour.ipynb)

---

## 7. Initialization

### What it is

GPT-2 initializes linear and embedding weights from a normal distribution with standard deviation `0.02`, and sets biases to zero.

Residual output projections (`c_proj` in attention and MLP) use a smaller standard deviation:

$$
\sigma = \frac{0.02}{\sqrt{2 \cdot n_{layer}}}
$$

### Why GPT needs it

Every block adds two residual branches. If each branch has the same variance as the input, the variance grows with depth. Scaling the residual output keeps the signal stable. This is a small detail with a large effect on deep model training.

### Where to see it

- Code: [`GPT.__init__`](../src/llm_from_scratch/model.py)
- Notebook: section "Weight Initialization in GPT" in [`03_gpt2_architecture_and_optimizations.ipynb`](../notebooks/03_gpt2_architecture_and_optimizations.ipynb)

---

## 8. Optimization

### AdamW

GPT-3 and GPT-2-style training use AdamW with:

```python
AdamW(lr=6e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
```

`betas=(0.9, 0.95)` differs from the usual `(0.9, 0.999)`. A shorter second-moment window helps large models adapt to changing gradient statistics.

### Weight decay groups

Weight decay is applied to matrices (dimension >= 2) but not to biases or LayerNorm parameters:

```text
decay:    linear weights, embedding matrices
no decay: biases, LayerNorm scales
```

This is a standard best practice. Decaying biases and normalization parameters usually hurts.

### Gradient clipping

The global gradient norm is clipped at `1.0`. If the norm is larger, all gradients are scaled down proportionally. This prevents rare gradient spikes from destabilizing training.

### Learning-rate schedule

Training uses:

1. **linear warmup** from a small learning rate to `max_lr`,
2. **cosine decay** to `10%` of `max_lr`.

Warmup avoids large early updates when the model is still random. Cosine decay gives finer updates near the end of training.

### Batch size and gradient accumulation

A large effective batch is more stable but may not fit in GPU memory. The solution is gradient accumulation:

```text
effective batch = micro_batch_size * gradient_accumulation_steps
```

Each micro-batch computes a loss, divides it by the number of accumulation steps, and calls `backward()`. Gradients add up. One optimizer step is taken per effective batch. Dividing the loss keeps the accumulated gradient equal to the gradient of the mean loss.

GPT-2 small was trained with an effective batch of about 524,288 tokens. On a small GPU that means many accumulation steps.

### Mixed precision

`torch.autocast` runs matrix multiplications in `bfloat16` (or `float16`) while keeping sensitive operations in `float32`. This reduces memory use and speeds up training. `bfloat16` has the same exponent range as `float32`, so it is usually more stable than `float16` for training.

### `torch.compile`, fused AdamW, and TF32

- `torch.compile` traces the model and generates faster kernels.
- Fused AdamW combines the optimizer update into fewer GPU kernels.
- TF32 is a faster float32 mode on Ampere and newer NVIDIA GPUs. `torch.set_float32_matmul_precision("high")` enables it.

These are performance optimizations. They do not change the math of the model.

### Where to see it

- Code: [`GPT.configure_optimizers`](../src/llm_from_scratch/model.py), [`get_lr`](../src/llm_from_scratch/train.py)
- Notebook: [`03_gpt2_architecture_and_optimizations.ipynb`](../notebooks/03_gpt2_architecture_and_optimizations.ipynb)

---

## 9. Sampling

### What it is

At inference time the model returns logits for the next token. Sampling converts those logits into a token:

1. divide by temperature,
2. optionally keep the top `k` tokens,
3. optionally keep the nucleus of probability `p`,
4. softmax to probabilities,
5. sample with `torch.multinomial`.

### Why each control exists

| Control | Effect |
|---------|--------|
| temperature < 1 | sharper distribution, more repetitive and confident |
| temperature > 1 | flatter distribution, more random and creative |
| top-k | removes the long tail of unlikely tokens |
| top-p | keeps the smallest set of tokens that covers probability `p` |

Greedy decoding (`argmax`) is deterministic but can loop. Sampling is stochastic and produces more varied text.

### Where to see it

- Code: [`GPT.generate`](../src/llm_from_scratch/model.py)
- Notebook: [`07_guided_tour.ipynb`](../notebooks/07_guided_tour.ipynb)

### Exercise

Sample the same prompt at temperatures `0.5`, `1.0`, and `1.5`. Keep the seed fixed. Which output is closest to the training data? Which is most surprising?

---

## 10. Data pipeline and sharding

### Small text files

For a small corpus:

1. read the text,
2. tokenize it with GPT-2 BPE,
3. split into train and validation,
4. create windows of `block_size` tokens,
5. shift by one to create targets.

### Large corpora

For billions of tokens:

1. stream documents without downloading the full dataset,
2. tokenize each document,
3. prepend an end-of-text token to mark a document boundary,
4. write flat `uint16` arrays to `.npy` shards,
5. memory-map one shard at a time during training.

`uint16` works because the GPT-2 vocabulary is below 65,536. Memory mapping avoids loading a 100M-token shard into RAM.

An **EOT token at the start** of each document tells the model "a new document begins here." In causal language modelling, what comes before a token matters most, so a leading boundary token is useful.

### Where to see it

- Code: [`TokenDataset`](../src/llm_from_scratch/data.py), [`DataLoaderLite`](../src/llm_from_scratch/data.py), [`ShardDataset`](../src/llm_from_scratch/data.py), [`tokenize_documents_to_shards`](../src/llm_from_scratch/data.py)
- Notebooks: [`05_data_pipeline_text.ipynb`](../notebooks/05_data_pipeline_text.ipynb), [`06_fineweb_edu_streaming.ipynb`](../notebooks/06_fineweb_edu_streaming.ipynb)

---

## 11. Scaling to GPT-2 124M

The model size is controlled by `n_layer`, `n_head`, and `n_embd`:

| Model | Layers | Heads | Embedding | Parameters |
|-------|--------|-------|-----------|------------|
| nano | 2 | 4 | 128 | 6.9M |
| tiny | 6 | 6 | 384 | 30.1M |
| GPT-2 small | 12 | 12 | 768 | 124.5M |
| GPT-2 medium | 24 | 16 | 1024 | 354M |
| GPT-2 large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1.5B |

The training recipe scales with the model:

- effective batch size grows,
- learning rate is tuned,
- gradient accumulation increases,
- mixed precision and fused kernels matter more,
- data must be much larger and cleaner.

A full GPT-2 124M pretraining run is not completed in this repository. The notebooks show the full recipe and a small demonstration. The FineWeb-Edu pipeline can prepare the data, but the compute budget is the limiting factor.

---

## 12. Where each concept lives

| Concept | Notebook | Code | Test |
|---------|----------|------|------|
| BPE | 01 | `tokenizer.py` | `test_tokenizer.py` |
| Next-token data | 05, 07 | `data.py` | `test_data.py` |
| Attention | 02 | `model.py::CausalSelfAttention` | `test_model.py` |
| Causality | 02, 07 | `model.py` | `test_causal_masking` |
| Block / MLP | 03 | `model.py::Block`, `MLP` | `test_model.py` |
| Weight tying | 03 | `model.py::GPT.__init__` | `test_weight_tying` |
| Initialization | 03 | `model.py::GPT.__init__` | - |
| Optimizer | 03 | `model.py::configure_optimizers` | `test_optimizer_parameter_groups` |
| LR schedule | 03 | `train.py::get_lr` | `test_train.py` |
| Gradient accumulation | 03, 04 | `train.py` | `test_train.py` |
| Sharding | 06 | `data.py` | `test_data.py` |
| Sampling | 03, 07 | `model.py::generate` | `test_generation_shapes` |

---

## 13. Learning checklist

You understand the core when you can answer these without notes:

- Why does BPE start from bytes instead of characters?
- Why does the split pattern put the leading space inside the token?
- Why are position embeddings added to token embeddings?
- Why does attention divide by `sqrt(head_dim)`?
- What exactly does the causal mask hide?
- Why are there residual connections around attention and the MLP?
- Why is `LayerNorm` applied before the sub-layer in GPT-2?
- Why do the token embedding and LM head share weights?
- Why is the vocabulary padded to 50,304?
- Why is the loss shifted by one token?
- Why divide the loss by the number of gradient accumulation steps?
- Why does gradient clipping help?
- What is the difference between top-k and top-p sampling?
- Why do shards use `uint16` and memory mapping?

If any answer is unclear, open the linked notebook or module and run a small experiment.

---

## 14. Common misunderstandings

**"The model reads one token at a time during training."**
No. Training processes a whole window in parallel. The causal mask enforces the left-to-right constraint. Generation is sequential because each new token becomes part of the next input.

**"Attention understands meaning."**
Attention is a learned weighted average. Meaning emerges from training on next-token prediction, not from the attention formula alone.

**"A larger vocabulary is always better."**
A larger vocabulary shortens sequences but increases the embedding and output matrices. It also changes how often rare words are split. The best size depends on the data and compute budget.

**"Lower training loss means a better model."**
Training loss can fall while validation loss rises. Always evaluate on held-out text.

**"Temperature changes the model."**
Temperature only changes the sampling distribution at inference time. The weights are unchanged.

**"BPE learns words."**
BPE learns frequency-based merges, not linguistic units. It will happily create tokens that span morphemes or punctuation.

**"FlashAttention is a different model."**
It is the same attention math with a more memory-efficient implementation. Outputs should match within numerical tolerance.
