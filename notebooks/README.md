# 📓 Interactive Notebooks Curriculum

Each notebook in this curriculum is designed to be **completely standalone, self-explanatory, and runnable directly in Google Colab** (with free CPU/T4 GPU runtime) or on a local machine.

The notebooks are organized in logical progression—from fundamental data representation (Unicode & BPE) to self-attention math, modern Transformer architecture, optimization strategies, and multi-billion-token data pipelines.

---

## 🗺️ Learning Curriculum & Colab Links

| # | Notebook | Core Questions & Key Concepts | Colab Launcher | Runtime |
|---|----------|-------------------------------|:--------------:|:-------:|
| **00** | [Start Here](00_start_here.ipynb) | Repository tour, roadmap, design principles, environment setup | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/00_start_here.ipynb) | ~5 min |
| **01** | [Tokenizer from Scratch](01_tokenizer_from_scratch.ipynb) | Unicode code points, UTF-8 byte encoding, Byte Pair Encoding (BPE) merge statistics, GPT-2/GPT-4 regex split patterns, compression ratio, tiktoken parity | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/01_tokenizer_from_scratch.ipynb) | ~45 min |
| **02** | [Bigram to Self-Attention](02_bigram_to_self_attention.ipynb) | From Bigram LM to context aggregation: bag-of-words averaging, Query/Key/Value formulation, causal masking matrix, Multi-Head Attention, residual connections, LayerNorm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/02_bigram_to_self_attention.ipynb) | ~60 min |
| **03** | [GPT-2 Architecture & Optimizations](03_gpt2_architecture_and_optimizations.ipynb) | Complete GPT-2 (124M) reproduction: CausalSelfAttention, MLP with GELU, Pre-LayerNorm, weight tying ($W_{emb}=W_{head}$), depth-scaled initialization ($\sigma=0.02/\sqrt{2N}$), loading pretrained OpenAI weights, FlashAttention (`F.scaled_dot_product_attention`), `torch.compile`, mixed precision `bfloat16`, vocab padding to 50,304 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/03_gpt2_architecture_and_optimizations.ipynb) | ~90 min |
| **04** | [Train GPT-2 on Tiny Shakespeare](04_train_gpt2_on_tiny_shakespeare.ipynb) | Full training loop: micro-batching, gradient accumulation to simulate 0.5M tokens/step, AdamW with 2D-tensor weight decay splitting, linear warmup + cosine decay schedule, gradient clipping ($\|\mathbf{g}\|_2 \le 1.0$), validation tracking, autoregressive text generation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/04_train_gpt2_on_tiny_shakespeare.ipynb) | ~45 min |
| **05** | [Data Pipeline for Text](05_data_pipeline_text.ipynb) | Turning raw text corpora into next-token training batches: tiktoken integration, document demarcation with `<\|endoftext\|>`, memory-efficient `DataLoaderLite`, inputs/targets shifted by 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/05_data_pipeline_text.ipynb) | ~25 min |
| **06** | [FineWeb-Edu Streaming & Sharding](06_fineweb_edu_streaming.ipynb) | Scaling to multi-billion-token pretraining corpora: streaming `HuggingFaceFW/fineweb-edu` (sample-10BT), sharding into flat `uint16` `.npy` binary arrays, memory-mapped reading via PyTorch `IterableDataset` with multi-worker support | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/06_fineweb_edu_streaming.ipynb) | ~40 min |
| **07** | [Guided Tour of Reference Package](07_guided_tour.ipynb) | End-to-end synthesis connecting the standalone notebooks to the modular `src/llm_from_scratch` library: training a BPE tokenizer, building batches, verifying causal invariance, inspecting parameter allocations, running training, and sampling with temperature & top-k | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mostafa-ja/LLM_from_scratch/blob/main/notebooks/07_guided_tour.ipynb) | ~25 min |

> 💡 **Early Scratch Work**: The [`practice/`](practice/) directory contains early exploratory notebooks (e.g. [`practice/bigram_from_scratch_practice.ipynb`](practice/bigram_from_scratch_practice.ipynb)). They are preserved to demonstrate the iterative engineering process and code evolution.

---

## 🚀 How to Run in Google Colab

1. **One-Click Launch**: Click any **Open in Colab** badge in the table above.
2. **GPU Acceleration (Recommended for 03 & 04)**:
   - In Colab, navigate to **Runtime** > **Change runtime type**.
   - Select **T4 GPU** (available on free tier) and save.
3. **Automatic Setup**:
   - Notebooks 00 and 07 include a top setup cell that clones the repository and installs the package in editable mode (`!pip install -q -e .`).
   - Notebooks 01–06 automatically download required datasets (e.g. Tiny Shakespeare) or install extra libraries (`tiktoken`, `datasets`) inside their respective setup cells.

---

## 💻 How to Run Locally

Clone the repository and install in editable mode:

```bash
git clone https://github.com/mostafa-ja/LLM_from_scratch.git
cd LLM_from_scratch

# Create and activate a virtual environment
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

# Install package and all dependencies (including dev and viz)
pip install -e ".[all]"

# Launch Jupyter Lab or Notebook
jupyter lab
```

---

## 🎯 Pedagogical Guidelines: How to Get Maximum Learning

- **Run Every Cell**: Don't treat these as blog posts. Run the code, print the intermediate tensors, and inspect shapes.
- **Predict Before Running**: Before executing a cell that alters a tensor, write down its predicted shape: `(B, T, C)`, `(B, nh, T, hs)`, etc.
- **Break Things Intentionally**:
  - What happens if you disable the causal mask? (The model cheats by looking into the future and loss collapses artificially).
  - What happens if you skip dividing by $\sqrt{d_k}$ in attention? (Large dot products push softmax into extreme saturation with near-zero gradients).
  - What happens if you don't scale weights at residual projections? (Activation variance doubles at every layer, blowing up deep networks).
- **Bridge Notebooks to Modular Code**: After completing a notebook, open the corresponding source file in `src/llm_from_scratch/` to see how interactive prototypes transform into production-quality, typed, and unit-tested software.
