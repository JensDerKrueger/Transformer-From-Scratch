# Transformer From Scratch

Educational C++ implementation of Transformer building blocks from scratch.

The project intentionally uses only modern C++ and the standard library. The
current code covers:

- byte tokenization
- corpus reading and byte statistics
- a simple BPE tokenizer/trainer
- an indexed priority queue used for faster BPE experiments
- a small tensor foundation with shape, strides and flat storage
- matrix multiplication and basic tensor operations
- token and position embeddings
- linear layers
- masked self-attention in small, inspectable steps
- residual connections and layer normalization
- GELU activation and feed-forward networks
- a first decoder block and language-model logits
- final softmax, top-k selection and cross-entropy loss
- trainable parameters, linear-layer backpropagation and a first SGD step
- trainable embeddings, LayerNorm and Attention backward passes
- a tiny decoder-only language model with corpus training, checkpoints and terminal suggestions

The reusable implementation lives in `include/tfs`. Each lesson has its own
small program in `examples`, so later lessons do not change the source shown by
earlier lessons.

## Build

```sh
make
```

## Demos

```sh
make run
make corpus
make bpe
make bpe-fast
make bpe-compare
make tensor
make matmul
make embedding
make position
make linear
make qkv
make attention-scores
make attention-weights
make attention-output
make attention-output-projection
make attention-residual
make layer-norm
make activation
make feed-forward
make feed-forward-block
make decoder-block
make logits
make softmax
make cross-entropy
make parameters
make linear-backward
make sgd-step
make trainable-embedding
make initialization
make token-windows
make layernorm-backward
make attention-backward
make decoder-backward
make tiny-lm
make train-lm
make checkpoint
make suggestions
make ipq
make ipq-benchmark
```

To train the tiny byte-level model on the Leipzig corpus and use the resulting
checkpoint for suggestions:

```sh
make train-lm BUILD=release TRAIN_BYTES=1000000 TRAIN_STEPS=0 CHECKPOINT=build/release/leipzig1M.params
make suggestions BUILD=release CHECKPOINT=build/release/leipzig1M.params SUGGEST_BYTES=1000000 SUGGEST_CANDIDATES=1000 PROMPT="der film war "
```

`TRAIN_STEPS` counts optimizer updates, not corpus bytes. Set it to `0` to
train one full pass over all windows in the loaded corpus. Positive values train
that many randomly sampled windows.

`make suggestions` loads the checkpoint and extracts frequent candidate words
from the corpus. The explicit toy demo is still available with:

```sh
./build/debug/tfs_demo --toy "der film war "
```

The corpus file is not part of this repository. By default the makefile expects
`data/leipzig1M.txt`; see `data/README.md` for a reproducible download recipe.
