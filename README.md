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
make ipq
make ipq-benchmark
```

The corpus file is not part of this repository. By default the makefile expects
`data/leipzig1M.txt`; see `data/README.md` for a reproducible download recipe.
