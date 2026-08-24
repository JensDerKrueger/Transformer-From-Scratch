# Transformer From Scratch

Educational C++ implementation of Transformer building blocks from scratch.

The project intentionally uses only modern C++ and the standard library. The
current code covers:

- byte tokenization
- corpus reading and byte statistics
- a simple BPE tokenizer/trainer
- an indexed priority queue used for faster BPE experiments

## Build

```sh
make
```

## Demos

```sh
make run
make corpus
make bpe
make ipq
make ipq-benchmark
```

The corpus file is not part of this repository. By default the makefile expects
`data/leipzig1M.txt`; see `data/README.md` for a reproducible download recipe.
