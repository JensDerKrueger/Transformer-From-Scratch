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
- supervised weather answers with a closed word vocabulary and held-out combinations

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
make weather-train BUILD=release
make weather-ask BUILD=release
make weather-interactive BUILD=release
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

## A small task with verifiable answers

The general byte-level corpus demo is still too small to promise useful language
completion. The weather experiment instead trains the same Transformer implementation
on a deliberately restricted task, without downloading a corpus:

```sh
make weather-train BUILD=release
make weather-ask BUILD=release WEATHER_PROMPT="In Berlin gibt es Regen. Was brauche ich? Antwort:"
make weather-ask BUILD=release WEATHER_PROMPT="In Berlin gibt es Sonne. Was brauche ich? Antwort:"
make weather-interactive BUILD=release
```

The observed answers to those two prompts after the default training were
`Nimm einen Schirm mit.` and `Nimm eine Sonnenbrille mit.`. The weather is
provided in the prompt: this experiment learns a recommendation, not a forecast
or current weather information.

The training set has 36 examples, using six cities (Berlin, Hamburg, Bonn,
Bremen, Kiel, Ulm), four conditions (Regen, Sonne, Schnee, Wolken), and two
questions (`Was brauche ich?` and `Was soll ich mitnehmen?`). Twelve examples
with held-out city/condition combinations are used only for evaluation.
Every individual word is present in the training vocabulary; the test measures
new combinations of known words, not unrestricted language understanding.

This separate demo uses a deterministic 32-token word/punctuation/EOS vocabulary,
32 tokens of context, and 7,208 parameters. Only response positions contribute to
the loss. Training uses teacher forcing; inference generates one token at a time
from model logits, using greedy selection and a learned EOS token. There is no
weather lookup or answer selection rule during inference. The condition/answer
table generates the supervised training data only.

The default 200 epochs perform 7,200 SGD updates. In the verified run, complete
held-out answers improved from 0/12 before training to 12/12 after training;
training answers were 36/36. Each exact comparison includes EOS. This small task
can train in seconds; duration alone does not measure learning.

`WEATHER_EPOCHS` changes the epoch count; each training invocation starts from
scratch. `WEATHER_CHECKPOINT` defaults to `build/$(BUILD)/weather.params`.
The weather binary is `build/$(BUILD)/weather_demo`, separate from earlier demos.
Its configuration and vocabulary are recreated deterministically by the program;
changing the dataset requires retraining. Weather and byte-model checkpoints
are incompatible. The interactive mode is line based and expects the demonstrated
sentence forms, capitalization, and vocabulary. Unknown words produce an explicit
error. It is not yet an open-domain question-answering system.

To check the response loss mask and backpropagation through the whole model:

```sh
make response-check BUILD=release
```

This checks tokenizer round trips, rejection of unknown words, and 80 sampled
parameter derivatives against finite differences across all parameter tensors.
