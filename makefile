CXX ?= c++
OSTYPE := $(shell uname)

BUILD ?= debug
OUTDIR := build/$(BUILD)
TARGET := tfs_demo
TARGET_PATH := $(OUTDIR)/$(TARGET)

CORPUS ?= data/leipzig1M.txt
MAX_LINES ?= 1000
BPE_LINES ?= 2000
BPE_MERGES ?= 32
BPE_TEXT ?= the transformer learns from text
COMPARE_LINES ?= 2000
COMPARE_MERGES ?= 128
BENCH_LINES ?= 1000000
BENCH_CANDIDATES ?= 1000000
BENCH_ITERATIONS ?= 1000

INCLUDES := -Iinclude -Iexamples
WARNINGS := -Wall -Wextra -Wpedantic -Wunreachable-code
CPPSTD := -std=c++20

ifeq ($(BUILD),release)
  OPTFLAGS := -O3 -DNDEBUG
else
  OPTFLAGS := -g -O0
endif

ifeq ($(OSTYPE),Linux)
  CXXFLAGS := $(CPPSTD) $(WARNINGS) $(OPTFLAGS)
  LFLAGS :=
else
  CXXFLAGS := $(CPPSTD) $(WARNINGS) $(OPTFLAGS)
  LFLAGS :=
endif

define build_example
	@mkdir -p $(OUTDIR)
	@$(CXX) $(CXXFLAGS) $(INCLUDES) examples/$(1).cpp $(LFLAGS) -o $(TARGET_PATH)
endef

.PHONY: all release run corpus bpe bpe-fast bpe-compare ipq ipq-benchmark
.PHONY: tensor matmul embedding position linear qkv attention-scores attention-weights
.PHONY: attention-output attention-output-projection attention-residual layer-norm
.PHONY: activation feed-forward feed-forward-block decoder-block logits softmax cross-entropy
.PHONY: parameters linear-backward sgd-step clean mrproper

all: run

release:
	$(MAKE) BUILD=release run

run:
	$(call build_example,01_byte_tokenizer)
	./$(TARGET_PATH)

corpus:
	$(call build_example,02_corpus_reader)
	./$(TARGET_PATH) --corpus "$(CORPUS)" $(MAX_LINES)

bpe:
	$(call build_example,03_bpe_tokenizer)
	./$(TARGET_PATH) --bpe "$(CORPUS)" $(BPE_LINES) $(BPE_MERGES) "$(BPE_TEXT)"

bpe-fast:
	$(call build_example,04_bpe_fast)
	./$(TARGET_PATH) --bpe-fast "$(CORPUS)" $(BPE_LINES) $(BPE_MERGES) "$(BPE_TEXT)"

bpe-compare:
	$(call build_example,04_bpe_compare)
	./$(TARGET_PATH) --bpe-compare "$(CORPUS)" $(COMPARE_LINES) $(COMPARE_MERGES)

ipq:
	$(call build_example,04a_indexed_priority_queue)
	./$(TARGET_PATH) --ipq

ipq-benchmark:
	@mkdir -p build/release
	@$(CXX) $(CPPSTD) $(WARNINGS) -O3 -DNDEBUG $(INCLUDES) examples/04a_indexed_priority_queue.cpp $(LFLAGS) -o build/release/$(TARGET)
	./build/release/$(TARGET) --ipq-benchmark "$(CORPUS)" $(BENCH_LINES) $(BENCH_CANDIDATES) $(BENCH_ITERATIONS)

tensor:
	$(call build_example,05_tensor)
	./$(TARGET_PATH) --tensor

matmul:
	$(call build_example,06_matrix_multiplication)
	./$(TARGET_PATH) --matmul

embedding:
	$(call build_example,07_token_embedding)
	./$(TARGET_PATH) --embedding

position:
	$(call build_example,08_position_embedding)
	./$(TARGET_PATH) --position

linear:
	$(call build_example,09_linear_layer)
	./$(TARGET_PATH) --linear

qkv:
	$(call build_example,10_attention_projections)
	./$(TARGET_PATH) --qkv

attention-scores:
	$(call build_example,11_attention_scores)
	./$(TARGET_PATH) --attention-scores

attention-weights:
	$(call build_example,12_attention_weights)
	./$(TARGET_PATH) --attention-weights

attention-output:
	$(call build_example,13_attention_output)
	./$(TARGET_PATH) --attention-output

attention-output-projection:
	$(call build_example,14_attention_output_projection)
	./$(TARGET_PATH) --attention-output-projection

attention-residual:
	$(call build_example,15_residual_connection)
	./$(TARGET_PATH) --attention-residual

layer-norm:
	$(call build_example,16_layer_norm)
	./$(TARGET_PATH) --layer-norm

activation:
	$(call build_example,17_gelu_activation)
	./$(TARGET_PATH) --activation

feed-forward:
	$(call build_example,18_feed_forward)
	./$(TARGET_PATH) --feed-forward

feed-forward-block:
	$(call build_example,19_feed_forward_block)
	./$(TARGET_PATH) --feed-forward-block

decoder-block:
	$(call build_example,20_decoder_block)
	./$(TARGET_PATH) --decoder-block

logits:
	$(call build_example,21_logits)
	./$(TARGET_PATH) --logits

softmax:
	$(call build_example,22_softmax_topk)
	./$(TARGET_PATH) --softmax

cross-entropy:
	$(call build_example,23_cross_entropy)
	./$(TARGET_PATH) --cross-entropy

parameters:
	$(call build_example,24_parameters)
	./$(TARGET_PATH) --parameters

linear-backward:
	$(call build_example,25_linear_backward)
	./$(TARGET_PATH) --linear-backward

sgd-step:
	$(call build_example,26_sgd_training_step)
	./$(TARGET_PATH) --sgd-step

clean:
	rm -rf build/debug build/release

mrproper: clean
