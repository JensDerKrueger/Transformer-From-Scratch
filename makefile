CXX ?= c++
OSTYPE := $(shell uname)
BUILD ?= debug

OUTDIR := build/$(BUILD)
OBJDIR := $(OUTDIR)/obj
DEPDIR := $(OUTDIR)/dep

TARGET := tfs_demo
TARGET_PATH := $(OUTDIR)/$(TARGET)
CORPUS ?= data/leipzig1M.txt
MAX_LINES ?= 1000
BPE_LINES ?= 2000
BPE_MERGES ?= 32
BPE_TEXT ?= the transformer learns from text
BENCH_LINES ?= 1000000
BENCH_CANDIDATES ?= 1000000
BENCH_ITERATIONS ?= 1000
COMPARE_LINES ?= 2000
COMPARE_MERGES ?= 128

SRC := src/main.cpp
OBJ := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SRC))
DEP := $(patsubst %.cpp,$(DEPDIR)/%.d,$(SRC))

INCLUDES := -Iinclude
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

.PHONY: all release run corpus bpe bpe-fast bpe-compare tensor matmul ipq ipq-benchmark clean mrproper

all: $(TARGET_PATH)

release:
	$(MAKE) BUILD=release all

run: $(TARGET_PATH)
	./$(TARGET_PATH)

corpus: $(TARGET_PATH)
	./$(TARGET_PATH) --corpus "$(CORPUS)" $(MAX_LINES)

bpe: $(TARGET_PATH)
	./$(TARGET_PATH) --bpe "$(CORPUS)" $(BPE_LINES) $(BPE_MERGES) "$(BPE_TEXT)"

bpe-fast: $(TARGET_PATH)
	./$(TARGET_PATH) --bpe-fast "$(CORPUS)" $(BPE_LINES) $(BPE_MERGES) "$(BPE_TEXT)"

bpe-compare: $(TARGET_PATH)
	./$(TARGET_PATH) --bpe-compare "$(CORPUS)" $(COMPARE_LINES) $(COMPARE_MERGES)

tensor: $(TARGET_PATH)
	./$(TARGET_PATH) --tensor

matmul: $(TARGET_PATH)
	./$(TARGET_PATH) --matmul

ipq: $(TARGET_PATH)
	./$(TARGET_PATH) --ipq

ipq-benchmark:
	$(MAKE) BUILD=release all
	./build/release/$(TARGET) --ipq-benchmark "$(CORPUS)" $(BENCH_LINES) $(BENCH_CANDIDATES) $(BENCH_ITERATIONS)

$(TARGET_PATH): $(OBJ) | $(OUTDIR)
	$(CXX) $(OBJ) $(LFLAGS) -o $@

$(OBJDIR)/%.o: %.cpp | $(OBJDIR) $(DEPDIR)
	@mkdir -p $(dir $@) $(dir $(patsubst $(OBJDIR)/%.o,$(DEPDIR)/%.d,$@))
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -MF $(patsubst $(OBJDIR)/%.o,$(DEPDIR)/%.d,$@) -c $< -o $@

$(OUTDIR):
	mkdir -p $@

$(OBJDIR): | $(OUTDIR)
	mkdir -p $@

$(DEPDIR): | $(OUTDIR)
	mkdir -p $@

clean:
	-rm -rf build

mrproper: clean

-include $(DEP)
