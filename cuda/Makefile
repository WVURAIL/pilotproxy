# ==============================================================================
# F-Statistic CUDA Kernel Makefile
# ==============================================================================
#
# Build targets:
#   make           - Build shared library (release)
#   make debug     - Build with debug symbols and bounds checking
#   make clean     - Remove build artifacts
#   make info      - Print configuration
#
# Configuration:
#   GPU architecture: auto-detected via nvidia-smi; override with make SM=86
#   Override compiler:            make NVCC=/usr/local/cuda/bin/nvcc
#
# ==============================================================================

# ------------------------------------------------------------------------------
# Toolchain
# ------------------------------------------------------------------------------
NVCC                  ?= nvcc
CC                    ?= gcc
CXX                   ?= g++
# Auto-detect the arch of the first visible GPU (drivers >= 460 report
# compute_cap). Empty on GPU-less hosts; guarded below so only goals that
# compile device code require it. Override for cross-compiles: make SM=90
SM                    ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. \r')
DP4A                  ?= 1
DETECTOR_WINDOW_SAMPLES ?= 128
BLOCK_THREADS         ?= 64
GRID_MAX_BLOCKS       ?= 4096
CONSTANT_WEIGHT_LANES ?= 0
SHARED_WEIGHT_LANES   ?= 1

# ------------------------------------------------------------------------------
# Compiler Flags
# ------------------------------------------------------------------------------
NVCC_BASE   := -std=c++14 -Xcompiler -fPIC --shared

# Architecture flags
ARCH_FLAGS  := -arch=sm_$(SM)

# Warning flags
WARN_FLAGS  := -Xcompiler -Wall

# Configuration defines shared by the library and CUDA regression test.
CONFIG_DEFS := \
	-DFSTAT_USE_DP4A=$(DP4A) \
	-DFSTAT_DETECTOR_WINDOW_SAMPLES=$(DETECTOR_WINDOW_SAMPLES) \
	-DFSTAT_BLOCK_THREADS=$(BLOCK_THREADS) \
	-DFSTAT_GRID_MAX_BLOCKS=$(GRID_MAX_BLOCKS) \
	-DFSTAT_USE_CONSTANT_WEIGHT_LANES=$(CONSTANT_WEIGHT_LANES) \
	-DFSTAT_USE_SHARED_WEIGHT_LANES=$(SHARED_WEIGHT_LANES)

# Release flags (default)
OPT_FLAGS   := -O3 -DNDEBUG -use_fast_math -lineinfo $(CONFIG_DEFS)

# Debug flags
DEBUG_FLAGS := -O0 -g -G -DDEBUG $(CONFIG_DEFS)

# Combined flags
RELEASE_FLAGS := $(NVCC_BASE) $(ARCH_FLAGS) $(OPT_FLAGS) $(WARN_FLAGS)
DEBUG_BUILD   := $(NVCC_BASE) $(ARCH_FLAGS) $(DEBUG_FLAGS) $(WARN_FLAGS)

# ------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------
override TARGET   := libfstatistic.so
override DEBUG_TARGET := libfstatistic_debug.so
SOURCES           := f_statistic.cu
HEADERS           := f_statistic.h config.h
REF_TEST          := test_fstat_reference
REF_TEST_SOURCES  := f_statistic_reference.cpp tests/test_fstat_reference.cpp
REF_TEST_HEADERS  := f_statistic_reference.h config.h
C_HEADER_TEST_OBJ := test_c_header.o
CUDA_TEST         := test_fstat_cuda
CUDA_TEST_SOURCES := tests/test_fstat_cuda.cpp f_statistic_reference.cpp
CUDA_TEST_HEADERS := f_statistic.h f_statistic_reference.h config.h

# ------------------------------------------------------------------------------
# Build-configuration tracking
# ------------------------------------------------------------------------------
# Goals that compile device code need a concrete SM; everything else (clean,
# test_ref, test_c_header, help, ...) must keep working on GPU-less hosts.
SM_REQUIRED_GOALS := all debug test test_cuda test_cuda_grid_stride $(TARGET) $(DEBUG_TARGET) $(CUDA_TEST)
ifeq ($(strip $(SM)),)
ifneq ($(filter $(SM_REQUIRED_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)),)
$(error no GPU detected (nvidia-smi absent or predates the compute_cap query); pass SM=<arch> explicitly, e.g. make SM=80 for A100)
endif
endif

# Fail before the recipe if the requested CUDA compiler is not executable. This
# catches stale hard-coded paths such as /usr/local/cuda/bin/nvcc when nvcc is
# actually installed elsewhere on PATH.
NVCC_REQUIRED_GOALS := $(SM_REQUIRED_GOALS)
NVCC_RESOLVED := $(shell command -v "$(NVCC)" 2>/dev/null)
ifeq ($(strip $(NVCC_RESOLVED)),)
ifneq ($(filter $(NVCC_REQUIRED_GOALS),$(if $(MAKECMDGOALS),$(MAKECMDGOALS),all)),)
$(error CUDA compiler not found: NVCC=$(NVCC). Run 'command -v nvcc' and either leave NVCC unset or pass NVCC=/absolute/path/to/nvcc)
endif
endif

# The arch and CONFIG_DEFS live only in recipe flags, invisible to make's
# timestamp check, so changing them used to reuse stale objects silently.
# Encode them and the build mode in stamp filenames: switching any knob
# invalidates the relevant output; release and debug builds remain independent.
CONFIG_KEY := sm$(SM)_dp4a$(DP4A)_k$(DETECTOR_WINDOW_SAMPLES)_bt$(BLOCK_THREADS)_gmb$(GRID_MAX_BLOCKS)_cwl$(CONSTANT_WEIGHT_LANES)_swl$(SHARED_WEIGHT_LANES)
RELEASE_CONFIG_STAMP := .config_release_$(CONFIG_KEY).stamp
DEBUG_CONFIG_STAMP := .config_debug_$(CONFIG_KEY).stamp

$(RELEASE_CONFIG_STAMP):
	@rm -f .config_release_*.stamp
	@touch $@

$(DEBUG_CONFIG_STAMP):
	@rm -f .config_debug_*.stamp
	@touch $@

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------
.PHONY: all debug clean info test_c_header test_ref test_cuda test_cuda_grid_stride test

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS) $(RELEASE_CONFIG_STAMP)
	$(NVCC) $(RELEASE_FLAGS) -o $@ $(SOURCES)
	@echo "Built $(TARGET) for SM $(SM)"

debug: $(DEBUG_TARGET)

$(DEBUG_TARGET): $(SOURCES) $(HEADERS) $(DEBUG_CONFIG_STAMP)
	$(NVCC) $(DEBUG_BUILD) -o $@ $(SOURCES)
	@echo "Built $(DEBUG_TARGET) for SM $(SM)"

$(REF_TEST): $(REF_TEST_SOURCES) $(REF_TEST_HEADERS) $(RELEASE_CONFIG_STAMP)
	$(CXX) -std=c++14 -O2 -Wall $(CONFIG_DEFS) -I. -o $@ $(REF_TEST_SOURCES)
	@echo "Built $(REF_TEST)"

$(C_HEADER_TEST_OBJ): tests/test_c_header.c f_statistic.h config.h $(RELEASE_CONFIG_STAMP)
	$(CC) -std=c11 -Wall $(CONFIG_DEFS) -I. -c tests/test_c_header.c -o $@
	@echo "Compiled C header smoke test"

test_c_header: $(C_HEADER_TEST_OBJ)

test_ref: $(REF_TEST)
	./$(REF_TEST)

$(CUDA_TEST): $(TARGET) $(CUDA_TEST_SOURCES) $(CUDA_TEST_HEADERS) $(RELEASE_CONFIG_STAMP)
	$(NVCC) -std=c++14 -O2 -I. $(ARCH_FLAGS) $(CONFIG_DEFS) $(CUDA_TEST_SOURCES) \
	    -L. -lfstatistic -Xlinker -rpath -Xlinker . -o $@
	@echo "Built $(CUDA_TEST)"

test_cuda: $(CUDA_TEST)
	./$(CUDA_TEST)

# Runs the CUDA regression at a deliberately tiny launch capacity to exercise
# the grid-stride paths. That build overwrites libfstatistic.so with a library
# whose capacity is 4 blocks, which then fails any later test needing more
# streams (the Python GPU suite uses 2048) with "num_streams exceeds the launch
# capacity" -- a confusing failure far from its cause. Restore the default
# build afterwards so the target cannot leave a degraded artifact behind.
test_cuda_grid_stride:
	$(MAKE) test_cuda GRID_MAX_BLOCKS=4
	@echo "Restoring default build (GRID_MAX_BLOCKS=$(GRID_MAX_BLOCKS))"
	$(MAKE) clean
	$(MAKE) all

test: test_c_header test_ref test_cuda

clean:
	rm -f $(TARGET) $(DEBUG_TARGET) $(REF_TEST) $(CUDA_TEST) $(C_HEADER_TEST_OBJ) .config_*.stamp
	@echo "Cleaned build artifacts"

info:
	@echo "========================================"
	@echo "F-Statistic Kernel Build Configuration"
	@echo "========================================"
	@echo "Compiler:     $(NVCC)"
	@echo "C compiler:   $(CC)"
	@echo "C++ compiler: $(CXX)"
	@echo "Target:       $(TARGET)"
	@echo "Debug target: $(DEBUG_TARGET)"
	@echo "SM Version:   $(if $(SM),$(SM),(not detected))"
	@echo "DP4A:         $(DP4A)"
	@echo "Window samples: $(DETECTOR_WINDOW_SAMPLES)"
	@echo "Block threads: $(BLOCK_THREADS)"
	@echo "Grid max blocks: $(GRID_MAX_BLOCKS)"
	@echo "Constant weight lanes: $(CONSTANT_WEIGHT_LANES)"
	@echo "Shared weight lanes: $(SHARED_WEIGHT_LANES)"
	@echo "Release:      $(RELEASE_FLAGS)"
	@echo "Debug:        $(DEBUG_BUILD)"
	@echo "========================================"

# ------------------------------------------------------------------------------
# Help
# ------------------------------------------------------------------------------
.PHONY: help
help:
	@echo "Usage: make [target] [options]"
	@echo ""
	@echo "Targets:"
	@echo "  all     Build release library (default)"
	@echo "  debug   Build libfstatistic_debug.so with debug symbols"
	@echo "  test    Build and run C header, CPU reference, and CUDA regression tests"
	@echo "  test_c_header"
	@echo "          Compile public C header smoke test"
	@echo "  test_ref"
	@echo "          Build and run CPU reference tests"
	@echo "  test_cuda"
	@echo "          Build and run CUDA-vs-CPU reference tests"
	@echo "  test_cuda_grid_stride"
	@echo "          Rebuild with GRID_MAX_BLOCKS=4 and run CUDA regression tests"
	@echo "  clean   Remove build artifacts"
	@echo "  info    Print build configuration"
	@echo "  help    Show this message"
	@echo ""
	@echo "Options:"
	@echo "  SM=XX   Set compute capability (default: auto-detected via nvidia-smi)"
	@echo "          Common values: 70 (V100), 80 (A100), 86 (RTX 30xx),"
	@echo "                         89 (RTX 40xx/Ada), 90 (H100)"
	@echo "  DETECTOR_WINDOW_SAMPLES=K"
	@echo "          Set detector-window samples (default: 128; supported: 64, 128)"
	@echo "  GRID_MAX_BLOCKS=N"
	@echo "          Set maximum power-accumulation grid blocks (default: 4096)"
	@echo "  CONSTANT_WEIGHT_LANES=0|1"
	@echo "          Store DP4A weight lanes in CUDA constant memory (default: 0)"
	@echo "  SHARED_WEIGHT_LANES=0|1"
	@echo "          Preload DP4A weight lanes into per-block shared memory (default: 1)"
	@echo ""
	@echo "Examples:"
	@echo "  make SM=89          Build for RTX 5000 Ada"
	@echo "  make debug SM=80    Debug build for A100"
