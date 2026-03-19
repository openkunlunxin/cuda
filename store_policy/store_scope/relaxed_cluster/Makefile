# cuda path
CUDA_PATH ?= /usr/local/cuda

# compile
NVCC       := $(CUDA_PATH)/bin/nvcc

# Profile
NCU        := $(CUDA_PATH)/bin/ncu

# target
TARGET     := store_scope

# src
SRC        := $(TARGET).cu

# GPU MODE
GPU_MODE   := h100

# PRO_FILE
PROF_FILE  := $(TARGET)_$(GPU_MODE).ncu-rep

# arch, h100
ARCH       := sm_90

# NVCC flags
NVCC_FLAGS := -std=c++17 -arch=$(ARCH) -O0 -I.

# NCU flags
NCU_FLAGS  := --set full --target-processes all

.PHONY: all clean

all: $(TARGET)

ptx: $(TARGET).ptx

sass: $(TARGET).sass

prof_run: $(PROF_FILE)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(TARGET).ptx: $(SRC)
	$(NVCC) -ptx $(NVCC_FLAGS) $^ -o $@

$(TARGET).sass: $(TARGET)
	cuobjdump -sass $< > $@

$(PROF_FILE): $(TARGET)
	$(NCU) $(NCU_FLAGS) -o $@ ./$^

run: $(TARGET)
	./$^

clean:
	rm -f $(TARGET) *.o $(TARGET).ptx $(TARGET)_$(GPU_MODE).ncu-rep $(TARGET).sass
