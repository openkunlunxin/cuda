# cuda path
CUDA_PATH ?= /usr/local/cuda-12.8

# compile
NVCC       := $(CUDA_PATH)/bin/nvcc

# Profile
NCU        := $(CUDA_PATH)/bin/ncu

# target
TARGET     := memst_con

# GPU
GPU_MODE   := h100

# src
SRC        := $(TARGET).cu

# PRO_FILE
PROF_FILE  := $(TARGET)_$(GPU_MODE).ncu-rep

# arch, 5090
ARCH       := sm_90

# NVCC flags
NVCC_FLAGS := -arch=$(ARCH) -Xptxas -O3 --maxrregcount=64 -keep -lineinfo

# NCU flags
NCU_FLAGS  := --set full --target-processes all

.PHONY: all clean

all: $(TARGET)

ptx: $(TARGET).ptx

prof_run: $(PROF_FILE)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

$(TARGET).ptx: $(SRC)
	$(NVCC) -ptx $(NVCC_FLAGS) $^ -o $@

$(PROF_FILE): $(TARGET)
	$(NCU) $(NCU_FLAGS) -o $@ ./$^

run: $(TARGET)
	./$^

clean:
	rm -f $(TARGET) *.o *.ptx *.ncu-rep