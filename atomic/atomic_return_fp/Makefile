# cuda path
CUDA_PATH ?= /usr/local/cuda

# compile
NVCC       := $(CUDA_PATH)/bin/nvcc

# Profile
NCU        := $(CUDA_PATH)/bin/ncu

# target
TARGET     := atomic_return

# src
SRC        := $(TARGET).cu

# PRO_FILE
PROF_FILE  := $(TARGET).ncu-rep

# arch, h100
ARCH       := sm_90

# NVCC flags
NVCC_FLAGS := -arch=$(ARCH) -O0

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
