# Jetson Nano CUDA Graphics Demos Makefile

NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -arch=sm_53
LIBS = -lX11

SOURCES := $(wildcard *.cu)
DEMOS := $(SOURCES:.cu=)

.PHONY: all clean help

all: $(DEMOS)

# CUDA 10.2 needs C++11 for these demos' host-side clocks and framebuffers.
cuda_metamorphosis cuda_not_quite_home cuda_molecule: NVCCFLAGS += -std=c++11

%: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LIBS)

run-%: cuda_%
	./cuda_$*

clean:
	rm -f $(DEMOS)

help:
	@echo "make all          Build all CUDA demos"
	@echo "make cuda_life    Build an individual demo"
	@echo "make run-life     Build and run an individual demo"
	@echo "make clean        Remove all demo binaries"
