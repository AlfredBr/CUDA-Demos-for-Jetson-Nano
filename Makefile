# Jetson Nano CUDA Graphics Demos Makefile

NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -O3 -arch=sm_53
LIBS = -lX11

DEMOS = cuda_render cuda_particles cuda_mandelbrot cuda_3d_cube cuda_fluid cuda_raymarcher cuda_nbody cuda_primitives

.PHONY: all clean help

all: $(DEMOS)

cuda_render: cuda_render.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_particles: cuda_particles.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_mandelbrot: cuda_mandelbrot.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_3d_cube: cuda_3d_cube.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_fluid: cuda_fluid.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_raymarcher: cuda_raymarcher.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_nbody: cuda_nbody.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

cuda_primitives: cuda_primitives.cu
	\$(NVCC) \$(NVCCFLAGS) -o \$@ \$< \$(LIBS)

run-%: cuda_%
	./cuda_$*

clean:
	rm -f $(DEMOS)
