# Windows CUDA Graphics Demos Makefile
# Uses NVCC with MSVC as host compiler (required on Windows)
# Target: RTX 3080 (Compute Capability 8.6)
#
# IMPORTANT: This Makefile must be run from a Visual Studio Developer Command Prompt
# or after running vcvarsall.bat to set up the MSVC environment.
#
# Use the provided build.bat script which handles this automatically.

# Compiler
NVCC = nvcc

# NVCC flags
# -arch: Target GPU architecture (sm_86 for RTX 3080)
# -allow-unsupported-compiler: Allow VS 2025 (not yet officially supported by CUDA 12.9)
# Note: CUDA on Windows requires MSVC as the host compiler
NVCCFLAGS = -O3 -arch=sm_86 -allow-unsupported-compiler

# Windows libraries (MSVC style)
LIBS = gdi32.lib user32.lib

# All demos to build
DEMOS = cuda_render cuda_particles cuda_mandelbrot cuda_3d_cube cuda_fluid \
        cuda_raymarcher cuda_nbody cuda_primitives cuda_functions cuda_life \
        cuda_perlin cuda_flame cuda_tunnel cuda_cornell cuda_corridor \
        cuda_fractal cuda_pyramid cuda_teapot cuda_planets

.PHONY: all clean help

all: $(DEMOS)
	@echo === All demos built successfully! ===
	@echo Run any demo with: demo_name.exe

# Build rules for each demo
cuda_render: cuda_render.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_render.exe cuda_render.cu $(LIBS)

cuda_particles: cuda_particles.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_particles.exe cuda_particles.cu $(LIBS)

cuda_mandelbrot: cuda_mandelbrot.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_mandelbrot.exe cuda_mandelbrot.cu $(LIBS)

cuda_3d_cube: cuda_3d_cube.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_3d_cube.exe cuda_3d_cube.cu $(LIBS)

cuda_fluid: cuda_fluid.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_fluid.exe cuda_fluid.cu $(LIBS)

cuda_raymarcher: cuda_raymarcher.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_raymarcher.exe cuda_raymarcher.cu $(LIBS)

cuda_nbody: cuda_nbody.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_nbody.exe cuda_nbody.cu $(LIBS)

cuda_primitives: cuda_primitives.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_primitives.exe cuda_primitives.cu $(LIBS)

cuda_functions: cuda_functions.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_functions.exe cuda_functions.cu $(LIBS)

cuda_life: cuda_life.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_life.exe cuda_life.cu $(LIBS)

cuda_perlin: cuda_perlin.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_perlin.exe cuda_perlin.cu $(LIBS)

cuda_flame: cuda_flame.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_flame.exe cuda_flame.cu $(LIBS)

cuda_tunnel: cuda_tunnel.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_tunnel.exe cuda_tunnel.cu $(LIBS)

cuda_cornell: cuda_cornell.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_cornell.exe cuda_cornell.cu $(LIBS)

cuda_corridor: cuda_corridor.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_corridor.exe cuda_corridor.cu $(LIBS)

cuda_fractal: cuda_fractal.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_fractal.exe cuda_fractal.cu $(LIBS)

cuda_pyramid: cuda_pyramid.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_pyramid.exe cuda_pyramid.cu $(LIBS)

cuda_teapot: cuda_teapot.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_teapot.exe cuda_teapot.cu $(LIBS)

cuda_planets: cuda_planets.cu win32_display.h
	$(NVCC) $(NVCCFLAGS) -o cuda_planets.exe cuda_planets.cu $(LIBS)

clean:
	@if exist *.exe del /Q *.exe
	@if exist *.obj del /Q *.obj
	@if exist *.exp del /Q *.exp
	@if exist *.lib del /Q *.lib

help:
	@echo CUDA Graphics Demos for Windows (NVCC/MSVC Toolchain)
	@echo.
	@echo IMPORTANT: Run this from a Visual Studio Developer Command Prompt
	@echo or use the build.bat script.
	@echo.
	@echo Available targets:
	@echo   all           - Build all demos
	@echo   clean         - Remove all build artifacts
	@echo   cuda_XXX      - Build a specific demo
	@echo.
	@echo Available demos:
	@echo   cuda_render      - Plasma effect
	@echo   cuda_particles   - Particle system with trails
	@echo   cuda_mandelbrot  - Interactive fractal explorer
	@echo   cuda_3d_cube     - Rotating 3D cube
	@echo   cuda_fluid       - Fluid simulation
	@echo   cuda_raymarcher  - Raymarched scenes
	@echo   cuda_nbody       - N-body gravitational simulation
	@echo   cuda_primitives  - Basic shape rendering
	@echo   cuda_functions   - Mathematical function visualizer
	@echo   cuda_life        - Conway's Game of Life
	@echo   cuda_perlin      - Perlin noise visualization
	@echo   cuda_flame       - Fire effect
	@echo   cuda_tunnel      - Tunnel flythrough
	@echo   cuda_cornell     - Cornell box raytracing
	@echo   cuda_corridor    - Corridor renderer
	@echo   cuda_fractal     - Fractal renderer
	@echo   cuda_pyramid     - 3D pyramid
	@echo   cuda_teapot      - Utah teapot
	@echo.
	@echo GPU Target: RTX 3080 (sm_86)
