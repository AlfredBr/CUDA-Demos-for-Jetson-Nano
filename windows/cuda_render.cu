/*
 * Windows CUDA Graphics Demo - Plasma Effect
 * Ported from Jetson Nano version
 *
 * Hardware-accelerated rendering using CUDA
 * Uses Win32 API for display (no Visual Studio required)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600

// CUDA kernel to render a colorful animated pattern
__global__ void renderKernel(unsigned char* pixels, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;  // BGRA format

    // Normalized coordinates
    float u = (float)x / width;
    float v = (float)y / height;

    // Create animated plasma effect
    float cx = u - 0.5f + sinf(time * 0.5f) * 0.3f;
    float cy = v - 0.5f + cosf(time * 0.7f) * 0.3f;

    float d1 = sqrtf(cx * cx + cy * cy);
    float d2 = sinf(u * 10.0f + time) * 0.5f + 0.5f;
    float d3 = cosf(v * 10.0f + time * 1.3f) * 0.5f + 0.5f;
    float d4 = sinf((u + v) * 5.0f + time * 0.8f) * 0.5f + 0.5f;

    float plasma = (d1 + d2 + d3 + d4) / 4.0f;

    // Color mapping with time-based hue shift
    float hue = plasma + time * 0.1f;
    hue = hue - floorf(hue);  // Keep in [0, 1]

    // HSV to RGB conversion (simplified)
    float h = hue * 6.0f;
    float c = 1.0f;
    float x_val = c * (1.0f - fabsf(fmodf(h, 2.0f) - 1.0f));

    float r, g, b;
    if (h < 1) { r = c; g = x_val; b = 0; }
    else if (h < 2) { r = x_val; g = c; b = 0; }
    else if (h < 3) { r = 0; g = c; b = x_val; }
    else if (h < 4) { r = 0; g = x_val; b = c; }
    else if (h < 5) { r = x_val; g = 0; b = c; }
    else { r = c; g = 0; b = x_val; }

    // Write BGRA
    pixels[idx + 0] = (unsigned char)(b * 255);  // Blue
    pixels[idx + 1] = (unsigned char)(g * 255);  // Green
    pixels[idx + 2] = (unsigned char)(r * 255);  // Red
    pixels[idx + 3] = 255;                        // Alpha
}

int main() {
    printf("=== Windows CUDA Graphics Demo ===\n");
    printf("Resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("Press Q or Escape to exit\n\n");

    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);
    printf("Memory: %.0f MB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0));

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA Native Rendering", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate host memory for pixels (pinned for faster transfer)
    unsigned char* h_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);

    // Allocate device memory
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    // CUDA grid configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    printf("Rendering with CUDA...\n");
    printf("Block size: %dx%d, Grid size: %dx%d\n\n",
           blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    int frameCount = 0;
    double lastFpsTime = win32_get_time(display);

    // Main render loop
    while (!win32_should_close(display)) {
        // Process Windows messages
        if (win32_process_events(display)) break;

        // Handle events
        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                if (event.key == XK_Escape || event.key == XK_q) {
                    goto cleanup;
                }
            }
        }

        float time = (float)win32_get_time(display);

        // Render on GPU
        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT, time);

        // Copy result to host
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

        // Display
        win32_blit_pixels(display, h_pixels);

        frameCount++;

        // FPS counter
        double now = win32_get_time(display);
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f (GPU-rendered frames)\n", frameCount / (now - lastFpsTime));
            frameCount = 0;
            lastFpsTime = now;
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);
    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
