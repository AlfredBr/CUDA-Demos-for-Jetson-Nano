/*
 * Windows CUDA Mandelbrot Set Explorer
 * Ported from Jetson Nano version
 *
 * Interactive fractal viewer with zoom and pan
 * All fractal calculations done in parallel on the GPU!
 *
 * Controls:
 *   Left click  - Zoom in at cursor
 *   Right click - Zoom out
 *   Arrow keys  - Pan around
 *   R           - Reset to default view
 *   +/-         - Increase/decrease max iterations
 *   Q/Escape    - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600

// Mandelbrot calculation kernel
__global__ void mandelbrotKernel(unsigned char* pixels, int width, int height,
                                  double centerX, double centerY, double zoom,
                                  int maxIter, float colorOffset) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // Map pixel to complex plane
    double x0 = centerX + (px - width / 2.0) / zoom;
    double y0 = centerY + (py - height / 2.0) / zoom;

    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    // Mandelbrot iteration: z = z^2 + c
    while (x*x + y*y <= 4.0 && iter < maxIter) {
        double xtemp = x*x - y*y + x0;
        y = 2.0*x*y + y0;
        x = xtemp;
        iter++;
    }

    int idx = (py * width + px) * 4;

    if (iter == maxIter) {
        // Inside the set - black
        pixels[idx + 0] = 0;
        pixels[idx + 1] = 0;
        pixels[idx + 2] = 0;
    } else {
        // Outside - color based on escape time
        // Smooth coloring using continuous potential
        double log_zn = log(x*x + y*y) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        double smooth_iter = iter + 1.0 - nu;

        // Create color gradient
        float t = (float)(smooth_iter / maxIter);
        t = t + colorOffset;
        t = t - floorf(t);  // Keep in [0, 1]

        // Multi-color gradient
        float r, g, b;
        if (t < 0.16f) {
            float s = t / 0.16f;
            r = 0.0f; g = 0.0f; b = s;
        } else if (t < 0.33f) {
            float s = (t - 0.16f) / 0.17f;
            r = 0.0f; g = s; b = 1.0f;
        } else if (t < 0.5f) {
            float s = (t - 0.33f) / 0.17f;
            r = s; g = 1.0f; b = 1.0f - s;
        } else if (t < 0.67f) {
            float s = (t - 0.5f) / 0.17f;
            r = 1.0f; g = 1.0f - s * 0.5f; b = 0.0f;
        } else if (t < 0.83f) {
            float s = (t - 0.67f) / 0.16f;
            r = 1.0f; g = 0.5f - s * 0.5f; b = s;
        } else {
            float s = (t - 0.83f) / 0.17f;
            r = 1.0f - s; g = 0.0f; b = 1.0f - s;
        }

        pixels[idx + 0] = (unsigned char)(b * 255);
        pixels[idx + 1] = (unsigned char)(g * 255);
        pixels[idx + 2] = (unsigned char)(r * 255);
    }
    pixels[idx + 3] = 255;
}

int main() {
    printf("=== Windows CUDA Mandelbrot Explorer ===\n\n");
    printf("Controls:\n");
    printf("  Left click   - Zoom in at cursor\n");
    printf("  Right click  - Zoom out\n");
    printf("  Arrow keys   - Pan around\n");
    printf("  +/-          - More/fewer iterations (detail)\n");
    printf("  C            - Cycle colors\n");
    printf("  R            - Reset view\n");
    printf("  Q/Escape     - Quit\n\n");

    // CUDA device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA Mandelbrot - Click to zoom!", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate memory
    unsigned char* h_pixels;
    unsigned char* d_pixels;

    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    // CUDA grid configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Mandelbrot parameters
    double centerX = -0.5;
    double centerY = 0.0;
    double zoom = 200.0;
    int maxIter = 256;
    float colorOffset = 0.0f;

    int needsRedraw = 1;
    int mouseX = WIDTH / 2, mouseY = HEIGHT / 2;

    double lastFpsTime = win32_get_time(display);
    int frameCount = 0;

    printf("Rendering at %dx%d, %d max iterations\n", WIDTH, HEIGHT, maxIter);
    printf("Zoom: %.2e\n\n", zoom);

    // Main loop
    while (!win32_should_close(display)) {
        // Process Windows messages
        if (win32_process_events(display)) break;

        // Handle events
        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                if (event.key == XK_Escape || event.key == XK_q) goto cleanup;

                if (event.key == XK_r) {
                    centerX = -0.5;
                    centerY = 0.0;
                    zoom = 200.0;
                    maxIter = 256;
                    needsRedraw = 1;
                    printf("Reset to default view\n");
                }

                if (event.key == XK_Left) { centerX -= 50.0 / zoom; needsRedraw = 1; }
                if (event.key == XK_Right) { centerX += 50.0 / zoom; needsRedraw = 1; }
                if (event.key == XK_Up) { centerY -= 50.0 / zoom; needsRedraw = 1; }
                if (event.key == XK_Down) { centerY += 50.0 / zoom; needsRedraw = 1; }

                if (event.key == XK_plus || event.key == XK_equal) {
                    maxIter = (int)(maxIter * 1.5);
                    if (maxIter > 10000) maxIter = 10000;
                    needsRedraw = 1;
                    printf("Max iterations: %d\n", maxIter);
                }
                if (event.key == XK_minus) {
                    maxIter = (int)(maxIter / 1.5);
                    if (maxIter < 32) maxIter = 32;
                    needsRedraw = 1;
                    printf("Max iterations: %d\n", maxIter);
                }

                if (event.key == XK_c) {
                    colorOffset += 0.1f;
                    if (colorOffset > 1.0f) colorOffset -= 1.0f;
                    needsRedraw = 1;
                }
            }

            if (event.type == WIN32_EVENT_MOUSE_PRESS) {
                mouseX = event.mouseX;
                mouseY = event.mouseY;

                // Convert mouse position to complex coordinates
                double clickX = centerX + (mouseX - WIDTH / 2.0) / zoom;
                double clickY = centerY + (mouseY - HEIGHT / 2.0) / zoom;

                if (event.button == Button1) {
                    // Left click - zoom in
                    zoom *= 2.0;
                    centerX = clickX;
                    centerY = clickY;
                    maxIter = (int)(maxIter * 1.1);
                    if (maxIter > 5000) maxIter = 5000;
                } else if (event.button == Button3) {
                    // Right click - zoom out
                    zoom /= 2.0;
                    if (zoom < 50.0) zoom = 50.0;
                    centerX = clickX;
                    centerY = clickY;
                }

                needsRedraw = 1;
                printf("Center: (%.10f, %.10f) Zoom: %.2e Iter: %d\n",
                       centerX, centerY, zoom, maxIter);
            }

            if (event.type == WIN32_EVENT_MOUSE_MOVE) {
                mouseX = event.mouseX;
                mouseY = event.mouseY;
            }
        }

        // Render if needed
        if (needsRedraw) {
            double startRender = win32_get_time(display);

            mandelbrotKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT,
                centerX, centerY, zoom, maxIter, colorOffset);

            cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

            double renderTime = win32_get_time(display) - startRender;

            win32_blit_pixels(display, h_pixels);

            needsRedraw = 0;
            frameCount++;

            printf("Rendered in %.1f ms\n", renderTime * 1000);
        }

        // Small delay to prevent busy-waiting
        win32_sleep_ms(10);

        // FPS counter
        double now = win32_get_time(display);
        if (now - lastFpsTime >= 5.0 && frameCount > 0) {
            printf("Avg render rate: %.1f frames in 5s\n", (float)frameCount);
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
