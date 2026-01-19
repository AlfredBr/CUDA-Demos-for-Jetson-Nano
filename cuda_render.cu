/*
 * Jetson Nano Native CUDA Graphics Demo
 *
 * This demonstrates hardware-accelerated rendering using CUDA on the Jetson Nano.
 * The GPU renders the pixels directly, then we display via X11.
 *
 * This bypasses OpenGL entirely and uses CUDA cores for rendering.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <sys/time.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 600

// CUDA kernel to render a colorful animated pattern
__global__ void renderKernel(unsigned char* pixels, int width, int height, float time) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;  // BGRA format for X11

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

    // Write BGRA (X11 format)
    pixels[idx + 0] = (unsigned char)(b * 255);  // Blue
    pixels[idx + 1] = (unsigned char)(g * 255);  // Green
    pixels[idx + 2] = (unsigned char)(r * 255);  // Red
    pixels[idx + 3] = 255;                        // Alpha
}

// Get current time in seconds
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("=== Jetson Nano Native CUDA Graphics Demo ===\n");
    printf("Resolution: %dx%d\n", WIDTH, HEIGHT);
    printf("Press Ctrl+C to exit\n\n");

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
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);  // Approximate for Tegra
    printf("Memory: %.0f MB\n\n", prop.totalGlobalMem / (1024.0 * 1024.0));

    // Open X11 display
    Display* display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Cannot open X display\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);

    // Create window
    XSetWindowAttributes attrs;
    attrs.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;

    Window window = XCreateWindow(display, root,
        100, 100, WIDTH, HEIGHT, 0,
        CopyFromParent, InputOutput, CopyFromParent,
        CWEventMask, &attrs);

    XStoreName(display, window, "Jetson CUDA Native Rendering");
    XMapWindow(display, window);

    // Wait for window to be mapped
    XEvent event;
    while (1) {
        XNextEvent(display, &event);
        if (event.type == MapNotify) break;
    }

    // Create X11 image for display
    Visual* visual = DefaultVisual(display, screen);
    int depth = DefaultDepth(display, screen);

    // Allocate host memory for pixels (pinned for faster transfer)
    unsigned char* h_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);

    // Allocate device memory
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    // Create XImage
    XImage* image = XCreateImage(display, visual, depth, ZPixmap, 0,
        (char*)h_pixels, WIDTH, HEIGHT, 32, WIDTH * 4);

    // Create GC for drawing
    GC gc = XCreateGC(display, window, 0, NULL);

    // CUDA grid configuration
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    printf("Rendering with CUDA...\n");
    printf("Block size: %dx%d, Grid size: %dx%d\n\n",
           blockSize.x, blockSize.y, gridSize.x, gridSize.y);

    double startTime = getTime();
    int frameCount = 0;
    double lastFpsTime = startTime;

    // Main render loop
    while (1) {
        // Check for X11 events
        while (XPending(display)) {
            XNextEvent(display, &event);
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_Escape || key == XK_q) {
                    goto cleanup;
                }
            }
            if (event.type == DestroyNotify) {
                goto cleanup;
            }
        }

        float time = (float)(getTime() - startTime);

        // Render on GPU
        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT, time);

        // Copy result to host (this is the bottleneck, but still uses DMA)
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

        // Display
        XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(display);

        frameCount++;

        // FPS counter
        double now = getTime();
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f (GPU-rendered frames)\n", frameCount / (now - lastFpsTime));
            frameCount = 0;
            lastFpsTime = now;
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    // Cleanup
    XFreeGC(display, gc);
    image->data = NULL;  // Prevent XDestroyImage from freeing our CUDA memory
    XDestroyImage(image);
    XDestroyWindow(display, window);
    XCloseDisplay(display);

    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
