/*
 * CUDA Fractal Pyramid - Shadertoy Port
 * Iterative folding fractal with volumetric rendering
 * Ported to CUDA for Windows
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "win32_display.h"

#define WIDTH 640
#define HEIGHT 480

// ============================================================================
// VECTOR MATH
// ============================================================================

struct vec2 {
    float x, y;
    __device__ vec2() : x(0), y(0) {}
    __device__ vec2(float v) : x(v), y(v) {}
    __device__ vec2(float x_, float y_) : x(x_), y(y_) {}
};

struct vec3 {
    float x, y, z;
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float v) : x(v), y(v), z(v) {}
    __device__ vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

struct vec4 {
    float x, y, z, w;
    __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __device__ vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

__device__ vec2 operator+(vec2 a, vec2 b) { return vec2(a.x+b.x, a.y+b.y); }
__device__ vec2 operator-(vec2 a, vec2 b) { return vec2(a.x-b.x, a.y-b.y); }
__device__ vec2 operator*(vec2 a, float t) { return vec2(a.x*t, a.y*t); }

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ vec3 operator*(vec3 a, float t) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator*(float t, vec3 a) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator/(vec3 a, float t) { return vec3(a.x/t, a.y/t, a.z/t); }

__device__ vec4 operator+(vec4 a, vec4 b) { return vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }

__device__ float dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float length(vec3 v) { return sqrtf(dot(v, v)); }
__device__ vec3 normalize(vec3 v) { float l = length(v); return l > 0 ? v * (1.0f/l) : vec3(0); }

__device__ vec3 cross(vec3 a, vec3 b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ vec3 mix(vec3 a, vec3 b, float t) {
    return vec3(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t);
}

__device__ float sign(float x) { return x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f); }
__device__ vec3 sign3(vec3 v) { return vec3(sign(v.x), sign(v.y), sign(v.z)); }

// ============================================================================
// PALETTE AND ROTATION
// ============================================================================

__device__ vec3 palette(float d) {
    return mix(vec3(0.2f, 0.7f, 0.9f), vec3(1.0f, 0.0f, 1.0f), d);
}

__device__ vec2 rotate(vec2 p, float a) {
    float c = cosf(a);
    float s = sinf(a);
    return vec2(p.x*c + p.y*s, -p.x*s + p.y*c);
}

// ============================================================================
// FRACTAL MAP - Iterative folding
// ============================================================================

__device__ float map(vec3 p, float iTime) {
    for (int i = 0; i < 8; ++i) {
        float t = iTime * 0.2f;

        // Rotate xz
        vec2 xz = rotate(vec2(p.x, p.z), t);
        p.x = xz.x;
        p.z = xz.y;

        // Rotate xy
        vec2 xy = rotate(vec2(p.x, p.y), t * 1.89f);
        p.x = xy.x;
        p.y = xy.y;

        // Fold - abs and subtract
        p.x = fabsf(p.x);
        p.z = fabsf(p.z);
        p.x -= 0.5f;
        p.z -= 0.5f;
    }
    return dot(sign3(p), p) / 5.0f;
}

// ============================================================================
// RAYMARCHING WITH VOLUMETRIC ACCUMULATION
// ============================================================================

__device__ vec4 rm(vec3 ro, vec3 rd, float iTime) {
    float t = 0.0f;
    vec3 col = vec3(0.0f);
    float d;

    for (float i = 0.0f; i < 64.0f; i += 1.0f) {
        vec3 p = ro + rd * t;
        d = map(p, iTime) * 0.5f;

        if (d < 0.02f) break;
        if (d > 100.0f) break;

        // Volumetric color accumulation
        vec3 palCol = palette(length(p) * 0.1f);
        col = col + palCol / (400.0f * d);

        t += d;
    }

    return vec4(col.x, col.y, col.z, 1.0f / (d * 100.0f));
}

// ============================================================================
// RENDER KERNEL
// ============================================================================

__global__ void renderKernel(unsigned char* pixels, int width, int height, float iTime) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // UV coordinates centered
    vec2 uv = vec2(
        ((float)px - width * 0.5f) / (float)width,
        ((float)py - height * 0.5f) / (float)width  // Use width for aspect
    );

    // Camera setup - orbiting
    vec3 ro = vec3(0.0f, 0.0f, -50.0f);
    vec2 ro_xz = rotate(vec2(ro.x, ro.z), iTime);
    ro.x = ro_xz.x;
    ro.z = ro_xz.y;

    // Camera frame
    vec3 cf = normalize(vec3(0.0f) - ro);  // Look at origin
    vec3 cs = normalize(cross(cf, vec3(0.0f, 1.0f, 0.0f)));
    vec3 cu = normalize(cross(cf, cs));

    // Ray direction
    vec3 uuv = ro + cf * 3.0f + cs * uv.x + cu * uv.y;
    vec3 rd = normalize(uuv - ro);

    // Raymarch
    vec4 col = rm(ro, rd, iTime);

    // Output with tone mapping
    int idx = ((height - 1 - py) * width + px) * 4;
    pixels[idx + 0] = (unsigned char)(fminf(col.z, 1.0f) * 255);
    pixels[idx + 1] = (unsigned char)(fminf(col.y, 1.0f) * 255);
    pixels[idx + 2] = (unsigned char)(fminf(col.x, 1.0f) * 255);
    pixels[idx + 3] = 255;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("=== CUDA Fractal Pyramid ===\n");
    printf("Iterative folding fractal with volumetric rendering\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);

    // Win32 setup
    Win32Display* display = win32_create_window("CUDA Fractal Pyramid", WIDTH, HEIGHT);
    if (!display) { printf("Cannot create window\n"); return 1; }

    // Allocate
    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    double startTime = win32_get_time(display);
    double lastFpsTime = startTime;
    int frameCount = 0;

    printf("Press Q or Escape to quit\n\n");

    while (!win32_should_close(display)) {
        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                if (event.key == XK_Escape || event.key == XK_q) goto cleanup;
            }
        }

        float iTime = (float)(win32_get_time(display) - startTime);

        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT, iTime);

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        double currentTime = win32_get_time(display);
        if (currentTime - lastFpsTime >= 1.0) {
            char title[128];
            snprintf(title, sizeof(title), "CUDA Fractal Pyramid | %.1f FPS | t=%.1fs",
                     frameCount / (currentTime - lastFpsTime), iTime);
            SetWindowTextA(display->hwnd, title);
            frameCount = 0;
            lastFpsTime = currentTime;
        }
    }

cleanup:
    cudaFree(d_pixels);
    free(h_pixels);
    win32_destroy_window(display);

    printf("Done!\n");
    return 0;
}
