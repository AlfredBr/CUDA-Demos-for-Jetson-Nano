/*
 * CUDA Flame Effect - Shadertoy Port (Windows)
 * Original by anatole duprat - XT95/2013
 * Ported to CUDA for Windows
 *
 * Volumetric flame using raymarching with procedural noise
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
    __device__ vec4(float v) : x(v), y(v), z(v), w(v) {}
    __device__ vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ vec3 operator*(vec3 a, vec3 b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ vec3 operator*(vec3 a, float t) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator*(float t, vec3 a) { return vec3(a.x*t, a.y*t, a.z*t); }

__device__ vec4 operator+(vec4 a, vec4 b) { return vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__device__ vec4 operator-(vec4 a, vec4 b) { return vec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__device__ vec4 operator*(vec4 a, float t) { return vec4(a.x*t, a.y*t, a.z*t, a.w*t); }
__device__ vec4 operator*(vec4 a, vec4 b) { return vec4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }

__device__ float dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float length(vec3 v) { return sqrtf(dot(v, v)); }
__device__ vec3 normalize(vec3 v) { float l = length(v); return l > 0 ? v * (1.0f/l) : vec3(0); }

__device__ vec3 floor3(vec3 v) { return vec3(floorf(v.x), floorf(v.y), floorf(v.z)); }
__device__ vec3 cos3(vec3 v) { return vec3(cosf(v.x), cosf(v.y), cosf(v.z)); }

__device__ vec4 mix(vec4 a, vec4 b, float t) {
    return vec4(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t, a.w + (b.w-a.w)*t);
}

__device__ float mix(float a, float b, float t) { return a + (b-a)*t; }

__device__ vec4 sin4(vec4 v) { return vec4(sinf(v.x), sinf(v.y), sinf(v.z), sinf(v.w)); }
__device__ vec4 cos4(vec4 v) { return vec4(cosf(v.x), cosf(v.y), cosf(v.z), cosf(v.w)); }

__device__ vec4 mix4(vec4 a, vec4 b, float t) {
    return vec4(a.x + (b.x-a.x)*t, a.y + (b.y-a.y)*t, a.z + (b.z-a.z)*t, a.w + (b.w-a.w)*t);
}

// ============================================================================
// NOISE FUNCTION (Las^Mercury style)
// ============================================================================

__device__ float noise(vec3 p) {
    vec3 i = floor3(p);
    float base = dot(i, vec3(1.0f, 57.0f, 21.0f));
    vec4 a = vec4(base, base + 57.0f, base + 21.0f, base + 78.0f);

    vec3 f = cos3((p - i) * 3.14159265f) * (-0.5f) + vec3(0.5f);

    // sin(cos(a)*a) and sin(cos(1+a)*(1+a))
    vec4 ca = cos4(a);
    vec4 sa = sin4(ca * a);

    vec4 a1 = a + vec4(1.0f);
    vec4 ca1 = cos4(a1);
    vec4 sb = sin4(ca1 * a1);

    vec4 m = mix4(sa, sb, f.x);

    // mix xy with zw based on f.y
    float mx = mix(m.x, m.z, f.y);
    float my = mix(m.y, m.w, f.y);

    return mix(mx, my, f.z);
}

// ============================================================================
// FLAME SDF AND SCENE
// ============================================================================

__device__ float sphere(vec3 p, vec4 spr) {
    return length(vec3(spr.x - p.x, spr.y - p.y, spr.z - p.z)) - spr.w;
}

__device__ float flame(vec3 p, float iTime) {
    vec3 ps = p * vec3(1.0f, 0.5f, 1.0f);
    float d = sphere(ps, vec4(0.0f, -1.0f, 0.0f, 1.0f));

    vec3 p1 = p + vec3(0.0f, iTime * 2.0f, 0.0f);
    vec3 p2 = p * 3.0f;

    float n = noise(p1) + noise(p2) * 0.5f;

    return d + n * 0.25f * p.y;
}

__device__ float scene(vec3 p, float iTime) {
    float bound = 100.0f - length(p);
    float f = fabsf(flame(p, iTime));
    return fminf(bound, f);
}

// ============================================================================
// RAYMARCHING
// ============================================================================

__device__ vec4 raymarch(vec3 org, vec3 dir, float iTime) {
    float d = 0.0f, glow = 0.0f, eps = 0.02f;
    vec3 p = org;
    bool glowed = false;

    for (int i = 0; i < 64; i++) {
        d = scene(p, iTime) + eps;
        p = p + dir * d;

        if (d > eps) {
            if (flame(p, iTime) < 0.0f)
                glowed = true;
            if (glowed)
                glow = (float)i / 64.0f;
        }
    }

    return vec4(p, glow);
}

// ============================================================================
// RENDER KERNEL
// ============================================================================

__global__ void renderKernel(unsigned char* pixels, int width, int height, float iTime) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Normalized coordinates [-1, 1]
    float u = -1.0f + 2.0f * (float)x / (float)width;
    float v = -1.0f + 2.0f * (float)y / (float)height;

    // Aspect ratio correction
    u *= (float)width / (float)height;

    // Camera
    vec3 org = vec3(0.0f, -2.0f, 4.0f);
    vec3 dir = normalize(vec3(u * 1.6f, -v, -1.5f));

    // Raymarch
    vec4 p = raymarch(org, dir, iTime);
    float glow = p.w;

    // Color gradient based on height
    vec4 col1 = vec4(1.0f, 0.5f, 0.1f, 1.0f);  // Orange/yellow
    vec4 col2 = vec4(0.1f, 0.5f, 1.0f, 1.0f);  // Blue
    vec4 col = mix(col1, col2, p.y * 0.02f + 0.4f);

    // Apply glow
    float intensity = powf(glow * 2.0f, 4.0f);
    vec4 finalCol = mix(vec4(0.0f), col, intensity);

    // Output
    int idx = ((height - 1 - y) * width + x) * 4;
    pixels[idx + 0] = (unsigned char)(fminf(finalCol.z, 1.0f) * 255);
    pixels[idx + 1] = (unsigned char)(fminf(finalCol.y, 1.0f) * 255);
    pixels[idx + 2] = (unsigned char)(fminf(finalCol.x, 1.0f) * 255);
    pixels[idx + 3] = 255;
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("=== CUDA Flame Effect (Windows) ===\n");
    printf("Original shader by anatole duprat - XT95/2013\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);

    // Win32 setup
    Win32Display* display = win32_create_window("CUDA Flame Effect", WIDTH, HEIGHT);
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

    while (1) {
        if (win32_process_events(display)) break;

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;
                if (key == XK_Escape || key == XK_q) goto cleanup;
            }
            else if (event.type == WIN32_EVENT_CLOSE) {
                goto cleanup;
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
            snprintf(title, sizeof(title), "CUDA Flame Effect | %.1f FPS | t=%.1fs",
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
