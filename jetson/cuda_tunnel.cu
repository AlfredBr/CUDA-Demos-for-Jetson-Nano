/*
 * CUDA Volumetric Tunnel - Shadertoy Port
 * Original by Frostbyte - Licensed under CC BY-NC-SA 4.0
 * Only 10 raymarch steps! Low-step volumetric magic.
 * Ported to CUDA for Jetson Nano
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>

#define WIDTH 640
#define HEIGHT 480

// Vector types
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
    __device__ vec3(vec2 v, float z_) : x(v.x), y(v.y), z(z_) {}
};

// Vector operations
__device__ vec2 operator+(vec2 a, vec2 b) { return vec2(a.x+b.x, a.y+b.y); }
__device__ vec2 operator-(vec2 a, vec2 b) { return vec2(a.x-b.x, a.y-b.y); }
__device__ vec2 operator*(vec2 a, float t) { return vec2(a.x*t, a.y*t); }
__device__ vec2 sin2(vec2 v) { return vec2(sinf(v.x), sinf(v.y)); }

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ vec3 operator*(vec3 a, float t) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator*(float t, vec3 a) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator*(vec3 a, vec3 b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ vec3 operator/(vec3 a, float t) { return vec3(a.x/t, a.y/t, a.z/t); }
__device__ vec3 operator/(vec3 a, vec3 b) { return vec3(a.x/b.x, a.y/b.y, a.z/b.z); }

__device__ float length2(vec2 v) { return sqrtf(v.x*v.x + v.y*v.y); }
__device__ float length3(vec3 v) { return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z); }
__device__ vec3 normalize3(vec3 v) { float l = length3(v); return l > 0 ? v / l : vec3(0); }

__device__ float dot3(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }

__device__ vec3 sin3(vec3 v) { return vec3(sinf(v.x), sinf(v.y), sinf(v.z)); }
__device__ vec3 cos3(vec3 v) { return vec3(cosf(v.x), cosf(v.y), cosf(v.z)); }

// 2D rotation
__device__ vec2 rot(vec2 v, float t) {
    float s = sinf(t), c = cosf(t);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

// ACES tonemap
__device__ vec3 aces(vec3 c) {
    // Simplified ACES
    float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
    vec3 x = c;
    return vec3(
        (x.x * (a * x.x + b)) / (x.x * (cc * x.x + d) + e),
        (x.y * (a * x.y + b)) / (x.y * (cc * x.y + d) + e),
        (x.z * (a * x.z + b)) / (x.z * (cc * x.z + d) + e)
    );
}

// Xor's Dot Noise
__device__ float dotNoise(vec3 p) {
    const float PHI = 1.618033988f;

    // GOLD matrix * p
    vec3 gp = vec3(
        -0.571464913f * p.x + 0.814921382f * p.y + 0.096597072f * p.z,
        -0.278044873f * p.x - 0.303026659f * p.y + 0.911518454f * p.z,
         0.772087367f * p.x + 0.494042493f * p.y + 0.399753815f * p.z
    );

    // PHI * p * GOLD (for sin term)
    vec3 pp = vec3(
        PHI * (-0.571464913f * p.x + 0.814921382f * p.y + 0.096597072f * p.z),
        PHI * (-0.278044873f * p.x - 0.303026659f * p.y + 0.911518454f * p.z),
        PHI * ( 0.772087367f * p.x + 0.494042493f * p.y + 0.399753815f * p.z)
    );

    vec3 c = cos3(gp);
    vec3 s = sin3(pp);

    return c.x * s.x + c.y * s.y + c.z * s.z;
}

// Main render kernel
__global__ void renderKernel(unsigned char* pixels, int width, int height, float iTime) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // Coordinates
    vec2 u = vec2((float)px, (float)py);
    vec2 res = vec2((float)width, (float)height);

    float t = iTime;

    // Initialize
    vec3 p = vec3(0.0f, 0.0f, t);  // Start at time position
    vec3 d = normalize3(vec3(2.0f * u.x - res.x, 2.0f * u.y - res.y, res.y));
    vec3 l = vec3(0.0f);

    // Only 10 raymarch steps!
    for (float i = 0.0f; i < 10.0f; i += 1.0f) {
        vec3 b = p;

        // Rotate with turbulence
        vec2 bxy = vec2(b.x, b.y);
        bxy = rot(sin2(bxy), t * 1.5f + b.z * 3.0f);
        b.x = bxy.x;
        b.y = bxy.y;

        // Noise-based distance
        float s = 0.001f + fabsf(dotNoise(b * 12.0f) / 12.0f - dotNoise(b)) * 0.4f;

        // Tunnel clear zone
        s = fmaxf(s, 2.0f - length2(vec2(p.x, p.y)));

        // Turbulent waves
        s += fabsf(p.y * 0.75f + sinf(p.z + t * 0.1f + p.x * 1.5f)) * 0.2f;

        // March
        p = p + d * s;

        // Accumulate glow with color variation
        float lenPxy = length2(vec2(p.x, p.y)) * 0.1f;
        vec3 colorMod = vec3(
            1.0f + sinf(i + lenPxy + 3.0f),
            1.0f + sinf(i + lenPxy + 1.5f),
            1.0f + sinf(i + lenPxy + 1.0f)
        );
        l = l + colorMod / s;
    }

    // Tonemap
    vec3 col = aces(l * l / 600.0f);

    // Clamp and gamma
    col.x = fminf(1.0f, fmaxf(0.0f, col.x));
    col.y = fminf(1.0f, fmaxf(0.0f, col.y));
    col.z = fminf(1.0f, fmaxf(0.0f, col.z));

    // Output (BGR for X11)
    int idx = ((height - 1 - py) * width + px) * 4;
    pixels[idx + 0] = (unsigned char)(col.z * 255);
    pixels[idx + 1] = (unsigned char)(col.y * 255);
    pixels[idx + 2] = (unsigned char)(col.x * 255);
    pixels[idx + 3] = 255;
}

// Timer
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("=== CUDA Volumetric Tunnel ===\n");
    printf("Original shader by Frostbyte\n");
    printf("Only 10 raymarch steps!\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);

    // X11 setup
    Display* display = XOpenDisplay(NULL);
    if (!display) { printf("Cannot open display\n"); return 1; }

    int screen = DefaultScreen(display);
    XVisualInfo vinfo;
    XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo);

    XSetWindowAttributes attrs;
    attrs.colormap = XCreateColormap(display, RootWindow(display, screen), vinfo.visual, AllocNone);
    attrs.event_mask = ExposureMask | KeyPressMask;

    Window window = XCreateWindow(display, RootWindow(display, screen), 0, 0, WIDTH, HEIGHT, 0,
                                   vinfo.depth, InputOutput, vinfo.visual,
                                   CWColormap | CWEventMask, &attrs);

    XStoreName(display, window, "CUDA Volumetric Tunnel");
    XMapWindow(display, window);

    GC gc = XCreateGC(display, window, 0, NULL);

    XEvent event;
    while (1) { XNextEvent(display, &event); if (event.type == Expose) break; }

    // Allocate
    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);
    XImage* ximage = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                   (char*)h_pixels, WIDTH, HEIGHT, 32, 0);

    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    double startTime = getTime();
    double lastFpsTime = startTime;
    int frameCount = 0;

    printf("Press Q or Escape to quit\n\n");

    while (1) {
        while (XPending(display)) {
            XNextEvent(display, &event);
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_Escape || key == XK_q) goto cleanup;
            }
        }

        float iTime = (float)(getTime() - startTime);

        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT, iTime);

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, ximage, 0, 0, 0, 0, WIDTH, HEIGHT);

        frameCount++;
        double currentTime = getTime();
        if (currentTime - lastFpsTime >= 1.0) {
            char title[128];
            snprintf(title, sizeof(title), "CUDA Volumetric Tunnel | %.1f FPS | t=%.1fs",
                     frameCount / (currentTime - lastFpsTime), iTime);
            XStoreName(display, window, title);
            frameCount = 0;
            lastFpsTime = currentTime;
        }
    }

cleanup:
    cudaFree(d_pixels);
    XDestroyImage(ximage);
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);

    printf("Done!\n");
    return 0;
}
