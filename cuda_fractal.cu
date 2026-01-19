/*
 * CUDA Fractal Zoom - Shadertoy Port
 * Original by Kishimisu - https://www.shadertoy.com/view/mtyGWy
 * Palette technique by Inigo Quilez
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

__device__ vec2 operator+(vec2 a, vec2 b) { return vec2(a.x+b.x, a.y+b.y); }
__device__ vec2 operator-(vec2 a, vec2 b) { return vec2(a.x-b.x, a.y-b.y); }
__device__ vec2 operator*(vec2 a, float t) { return vec2(a.x*t, a.y*t); }
__device__ vec2 operator*(float t, vec2 a) { return vec2(a.x*t, a.y*t); }

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ vec3 operator*(vec3 a, float t) { return vec3(a.x*t, a.y*t, a.z*t); }
__device__ vec3 operator*(float t, vec3 a) { return vec3(a.x*t, a.y*t, a.z*t); }

__device__ float length(vec2 v) { return sqrtf(v.x*v.x + v.y*v.y); }

__device__ float fract(float x) { return x - floorf(x); }
__device__ vec2 fract2(vec2 v) { return vec2(fract(v.x), fract(v.y)); }

// ============================================================================
// IQ PALETTE - https://iquilezles.org/articles/palettes/
// ============================================================================

__device__ vec3 palette(float t) {
    vec3 a = vec3(0.5f, 0.5f, 0.5f);
    vec3 b = vec3(0.5f, 0.5f, 0.5f);
    vec3 c = vec3(1.0f, 1.0f, 1.0f);
    vec3 d = vec3(0.263f, 0.416f, 0.557f);
    
    float angle = 6.28318f * (c.x * t + d.x);
    float angle_y = 6.28318f * (c.y * t + d.y);
    float angle_z = 6.28318f * (c.z * t + d.z);
    
    return vec3(a.x + b.x * cosf(angle),
                a.y + b.y * cosf(angle_y),
                a.z + b.z * cosf(angle_z));
}

// ============================================================================
// RENDER KERNEL
// ============================================================================

__global__ void renderKernel(unsigned char* pixels, int width, int height, float iTime) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= width || py >= height) return;
    
    // Normalized coordinates centered at origin
    float aspect = (float)width / (float)height;
    vec2 uv = vec2(
        (2.0f * px / width - 1.0f) * aspect,
        2.0f * py / height - 1.0f
    );
    vec2 uv0 = uv;
    
    vec3 finalColor = vec3(0.0f);
    
    // Fractal iteration
    for (float i = 0.0f; i < 4.0f; i += 1.0f) {
        // Tile and center
        uv = fract2(uv * 1.5f) - vec2(0.5f);
        
        // Distance with exponential falloff
        float d = length(uv) * expf(-length(uv0));
        
        // Color from palette based on position and time
        vec3 col = palette(length(uv0) + i * 0.4f + iTime * 0.4f);
        
        // Sinusoidal ring pattern
        d = sinf(d * 8.0f + iTime) / 8.0f;
        d = fabsf(d);
        
        // Glow effect
        d = powf(0.01f / d, 1.2f);
        
        // Accumulate color
        finalColor = finalColor + col * d;
    }
    
    // Clamp and output
    int idx = ((height - 1 - py) * width + px) * 4;
    pixels[idx + 0] = (unsigned char)(fminf(finalColor.z, 1.0f) * 255);
    pixels[idx + 1] = (unsigned char)(fminf(finalColor.y, 1.0f) * 255);
    pixels[idx + 2] = (unsigned char)(fminf(finalColor.x, 1.0f) * 255);
    pixels[idx + 3] = 255;
}

// ============================================================================
// MAIN
// ============================================================================

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    printf("=== CUDA Fractal Zoom ===\n");
    printf("Original shader by Kishimisu\n");
    printf("Palette by Inigo Quilez\n\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);
    
    // X11 setup
    Display* display = XOpenDisplay(NULL);
    if (!display) { printf("Cannot open display\n"); return 1; }
    
    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);
    
    XVisualInfo vinfo;
    XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo);
    
    XSetWindowAttributes attrs;
    attrs.colormap = XCreateColormap(display, root, vinfo.visual, AllocNone);
    attrs.event_mask = ExposureMask | KeyPressMask;
    
    Window window = XCreateWindow(display, root, 0, 0, WIDTH, HEIGHT, 0,
                                   vinfo.depth, InputOutput, vinfo.visual,
                                   CWColormap | CWEventMask, &attrs);
    
    XStoreName(display, window, "CUDA Fractal Zoom");
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
            snprintf(title, sizeof(title), "CUDA Fractal Zoom | %.1f FPS | t=%.1fs",
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
