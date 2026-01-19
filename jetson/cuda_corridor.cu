// Infinite Corridor - CUDA port for Jetson Nano
// Shadertoy-style infinite tunnel with rotating geometry
// Uses X11 + CUDA (no OpenGL)

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#define WIDTH 640
#define HEIGHT 480

// Vector types
struct vec2 { 
    float x, y; 
    __device__ vec2() : x(0), y(0) {}
    __device__ vec2(float a) : x(a), y(a) {}
    __device__ vec2(float a, float b) : x(a), y(b) {}
};

struct vec3 { 
    float x, y, z; 
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float a) : x(a), y(a), z(a) {}
    __device__ vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

struct vec4 { 
    float x, y, z, w; 
    __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __device__ vec4(float a) : x(a), y(a), z(a), w(a) {}
    __device__ vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    __device__ vec4(vec3 v, float a) : x(v.x), y(v.y), z(v.z), w(a) {}
};

// Vector operations
__device__ vec2 operator+(vec2 a, vec2 b) { return vec2(a.x + b.x, a.y + b.y); }
__device__ vec2 operator-(vec2 a, vec2 b) { return vec2(a.x - b.x, a.y - b.y); }
__device__ vec2 operator-(vec2 a, float b) { return vec2(a.x - b, a.y - b); }
__device__ vec2 operator*(vec2 a, float b) { return vec2(a.x * b, a.y * b); }
__device__ vec2 operator*(float a, vec2 b) { return vec2(a * b.x, a * b.y); }

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
__device__ vec3 operator-(vec3 a, float b) { return vec3(a.x - b, a.y - b, a.z - b); }
__device__ vec3 operator*(vec3 a, float b) { return vec3(a.x * b, a.y * b, a.z * b); }
__device__ vec3 operator*(float a, vec3 b) { return vec3(a * b.x, a * b.y, a * b.z); }
__device__ vec3 operator*(vec3 a, vec3 b) { return vec3(a.x * b.x, a.y * b.y, a.z * b.z); }
__device__ vec3 operator+(vec3 a, float b) { return vec3(a.x + b, a.y + b, a.z + b); }
__device__ vec3 operator+(float a, vec3 b) { return vec3(a + b.x, a + b.y, a + b.z); }

__device__ vec4 operator+(vec4 a, vec4 b) { return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
__device__ vec4 operator*(vec4 a, float b) { return vec4(a.x * b, a.y * b, a.z * b, a.w * b); }

__device__ float dot(vec3 a, vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ float length(vec2 v) { return sqrtf(v.x * v.x + v.y * v.y); }
__device__ float length(vec3 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }

__device__ vec3 normalize(vec3 v) {
    float len = length(v);
    return len > 0.0f ? v * (1.0f / len) : vec3(0.0f);
}

__device__ vec3 abs3(vec3 v) { return vec3(fabsf(v.x), fabsf(v.y), fabsf(v.z)); }
__device__ vec3 sqrt3(vec3 v) { return vec3(sqrtf(v.x), sqrtf(v.y), sqrtf(v.z)); }

__device__ float fract(float x) { return x - floorf(x); }

__device__ float mod(float x, float y) { return x - y * floorf(x / y); }

__device__ vec2 mod2(vec2 v, float y) { 
    return vec2(mod(v.x, y), mod(v.y, y)); 
}

// 2D rotation
__device__ vec2 rot(vec2 p, float a) {
    float c = cosf(a);
    float s = sinf(a);
    return vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Distance function - the infinite corridor geometry
__device__ float map(vec3 p) {
    vec3 n = vec3(0.0f, 1.0f, 0.0f);
    float k1 = 1.9f;
    float k2 = (sinf(p.x * k1) + sinf(p.z * k1)) * 0.8f;
    float k3 = (sinf(p.y * k1) + sinf(p.z * k1)) * 0.8f;
    
    // Wall planes with sine distortion
    float w1 = 4.0f - dot(abs3(p), normalize(n)) + k2;
    float w2 = 4.0f - dot(abs3(p), normalize(vec3(n.y, n.z, n.x))) + k3;
    
    // Repeating spheres/tubes
    vec2 mp1 = mod2(vec2(p.x, p.y) + vec2(sinf((p.z + p.x) * 2.0f) * 0.3f, 
                                           cosf((p.z + p.x) * 1.0f) * 0.5f), 2.0f) - 1.0f;
    float s1 = length(mp1) - 0.2f;
    
    vec2 mp2 = mod2(vec2(0.5f + p.y, p.z) + vec2(sinf((p.z + p.x) * 2.0f) * 0.3f, 
                                                  cosf((p.z + p.x) * 1.0f) * 0.3f), 2.0f) - 1.0f;
    float s2 = length(mp2) - 0.2f;
    
    return fminf(w1, fminf(w2, fminf(s1, s2)));
}

// Main render kernel
__global__ void renderKernel(unsigned char* pixels, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    
    // Flip Y for correct orientation
    int flippedY = HEIGHT - 1 - py;
    
    // UV coordinates
    vec2 uv = vec2(
        (2.0f * px / (float)WIDTH - 1.0f) * ((float)WIDTH / (float)HEIGHT),
        2.0f * flippedY / (float)HEIGHT - 1.0f
    );
    
    // Ray direction
    vec3 dir = normalize(vec3(uv.x, uv.y, 1.0f));
    
    // Rotate view
    vec2 xz = rot(vec2(dir.x, dir.z), time * 0.23f);
    dir = vec3(xz.x, dir.y, xz.y);
    // Swizzle: dir = dir.yzx
    dir = vec3(dir.y, dir.z, dir.x);
    
    xz = rot(vec2(dir.x, dir.z), time * 0.2f);
    dir = vec3(xz.x, dir.y, xz.y);
    // Swizzle: dir = dir.yzx
    dir = vec3(dir.y, dir.z, dir.x);
    
    // Ray origin - moving through the corridor
    vec3 pos = vec3(0.0f, 0.0f, time);
    
    // Ray march
    float t = 0.0f;
    float tt = 0.0f;
    
    for (int i = 0; i < 100; i++) {
        vec3 rayPos = pos + dir * t;
        tt = map(rayPos);
        if (tt < 0.001f) break;
        t += tt * 0.45f;
    }
    
    // Hit point
    vec3 ip = pos + dir * t;
    
    // Color calculation
    vec3 col = vec3(t * 0.1f);
    col = sqrt3(col);
    
    // Final color with direction-based tint and glow
    vec3 absDir = abs3(dir);
    float edgeGlow = fmaxf(0.0f, map(ip - 0.1f) - tt);
    
    vec4 fragColor = vec4(
        0.05f * t + absDir.x * col.x + edgeGlow,
        0.05f * t + absDir.y * col.y + edgeGlow,
        0.05f * t + absDir.z * col.z + edgeGlow,
        1.0f
    );
    
    // Alpha-based fog/glow (used for blending effect)
    float alpha = 1.0f / (t * t * t * t + 0.0001f);
    alpha = fminf(alpha, 1.0f);
    
    // Enhance colors
    fragColor.x = fminf(fragColor.x * (0.5f + alpha * 0.5f), 1.0f);
    fragColor.y = fminf(fragColor.y * (0.5f + alpha * 0.5f), 1.0f);
    fragColor.z = fminf(fragColor.z * (0.5f + alpha * 0.5f), 1.0f);
    
    // Gamma and tone mapping
    fragColor.x = powf(fragColor.x, 0.8f);
    fragColor.y = powf(fragColor.y, 0.8f);
    fragColor.z = powf(fragColor.z, 0.8f);
    
    // Output
    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = (unsigned char)(fminf(fragColor.z * 255.0f, 255.0f));  // B
    pixels[idx + 1] = (unsigned char)(fminf(fragColor.y * 255.0f, 255.0f));  // G
    pixels[idx + 2] = (unsigned char)(fminf(fragColor.x * 255.0f, 255.0f));  // R
    pixels[idx + 3] = 255;
}

int main() {
    printf("=== Infinite Corridor - CUDA Demo ===\n");
    printf("Shadertoy-style infinite tunnel\n");
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);
    printf("Controls:\n");
    printf("  Q/ESC - Quit\n");
    printf("  SPACE - Pause\n\n");
    
    // X11 setup
    Display* display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Cannot open display\n");
        return 1;
    }
    
    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);
    
    XVisualInfo vinfo;
    XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo);
    
    XSetWindowAttributes attrs;
    attrs.colormap = XCreateColormap(display, root, vinfo.visual, AllocNone);
    attrs.border_pixel = 0;
    attrs.background_pixel = 0;
    attrs.event_mask = ExposureMask | KeyPressMask;
    
    Window window = XCreateWindow(display, root, 0, 0, WIDTH, HEIGHT, 0,
                                  vinfo.depth, InputOutput, vinfo.visual,
                                  CWColormap | CWBorderPixel | CWBackPixel | CWEventMask,
                                  &attrs);
    
    XStoreName(display, window, "Infinite Corridor - CUDA");
    XMapWindow(display, window);
    
    GC gc = XCreateGC(display, window, 0, NULL);
    XImage* image = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                 (char*)malloc(WIDTH * HEIGHT * 4), WIDTH, HEIGHT, 32, 0);
    
    // CUDA setup
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    float time = 0.0f;
    int paused = 0;
    int running = 1;
    
    printf("Running... Flying through infinite corridor!\n");
    
    while (running) {
        // Handle events
        while (XPending(display)) {
            XEvent event;
            XNextEvent(display, &event);
            
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_q || key == XK_Q || key == XK_Escape) {
                    running = 0;
                } else if (key == XK_space) {
                    paused = !paused;
                    printf("%s\n", paused ? "Paused" : "Running");
                }
            }
        }
        
        // Render
        renderKernel<<<gridDim, blockDim>>>(d_pixels, time);
        cudaDeviceSynchronize();
        
        // Copy to host and display
        cudaMemcpy(image->data, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(display);
        
        // Update time
        if (!paused) {
            time += 0.016f;
        }
        
        usleep(16666);  // ~60 FPS
    }
    
    // Cleanup
    cudaFree(d_pixels);
    XDestroyImage(image);
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    
    printf("Goodbye!\n");
    return 0;
}
