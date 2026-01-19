// Mathematical Function Visualizer - Shadertoy 2244 Port
// Plots many mathematical functions on an interactive grid
// For Jetson Nano

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#define WIDTH 800
#define HEIGHT 600

// ============== Vector Types ==============

struct vec2 {
    float x, y;
    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(float a) : x(a), y(a) {}
    __host__ __device__ vec2(float a, float b) : x(a), y(b) {}
};

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float a) : x(a), y(a), z(a) {}
    __host__ __device__ vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

struct vec4 {
    float x, y, z, w;
    __host__ __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ vec4(float a) : x(a), y(a), z(a), w(a) {}
    __host__ __device__ vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    __host__ __device__ vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

struct uvec2 {
    unsigned int x, y;
    __host__ __device__ uvec2() : x(0), y(0) {}
    __host__ __device__ uvec2(unsigned int a, unsigned int b) : x(a), y(b) {}
};

struct uvec3 {
    unsigned int x, y, z;
    __host__ __device__ uvec3() : x(0), y(0), z(0) {}
    __host__ __device__ uvec3(unsigned int a, unsigned int b, unsigned int c) : x(a), y(b), z(c) {}
};

// Vector ops
__device__ vec2 operator+(vec2 a, vec2 b) { return vec2(a.x+b.x, a.y+b.y); }
__device__ vec2 operator-(vec2 a, vec2 b) { return vec2(a.x-b.x, a.y-b.y); }
__device__ vec2 operator*(vec2 a, float b) { return vec2(a.x*b, a.y*b); }
__device__ vec2 operator*(float a, vec2 b) { return vec2(a*b.x, a*b.y); }
__device__ vec2 operator*(vec2 a, vec2 b) { return vec2(a.x*b.x, a.y*b.y); }
__device__ vec2 operator/(vec2 a, float b) { return vec2(a.x/b, a.y/b); }
__device__ vec2 operator/(vec2 a, vec2 b) { return vec2(a.x/b.x, a.y/b.y); }

__device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ vec3 operator*(vec3 a, float b) { return vec3(a.x*b, a.y*b, a.z*b); }
__device__ vec3 operator*(float a, vec3 b) { return vec3(a*b.x, a*b.y, a*b.z); }
__device__ vec3 operator*(vec3 a, vec3 b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }

__device__ vec4 operator+(vec4 a, vec4 b) { return vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__device__ vec4 operator*(vec4 a, vec4 b) { return vec4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }

__device__ float dot(vec2 a, vec2 b) { return a.x*b.x + a.y*b.y; }
__device__ float length(vec2 v) { return sqrtf(dot(v,v)); }
__device__ vec2 normalize(vec2 v) { float l = length(v); return l > 0.0001f ? v*(1.0f/l) : vec2(0); }

// ============== Hash Functions (IQ) ==============

#define M1 1597334677u
#define M2 3812015801u
#define M3 3299493293u
#define F0 4294967295.0f

__device__ unsigned int hash(unsigned int n) { return n*(n^(n>>15)); }
__device__ float hash_uf(unsigned int n) { return float(hash(n))/F0; }

__device__ unsigned int hash_uu(unsigned int p) { return p*M1; }
__device__ unsigned int hash_u3u(uvec3 p) { return p.x*M1 ^ p.y*M2 ^ p.z*M3; }

// Per-thread random state
struct RandState {
    unsigned int seed;
    __device__ void init(uvec3 s) { seed = hash_u3u(s); }
    __device__ float rand() { return hash_uf(hash_uu(seed++)); }
};

// ============== Constants ==============

#define PI 3.14159265359f
#define PI2 (PI*2.0f)

// ============== Helper Functions ==============

__device__ float to01(float a) { return a*0.5f + 0.5f; }
__device__ float to11(float a) { return a*2.0f - 1.0f; }
__device__ vec2 to11(vec2 a) { return vec2(a.x*2.0f-1.0f, a.y*2.0f-1.0f); }
__device__ float clamp01(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

// ============== Mathematical Functions ==============

__device__ float fx(float x) { return x; }
__device__ float f1x(float x) { return -x; }
__device__ float f3(float x, float t) { return 1.0f/(1.0f+exp2f(-x*16.0f*sinf(t*PI2/4.0f))); }
__device__ float f5(float x) { return 1.0f/(exp2f(-x/0.05f)+exp2f(x/1.0f)); }
__device__ float f6(float x, float t) { return 1.0f/(exp2f(-x*exp2f(6.0f*sinf(t*PI2/3.0f)))+exp2f(x*exp2f(6.0f*sinf(t*PI2/4.0f)))); }
__device__ float f6log(float x, float t) { return log2f(fmaxf(f6(x,t), 1e-10f)); }

__device__ float fpdf(float x) { return expf(-0.5f*x*x)/sqrtf(PI2); }
__device__ float fpdflog(float x) { return log2f(fmaxf(fpdf(x), 1e-10f)); }

__device__ float f8(float x) { return exp2f(-powf(x, 3.0f)); }
__device__ float f8log(float x) { return log2f(fmaxf(f8(x), 1e-10f)); }

__device__ float f9(float x) { return -powf(fabsf(x), 2.0f); }
__device__ float f9exp(float x) { return exp2f(f9(x)); }

__device__ float f7xinv(float x) { return fabsf(x) > 0.01f ? 1.0f/-x : 0.0f; }
__device__ float fexp(float x) { return exp2f(x); }
__device__ float flog(float x) { return x > 0.001f ? log2f(x) : -10.0f; }
__device__ float fpow(float x, float t) { return powf(fabsf(x)+0.001f, exp2f(4.0f*sinf(t*PI2/5.0f))); }
__device__ float fpowlog(float x, float t) { return log2f(fmaxf(fpow(x,t), 1e-10f)); }
__device__ float f11sin(float x, float t) { x*=0.5f; return sinf(x+t*PI2*4.0f)*0.5f-1.5f; }
__device__ float f12sin(float x, float t) { return sinf(x*x+t*PI2*4.0f)*0.5f-0.5f; }

__device__ float sigmoid(float x, float w) { return 1.0f/(1.0f+exp2f(-x*w)); }
__device__ float fsigm(float x) { return sigmoid(x, 6.0f); }

__device__ float wave_cos(float x) { return to01(-cosf(x*PI)); }

__device__ float f13(float x) {
    float f = 0.95f;
    float b1 = exp2f(-6.0f);
    float b2 = exp2f(-3.0f);
    float a = (1.0f-exp2f(-x/b1))*(1.0f-exp2f(-(1.0f-x)/b2))*f;
    a *= 1.0f - powf(fmaxf(x, 0.0f), 4.1f);
    return a;
}

// ============== Grid and Line Functions ==============

__device__ float dist_line(vec2 p, vec2 a, vec2 n) {
    p = p - a;
    return length(p - dot(p,n)*n);
}

__device__ float act(vec2 p, float f, vec2 n, vec2 r2sc) {
    float d = dist_line(p, vec2(p.x, f), normalize(n));
    d *= r2sc.x;
    return clamp01(1.5f - d/1.0f);
}

__device__ float grid(vec2 u, float sc, vec2 r2sc) {
    u = u / sc;
    return clamp01(1.5f - fabsf(u.x - roundf(u.x))*r2sc.x*sc) +
           clamp01(1.5f - fabsf(u.y - roundf(u.y))*r2sc.y*sc);
}

// ============== Main Kernel ==============

__global__ void renderKernel(unsigned char* pixels, int frame, float iTime, 
                              vec2 sc, vec2 ctr) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= WIDTH || py >= HEIGHT) return;
    
    vec2 p = vec2((float)px, (float)(HEIGHT - 1 - py));
    vec2 r = vec2((float)WIDTH, (float)HEIGHT);
    
    // Random init
    RandState rnd;
    rnd.init(uvec3(frame % 3600, (unsigned int)p.x, (unsigned int)p.y));
    float iTimeDelta_adj = rnd.rand() * 0.016f;
    float t = iTime + iTimeDelta_adj;
    
    vec2 asp = vec2(r.x/r.y, 1.0f);
    vec2 u = to11(p / r);
    u = u * asp;
    u = u * sc;
    u = u + ctr;
    vec2 r2sc = r / sc / asp;
    
    float x = u.x;
    float x1 = ceilf(u.x*r2sc.x + 1.01f) / r2sc.x;
    float dx = x1 - x;
    
    vec3 c = vec3(0);
    
    // Grid
    c = c + vec3(1) * grid(u, 10.0f, r2sc) * exp2f(-4.0f);
    c = c + vec3(1) * grid(u, 1.0f, r2sc) * exp2f(-6.0f);
    c = c + vec3(1) * grid(u, 0.1f, r2sc) * exp2f(-8.0f);
    
    // Axis lines
    #define ADD_FUNC(func, col) { \
        float fval = func; \
        float fval1 = func; \
        c = c + col * act(u, fval, vec2(dx, fval1 - fval + 0.001f), r2sc); \
    }
    
    // Identity and negative
    ADD_FUNC(fx(x), vec3(1)*exp2f(-6.0f));
    ADD_FUNC(f1x(x), vec3(1)*exp2f(-6.0f));
    
    // Various functions
    ADD_FUNC(f3(x, t), vec3(1.0f, 0.0f, 0.0f));
    ADD_FUNC(f5(x), vec3(0.0f, 1.0f, 0.0f));
    ADD_FUNC(fsigm(x), vec3(1.0f, 0.25f, 0.0f));
    ADD_FUNC(f6(x, t), vec3(1.0f, 0.0f, 0.0f));
    ADD_FUNC(f6log(x, t), vec3(1.0f, 0.0f, 0.0f)*0.25f);
    ADD_FUNC(f7xinv(x), vec3(1)*0.25f);
    ADD_FUNC(fexp(x), vec3(0.0f, 0.5f, 1.0f));
    ADD_FUNC(flog(x), vec3(0.0f, 0.5f, 1.0f)*0.5f);
    ADD_FUNC(fpow(x, t), vec3(1.0f, 1.0f, 0.0f));
    ADD_FUNC(fpowlog(x, t), vec3(1.0f, 1.0f, 0.0f)*0.25f);
    ADD_FUNC(f11sin(x, t), vec3(1.0f, 0.0f, 0.0f));
    ADD_FUNC(f12sin(x, t), vec3(0.0f, 1.0f, 0.0f));
    ADD_FUNC(wave_cos(x), vec3(0.0f, 1.0f, 0.0f));
    ADD_FUNC(f13(x), vec3(1.0f, 0.0f, 0.0f));
    
    // Gamma correction
    float gamma = 1.0f/2.25f;
    c.x = powf(fmaxf(c.x, 0.0f), gamma);
    c.y = powf(fmaxf(c.y, 0.0f), gamma);
    c.z = powf(fmaxf(c.z, 0.0f), gamma);
    
    // Clamp
    c.x = fminf(c.x, 1.0f);
    c.y = fminf(c.y, 1.0f);
    c.z = fminf(c.z, 1.0f);
    
    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = (unsigned char)(c.z * 255.0f);
    pixels[idx + 1] = (unsigned char)(c.y * 255.0f);
    pixels[idx + 2] = (unsigned char)(c.x * 255.0f);
    pixels[idx + 3] = 255;
}

// ============== Main ==============

int main() {
    printf("=== Mathematical Function Visualizer ===\n");
    printf("Shadertoy 2244 Port - CUDA\n");
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);
    
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    
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
    
    XStoreName(display, window, "Math Functions - CUDA");
    XMapWindow(display, window);
    
    GC gc = XCreateGC(display, window, 0, NULL);
    XImage* image = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                 (char*)malloc(WIDTH * HEIGHT * 4), WIDTH, HEIGHT, 32, 0);
    
    printf("Controls:\n");
    printf("  Arrow keys - Pan view\n");
    printf("  +/-        - Zoom in/out\n");
    printf("  R          - Reset view\n");
    printf("  Q/ESC      - Quit\n\n");
    
    vec2 sc = vec2(1.5f, 1.5f);
    vec2 ctr = vec2(0.0f, 0.0f);
    int frame = 0;
    int running = 1;
    
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    while (running) {
        while (XPending(display)) {
            XEvent event;
            XNextEvent(display, &event);
            
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_q || key == XK_Q || key == XK_Escape) {
                    running = 0;
                } else if (key == XK_Left) {
                    ctr.x -= sc.x * 0.1f;
                } else if (key == XK_Right) {
                    ctr.x += sc.x * 0.1f;
                } else if (key == XK_Up) {
                    ctr.y += sc.y * 0.1f;
                } else if (key == XK_Down) {
                    ctr.y -= sc.y * 0.1f;
                } else if (key == XK_plus || key == XK_equal) {
                    sc.x *= 0.9f;
                    sc.y *= 0.9f;
                } else if (key == XK_minus) {
                    sc.x *= 1.1f;
                    sc.y *= 1.1f;
                } else if (key == XK_r || key == XK_R) {
                    sc = vec2(1.5f, 1.5f);
                    ctr = vec2(0.0f, 0.0f);
                    printf("View reset\n");
                }
            }
        }
        
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        float iTime = (now.tv_sec - start.tv_sec) + (now.tv_nsec - start.tv_nsec) / 1e9f;
        
        renderKernel<<<gridSize, blockSize>>>(d_pixels, frame, iTime, sc, ctr);
        cudaDeviceSynchronize();
        
        cudaMemcpy(image->data, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(display);
        
        frame++;
        usleep(16666);
    }
    
    cudaFree(d_pixels);
    XDestroyImage(image);
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    
    printf("Goodbye!\n");
    return 0;
}
