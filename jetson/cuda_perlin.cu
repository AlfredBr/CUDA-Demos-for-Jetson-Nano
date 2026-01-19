/*
 * CUDA Perlin Noise Explorer
 * Interactive demo with fBm, domain warping, and multiple color modes
 * Keyboard UI for all parameters
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>

#define WIDTH 640
#define HEIGHT 480

// Parameters struct
struct Params {
    float frequency;
    float lacunarity;
    float persistence;
    float warpStrength;
    float timeSpeed;
    int octaves;
    int colorMode;
    int noiseMode;
    int seed;
};

__constant__ unsigned char d_perm[512];
__constant__ Params d_params;

// Perlin noise functions
__device__ float fade(float t) { return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f); }
__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }

__device__ float grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

__device__ float perlin3D(float x, float y, float z) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    int Z = (int)floorf(z) & 255;
    x -= floorf(x); y -= floorf(y); z -= floorf(z);
    float u = fade(x), v = fade(y), w = fade(z);
    int A = d_perm[X] + Y, AA = d_perm[A] + Z, AB = d_perm[A + 1] + Z;
    int B = d_perm[X + 1] + Y, BA = d_perm[B] + Z, BB = d_perm[B + 1] + Z;
    return lerp(lerp(lerp(grad(d_perm[AA], x, y, z), grad(d_perm[BA], x-1, y, z), u),
                     lerp(grad(d_perm[AB], x, y-1, z), grad(d_perm[BB], x-1, y-1, z), u), v),
                lerp(lerp(grad(d_perm[AA+1], x, y, z-1), grad(d_perm[BA+1], x-1, y, z-1), u),
                     lerp(grad(d_perm[AB+1], x, y-1, z-1), grad(d_perm[BB+1], x-1, y-1, z-1), u), v), w);
}

__device__ float fbm(float x, float y, float z, int octaves, float lacunarity, float persistence) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, maxValue = 0.0f;
    for (int i = 0; i < octaves; i++) {
        value += amplitude * perlin3D(x * frequency, y * frequency, z * frequency);
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / maxValue;
}

__device__ float ridgedFbm(float x, float y, float z, int octaves, float lacunarity, float persistence) {
    float value = 0.0f, amplitude = 1.0f, frequency = 1.0f, maxValue = 0.0f;
    for (int i = 0; i < octaves; i++) {
        float n = perlin3D(x * frequency, y * frequency, z * frequency);
        n = 1.0f - fabsf(n); n = n * n;
        value += amplitude * n;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / maxValue;
}

// Color ramps
__device__ void grayscale(float n, unsigned char* r, unsigned char* g, unsigned char* b) {
    n = fmaxf(0.0f, fminf(1.0f, (n + 1.0f) * 0.5f));
    *r = *g = *b = (unsigned char)(n * 255);
}

__device__ void terrainRamp(float n, unsigned char* r, unsigned char* g, unsigned char* b) {
    n = fmaxf(0.0f, fminf(1.0f, (n + 1.0f) * 0.5f));
    if (n < 0.3f) { float t = n / 0.3f; *r = 20 + t * 40; *g = 50 + t * 80; *b = 120 + t * 60; }
    else if (n < 0.4f) { float t = (n - 0.3f) / 0.1f; *r = 60 + t * 140; *g = 130 + t * 70; *b = 180 - t * 100; }
    else if (n < 0.5f) { float t = (n - 0.4f) / 0.1f; *r = 200 - t * 140; *g = 200 - t * 50; *b = 80 - t * 50; }
    else if (n < 0.7f) { float t = (n - 0.5f) / 0.2f; *r = 60 + t * 60; *g = 150 - t * 50; *b = 30 + t * 40; }
    else if (n < 0.85f) { float t = (n - 0.7f) / 0.15f; *r = 120 + t * 40; *g = 100 + t * 40; *b = 70 + t * 50; }
    else { float t = (n - 0.85f) / 0.15f; *r = 160 + t * 95; *g = 140 + t * 115; *b = 120 + t * 135; }
}

__device__ void marbleRamp(float n, unsigned char* r, unsigned char* g, unsigned char* b) {
    float marble = (sinf((n * 30.0f) * 3.14159f) + 1.0f) * 0.5f;
    *r = 180 + marble * 75; *g = 170 + marble * 85; *b = 160 + marble * 95;
}

__device__ void plasmaRamp(float n, unsigned char* r, unsigned char* g, unsigned char* b) {
    n = (n + 1.0f) * 0.5f;
    *r = (unsigned char)(sinf(n * 6.28f) * 127 + 128);
    *g = (unsigned char)(sinf(n * 6.28f + 2.094f) * 127 + 128);
    *b = (unsigned char)(sinf(n * 6.28f + 4.188f) * 127 + 128);
}

__global__ void renderKernel(unsigned char* pixels, int width, int height, float time) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;
    
    float x = (float)px / width, y = (float)py / height;
    float t = time * d_params.timeSpeed;
    float n;
    
    if (d_params.noiseMode == 0) {
        n = fbm(x * d_params.frequency, y * d_params.frequency, t,
                d_params.octaves, d_params.lacunarity, d_params.persistence);
    } else if (d_params.noiseMode == 1) {
        float wx = fbm((x + 17.0f) * d_params.frequency, (y + 31.0f) * d_params.frequency, t,
                       d_params.octaves, d_params.lacunarity, d_params.persistence);
        float wy = fbm((x + 47.0f) * d_params.frequency, (y + 11.0f) * d_params.frequency, t,
                       d_params.octaves, d_params.lacunarity, d_params.persistence);
        n = fbm((x + wx * d_params.warpStrength) * d_params.frequency,
                (y + wy * d_params.warpStrength) * d_params.frequency, t,
                d_params.octaves, d_params.lacunarity, d_params.persistence);
    } else {
        n = ridgedFbm(x * d_params.frequency, y * d_params.frequency, t,
                      d_params.octaves, d_params.lacunarity, d_params.persistence) * 2.0f - 1.0f;
    }
    
    unsigned char r, g, b;
    switch (d_params.colorMode) {
        case 0: grayscale(n, &r, &g, &b); break;
        case 1: terrainRamp(n, &r, &g, &b); break;
        case 2: marbleRamp(n, &r, &g, &b); break;
        case 3: plasmaRamp(n, &r, &g, &b); break;
        default: grayscale(n, &r, &g, &b); break;
    }
    
    int idx = ((height - 1 - py) * width + px) * 4;
    pixels[idx] = b; pixels[idx + 1] = g; pixels[idx + 2] = r; pixels[idx + 3] = 255;
}

// Simple 4x6 font - just numbers and basic chars
void getGlyph(char c, unsigned char glyph[6]) {
    static const unsigned char font[][6] = {
        {0x6,0x9,0x9,0x9,0x9,0x6},   // 0
        {0x2,0x6,0x2,0x2,0x2,0x7},   // 1
        {0xF,0x1,0x6,0x8,0x8,0xF},   // 2
        {0xE,0x1,0x6,0x1,0x1,0xE},   // 3
        {0x9,0x9,0xF,0x1,0x1,0x1},   // 4
        {0xF,0x8,0xE,0x1,0x1,0xE},   // 5
        {0x6,0x8,0xE,0x9,0x9,0x6},   // 6
        {0xF,0x1,0x2,0x4,0x4,0x4},   // 7
        {0x6,0x9,0x6,0x9,0x9,0x6},   // 8
        {0x6,0x9,0x7,0x1,0x1,0x6},   // 9
        {0x0,0x0,0x0,0x0,0x6,0x6},   // .
        {0x0,0x0,0xF,0x0,0x0,0x0},   // -
        {0x0,0x4,0xE,0x4,0x0,0x0},   // +
        {0x0,0x6,0x0,0x0,0x6,0x0},   // :
        {0x0,0x0,0xF,0x0,0xF,0x0},   // =
        {0x1,0x2,0x4,0x8,0x0,0x0},   // /
        {0x6,0x9,0x8,0x8,0x9,0x6},   // C
        {0xE,0x9,0x9,0xE,0x8,0x8},   // P
        {0xF,0x8,0xE,0x8,0x8,0xF},   // E
        {0x6,0x9,0x8,0xB,0x9,0x6},   // G
        {0x8,0x8,0x8,0x8,0x8,0xF},   // L
        {0x9,0x9,0x9,0x9,0x9,0x6},   // U
        {0x9,0x9,0xF,0x9,0x9,0x9},   // H
        {0xE,0x4,0x4,0x4,0x4,0xE},   // I
        {0x9,0xD,0xB,0x9,0x9,0x9},   // N
        {0x6,0x9,0x9,0x9,0x9,0x6},   // O
        {0xE,0x9,0xE,0xC,0xA,0x9},   // R
        {0xF,0x4,0x4,0x4,0x4,0x4},   // T
        {0x0,0x0,0x6,0x1,0x7,0x5},   // a
        {0x8,0x8,0xE,0x9,0x9,0xE},   // b
        {0x0,0x0,0x7,0x8,0x8,0x7},   // c
        {0x1,0x1,0x7,0x9,0x9,0x7},   // d
        {0x0,0x6,0x9,0xF,0x8,0x7},   // e
        {0x2,0x4,0xE,0x4,0x4,0x4},   // f
        {0x0,0x7,0x9,0x7,0x1,0x6},   // g
        {0x8,0x8,0xE,0x9,0x9,0x9},   // h
        {0x4,0x0,0x4,0x4,0x4,0x4},   // i
        {0x0,0x8,0xC,0x8,0x8,0xA},   // k
        {0x6,0x2,0x2,0x2,0x2,0x7},   // l
        {0x0,0x0,0xA,0xF,0x9,0x9},   // m
        {0x0,0x0,0xE,0x9,0x9,0x9},   // n
        {0x0,0x0,0x6,0x9,0x9,0x6},   // o
        {0x0,0xE,0x9,0xE,0x8,0x8},   // p
        {0x0,0x0,0x7,0x9,0x7,0x1},   // q
        {0x0,0x0,0xB,0xC,0x8,0x8},   // r
        {0x0,0x7,0x8,0x6,0x1,0xE},   // s
        {0x4,0x4,0xE,0x4,0x4,0x3},   // t
        {0x0,0x0,0x9,0x9,0x9,0x7},   // u
        {0x0,0x0,0x9,0x9,0x6,0x6},   // v
        {0x0,0x0,0x9,0x9,0xF,0x6},   // w
        {0x0,0x0,0x9,0x6,0x6,0x9},   // x
        {0x0,0x9,0x9,0x7,0x1,0x6},   // y
        {0x0,0x0,0xF,0x2,0x4,0xF},   // z
        {0x6,0x9,0x9,0xF,0x9,0x9},   // A
        {0x9,0x9,0x9,0x9,0x6,0x6},   // V
        {0x9,0x9,0x6,0x4,0x4,0x4},   // Y
        {0x1,0x2,0x4,0x2,0x1,0x0},   // <
        {0x4,0x2,0x1,0x2,0x4,0x0},   // >
        {0x0,0x0,0x0,0x0,0x0,0x0},   // space
        {0x9,0x9,0xF,0x9,0x9,0x6},   // W (mapped)
        {0xE,0x9,0x9,0xE,0x9,0x9},   // B
        {0x1C,0x12,0x11,0x11,0x12,0x1C}, // D
        {0xF,0x10,0x0E,0x01,0x01,0x1E},  // S
        {0xF,0x9,0x9,0x9,0x9,0x9},   // M
        {0xF,0x4,0x4,0x4,0x4,0x4},   // F
        {0xE,0x8,0x8,0x8,0x8,0xE},   // [
        {0xE,0x2,0x2,0x2,0x2,0xE},   // ]
    };
    int idx = -1;
    if (c >= '0' && c <= '9') idx = c - '0';
    else if (c == '.') idx = 10;
    else if (c == '-') idx = 11;
    else if (c == '+') idx = 12;
    else if (c == ':') idx = 13;
    else if (c == '=') idx = 14;
    else if (c == '/') idx = 15;
    else if (c == 'C') idx = 16;
    else if (c == 'P') idx = 17;
    else if (c == 'E') idx = 18;
    else if (c == 'G') idx = 19;
    else if (c == 'L') idx = 20;
    else if (c == 'U') idx = 21;
    else if (c == 'H') idx = 22;
    else if (c == 'I') idx = 23;
    else if (c == 'N') idx = 24;
    else if (c == 'O') idx = 25;
    else if (c == 'R') idx = 26;
    else if (c == 'T') idx = 27;
    else if (c == 'a') idx = 28;
    else if (c == 'b') idx = 29;
    else if (c == 'c') idx = 30;
    else if (c == 'd') idx = 31;
    else if (c == 'e') idx = 32;
    else if (c == 'f') idx = 33;
    else if (c == 'g') idx = 34;
    else if (c == 'h') idx = 35;
    else if (c == 'i') idx = 36;
    else if (c == 'k') idx = 37;
    else if (c == 'l') idx = 38;
    else if (c == 'm') idx = 39;
    else if (c == 'n') idx = 40;
    else if (c == 'o') idx = 41;
    else if (c == 'p') idx = 42;
    else if (c == 'q') idx = 43;
    else if (c == 'r') idx = 44;
    else if (c == 's') idx = 45;
    else if (c == 't') idx = 46;
    else if (c == 'u') idx = 47;
    else if (c == 'v') idx = 48;
    else if (c == 'w') idx = 49;
    else if (c == 'x') idx = 50;
    else if (c == 'y') idx = 51;
    else if (c == 'z') idx = 52;
    else if (c == 'A') idx = 53;
    else if (c == 'V') idx = 54;
    else if (c == 'Y') idx = 55;
    else if (c == 'W') idx = 58;
    else if (c == 'B') idx = 59;
    else if (c == 'D') idx = 60;
    else if (c == 'S') idx = 61;
    else if (c == 'M') idx = 62;
    else if (c == 'F') idx = 63;
    else if (c == '[') idx = 64;
    else if (c == ']') idx = 65;
    else idx = 57;  // space
    
    if (idx >= 0 && idx < 66) memcpy(glyph, font[idx], 6);
    else memset(glyph, 0, 6);
}

void drawChar(unsigned char* pixels, int width, int height, int x, int y, char c, 
              unsigned char r, unsigned char g, unsigned char b) {
    unsigned char glyph[6];
    getGlyph(c, glyph);
    for (int row = 0; row < 6; row++) {
        for (int col = 0; col < 4; col++) {
            if (glyph[row] & (0x8 >> col)) {
                int px = x + col, py = y + row;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 4;
                    pixels[idx] = b; pixels[idx + 1] = g; pixels[idx + 2] = r;
                }
            }
        }
    }
}

void drawString(unsigned char* pixels, int width, int height, int x, int y, const char* str,
                unsigned char r, unsigned char g, unsigned char b) {
    while (*str) { drawChar(pixels, width, height, x, y, *str, r, g, b); x += 5; str++; }
}

void generatePermutation(unsigned char* perm, int seed) {
    for (int i = 0; i < 256; i++) perm[i] = i;
    srand(seed);
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        unsigned char tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    for (int i = 0; i < 256; i++) perm[256 + i] = perm[i];
}

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

const char* noiseModeNames[] = {"fBm", "Domain Warp", "Ridged"};
const char* colorModeNames[] = {"Grayscale", "Terrain", "Marble", "Plasma"};

int main() {
    printf("=== CUDA Perlin Noise Explorer ===\n\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\nResolution: %dx%d\n\n", prop.name, WIDTH, HEIGHT);
    
    Params params = {4.0f, 2.0f, 0.5f, 0.5f, 0.3f, 6, 1, 0, 42};
    unsigned char perm[512];
    generatePermutation(perm, params.seed);
    cudaMemcpyToSymbol(d_perm, perm, 512);
    cudaMemcpyToSymbol(d_params, &params, sizeof(Params));
    
    Display* display = XOpenDisplay(NULL);
    if (!display) { printf("Cannot open display\n"); return 1; }
    int screen = DefaultScreen(display);
    XVisualInfo vinfo;
    XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo);
    XSetWindowAttributes attrs;
    attrs.colormap = XCreateColormap(display, RootWindow(display, screen), vinfo.visual, AllocNone);
    attrs.event_mask = ExposureMask | KeyPressMask;
    Window window = XCreateWindow(display, RootWindow(display, screen), 0, 0, WIDTH, HEIGHT, 0,
                                   vinfo.depth, InputOutput, vinfo.visual, CWColormap | CWEventMask, &attrs);
    XStoreName(display, window, "CUDA Perlin Noise Explorer");
    XMapWindow(display, window);
    GC gc = XCreateGC(display, window, 0, NULL);
    XEvent event;
    while (1) { XNextEvent(display, &event); if (event.type == Expose) break; }
    
    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);
    XImage* ximage = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                   (char*)h_pixels, WIDTH, HEIGHT, 32, 0);
    unsigned char* d_pixels;
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    
    dim3 blockSize(16, 16), gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    double startTime = getTime(), lastFpsTime = startTime;
    int frameCount = 0;
    float fps = 0;
    bool showHelp = true, paused = false;
    float pausedTime = 0;
    
    printf("F/G=freq L/K=lacun P/O=persist W/S=warp T/R=speed +/-=octaves\n");
    printf("C=color N=noise Z=seed H=help Space=pause Q=quit\n\n");
    
    while (1) {
        while (XPending(display)) {
            XNextEvent(display, &event);
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                bool needsUpdate = false;
                switch (key) {
                    case XK_Escape: case XK_q: goto cleanup;
                    case XK_f: params.frequency *= 1.1f; needsUpdate = true; break;
                    case XK_g: params.frequency /= 1.1f; needsUpdate = true; break;
                    case XK_l: params.lacunarity += 0.1f; needsUpdate = true; break;
                    case XK_k: params.lacunarity = fmaxf(1.1f, params.lacunarity - 0.1f); needsUpdate = true; break;
                    case XK_p: params.persistence = fminf(1.0f, params.persistence + 0.05f); needsUpdate = true; break;
                    case XK_o: params.persistence = fmaxf(0.1f, params.persistence - 0.05f); needsUpdate = true; break;
                    case XK_w: params.warpStrength += 0.1f; needsUpdate = true; break;
                    case XK_s: params.warpStrength = fmaxf(0.0f, params.warpStrength - 0.1f); needsUpdate = true; break;
                    case XK_t: params.timeSpeed += 0.1f; needsUpdate = true; break;
                    case XK_r: params.timeSpeed = fmaxf(0.0f, params.timeSpeed - 0.1f); needsUpdate = true; break;
                    case XK_plus: case XK_equal: params.octaves = params.octaves < 10 ? params.octaves + 1 : 10; needsUpdate = true; break;
                    case XK_minus: params.octaves = params.octaves > 1 ? params.octaves - 1 : 1; needsUpdate = true; break;
                    case XK_c: params.colorMode = (params.colorMode + 1) % 4; needsUpdate = true; break;
                    case XK_n: params.noiseMode = (params.noiseMode + 1) % 3; needsUpdate = true; break;
                    case XK_z: params.seed = rand(); generatePermutation(perm, params.seed);
                               cudaMemcpyToSymbol(d_perm, perm, 512); needsUpdate = true; break;
                    case XK_h: showHelp = !showHelp; break;
                    case XK_space: paused = !paused;
                        if (paused) pausedTime = getTime() - startTime;
                        else startTime = getTime() - pausedTime; break;
                }
                if (needsUpdate) cudaMemcpyToSymbol(d_params, &params, sizeof(Params));
            }
        }
        
        float time = paused ? pausedTime : (float)(getTime() - startTime);
        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT, time);
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        
        if (showHelp) {
            char buf[64]; int y = 5;
            snprintf(buf, sizeof(buf), "Mode: %s", noiseModeNames[params.noiseMode]);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 255); y += 8;
            snprintf(buf, sizeof(buf), "Color: %s", colorModeNames[params.colorMode]);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 255); y += 8;
            snprintf(buf, sizeof(buf), "Freq: %.1f [F/G]", params.frequency);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 0); y += 8;
            snprintf(buf, sizeof(buf), "Octaves: %d [+/-]", params.octaves);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 0); y += 8;
            snprintf(buf, sizeof(buf), "Lacun: %.1f [L/K]", params.lacunarity);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 0); y += 8;
            snprintf(buf, sizeof(buf), "Persist: %.2f [P/O]", params.persistence);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 0); y += 8;
            if (params.noiseMode == 1) {
                snprintf(buf, sizeof(buf), "Warp: %.1f [W/S]", params.warpStrength);
                drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 0, 255, 255); y += 8;
            }
            snprintf(buf, sizeof(buf), "Speed: %.1f [T/R]", params.timeSpeed);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 255, 255, 0); y += 8;
            snprintf(buf, sizeof(buf), "%.1f FPS", fps);
            drawString(h_pixels, WIDTH, HEIGHT, 5, y, buf, 0, 255, 0); y += 8;
            drawString(h_pixels, WIDTH, HEIGHT, 5, HEIGHT - 10, "H=help C=color N=noise Q=quit", 150, 150, 150);
        }
        
        XPutImage(display, window, gc, ximage, 0, 0, 0, 0, WIDTH, HEIGHT);
        frameCount++;
        double currentTime = getTime();
        if (currentTime - lastFpsTime >= 0.5) {
            fps = frameCount / (currentTime - lastFpsTime);
            char title[128];
            snprintf(title, sizeof(title), "CUDA Perlin | %s | %s | %.1f FPS",
                     noiseModeNames[params.noiseMode], colorModeNames[params.colorMode], fps);
            XStoreName(display, window, title);
            frameCount = 0; lastFpsTime = currentTime;
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
