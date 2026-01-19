// Conway's Game of Life - CUDA Demo for Jetson Nano
// Interactive cellular automaton with zoom/pan, painting, rule editor
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

// Display size
#define WIDTH 800
#define HEIGHT 600

// Grid size (can be larger than display with zoom/pan)
#define GRID_W 1024
#define GRID_H 1024

// Simulation state
struct SimState {
    // Rule: birthRule[n] = 1 if dead cell with n neighbors becomes alive
    //       surviveRule[n] = 1 if alive cell with n neighbors survives
    int birthRule[9];    // B3 = birth on 3 neighbors
    int surviveRule[9];  // S23 = survive on 2 or 3 neighbors

    float zoom;          // Pixels per cell
    float panX, panY;    // Pan offset in cells
    int paused;
    int speed;           // Ticks per frame (1-10)
    int wrapEdges;       // 1 = toroidal, 0 = bounded
    int brushSize;       // 1-5
    int generation;
    int population;
};

// Device constants for rules
__constant__ int d_birthRule[9];
__constant__ int d_surviveRule[9];
__constant__ int d_wrapEdges;

// Count neighbors with wrap or bounded edges
__device__ int countNeighbors(const unsigned char* grid, int x, int y) {
    int count = 0;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (d_wrapEdges) {
                // Toroidal wrap
                nx = (nx + GRID_W) % GRID_W;
                ny = (ny + GRID_H) % GRID_H;
            } else {
                // Bounded - out of bounds = dead
                if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H)
                    continue;
            }

            count += grid[ny * GRID_W + nx];
        }
    }
    return count;
}

// Simulation kernel - one thread per cell
__global__ void simulateKernel(const unsigned char* __restrict__ front,
                                unsigned char* __restrict__ back) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_W || y >= GRID_H) return;

    int idx = y * GRID_W + x;
    int neighbors = countNeighbors(front, x, y);
    int alive = front[idx];

    // Apply rules
    if (alive) {
        back[idx] = d_surviveRule[neighbors];
    } else {
        back[idx] = d_birthRule[neighbors];
    }
}

// Render kernel - convert grid to RGBA with zoom/pan
__global__ void renderKernel(const unsigned char* __restrict__ grid,
                             unsigned char* __restrict__ pixels,
                             float zoom, float panX, float panY) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= WIDTH || py >= HEIGHT) return;

    // Map pixel to grid cell
    float cellX = panX + px / zoom;
    float cellY = panY + py / zoom;

    int gx = (int)floorf(cellX);
    int gy = (int)floorf(cellY);

    // Background color (dark blue grid pattern)
    unsigned char r = 10, g = 15, b = 25;

    // Check if in bounds
    if (gx >= 0 && gx < GRID_W && gy >= 0 && gy < GRID_H) {
        int alive = grid[gy * GRID_W + gx];

        if (alive) {
            // Alive cells: bright green with slight variation
            float brightness = 0.8f + 0.2f * sinf(gx * 0.1f + gy * 0.1f);
            r = (unsigned char)(50 * brightness);
            g = (unsigned char)(255 * brightness);
            b = (unsigned char)(100 * brightness);
        } else {
            // Dead cells: subtle grid pattern when zoomed in
            if (zoom > 3.0f) {
                float fx = cellX - gx;
                float fy = cellY - gy;
                // Grid lines
                if (fx < 0.05f || fx > 0.95f || fy < 0.05f || fy > 0.95f) {
                    r = 20; g = 25; b = 35;
                }
            }
        }
    }

    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = b;  // BGRA format for X11
    pixels[idx + 1] = g;
    pixels[idx + 2] = r;
    pixels[idx + 3] = 255;
}

// Simple 4x6 font for UI
__device__ unsigned char getPixelFromChar(char c, int px, int py) {
    // Simplified bitmap patterns for key characters
    static const unsigned char font_0[] = {0x7C,0x82,0x82,0x82,0x82,0x7C};
    static const unsigned char font_1[] = {0x08,0x18,0x08,0x08,0x08,0x3E};
    static const unsigned char font_2[] = {0x7C,0x02,0x7C,0x80,0x80,0xFE};
    static const unsigned char font_3[] = {0x7C,0x02,0x3C,0x02,0x02,0x7C};
    static const unsigned char font_4[] = {0x82,0x82,0x7E,0x02,0x02,0x02};
    static const unsigned char font_5[] = {0xFE,0x80,0xFC,0x02,0x02,0xFC};
    static const unsigned char font_6[] = {0x7C,0x80,0xFC,0x82,0x82,0x7C};
    static const unsigned char font_7[] = {0xFE,0x02,0x04,0x08,0x10,0x10};
    static const unsigned char font_8[] = {0x7C,0x82,0x7C,0x82,0x82,0x7C};
    static const unsigned char font_9[] = {0x7C,0x82,0x7E,0x02,0x02,0x7C};

    const unsigned char* pattern = NULL;

    if (c >= '0' && c <= '9') {
        switch(c) {
            case '0': pattern = font_0; break;
            case '1': pattern = font_1; break;
            case '2': pattern = font_2; break;
            case '3': pattern = font_3; break;
            case '4': pattern = font_4; break;
            case '5': pattern = font_5; break;
            case '6': pattern = font_6; break;
            case '7': pattern = font_7; break;
            case '8': pattern = font_8; break;
            case '9': pattern = font_9; break;
        }
    }

    if (pattern && py < 6 && px < 8) {
        return (pattern[py] >> (7 - px)) & 1;
    }

    // For letters, use simple block patterns
    if (c >= 'A' && c <= 'Z') {
        // Just fill most of cell for visibility
        return (px >= 1 && px <= 5 && py >= 1 && py <= 4) ? 1 : 0;
    }
    if (c >= 'a' && c <= 'z') {
        return (px >= 1 && px <= 5 && py >= 2 && py <= 5) ? 1 : 0;
    }
    if (c == ':' || c == '/' || c == '|' || c == '-' || c == '=' || c == '+') {
        return (px == 3 || py == 3) ? 1 : 0;
    }
    if (c == '[' || c == ']' || c == '(' || c == ')') {
        return (px <= 2 || px >= 5) && (py >= 0 && py <= 5) ? 1 : 0;
    }

    return 0;
}

// Draw text on screen
__global__ void drawTextKernel(unsigned char* pixels, const char* text, int textLen,
                               int startX, int startY, unsigned char r, unsigned char g, unsigned char b) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= textLen * 8 * 6) return;

    int charIdx = tid / (8 * 6);
    int pixelInChar = tid % (8 * 6);
    int px = pixelInChar % 8;
    int py = pixelInChar / 8;

    if (charIdx >= textLen) return;

    char c = text[charIdx];
    int screenX = startX + charIdx * 7 + px;
    int screenY = startY + py;

    if (screenX >= 0 && screenX < WIDTH && screenY >= 0 && screenY < HEIGHT) {
        if (getPixelFromChar(c, px, py)) {
            int idx = (screenY * WIDTH + screenX) * 4;
            pixels[idx + 0] = b;
            pixels[idx + 1] = g;
            pixels[idx + 2] = r;
            pixels[idx + 3] = 255;
        }
    }
}

// Fill rectangle (for UI background)
__global__ void fillRectKernel(unsigned char* pixels, int x, int y, int w, int h,
                               unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= w || py >= h) return;

    int sx = x + px;
    int sy = y + py;

    if (sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT) {
        int idx = (sy * WIDTH + sx) * 4;
        // Alpha blend
        float alpha = a / 255.0f;
        pixels[idx + 0] = (unsigned char)(b * alpha + pixels[idx + 0] * (1-alpha));
        pixels[idx + 1] = (unsigned char)(g * alpha + pixels[idx + 1] * (1-alpha));
        pixels[idx + 2] = (unsigned char)(r * alpha + pixels[idx + 2] * (1-alpha));
        pixels[idx + 3] = 255;
    }
}

// Paint cells kernel
__global__ void paintKernel(unsigned char* grid, int cx, int cy, int brushSize, int value) {
    int dx = blockIdx.x * blockDim.x + threadIdx.x - brushSize;
    int dy = blockIdx.y * blockDim.y + threadIdx.y - brushSize;

    int x = cx + dx;
    int y = cy + dy;

    if (x >= 0 && x < GRID_W && y >= 0 && y < GRID_H) {
        float dist = sqrtf((float)(dx*dx + dy*dy));
        if (dist <= brushSize) {
            grid[y * GRID_W + x] = value;
        }
    }
}

// Randomize kernel
__global__ void randomizeKernel(unsigned char* grid, unsigned int seed, float density) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_W || y >= GRID_H) return;

    // Simple hash-based random
    unsigned int h = seed + x * 374761393u + y * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    h = h ^ (h >> 16);

    float r = (h & 0xFFFF) / 65535.0f;
    grid[y * GRID_W + x] = (r < density) ? 1 : 0;
}

// Clear kernel
__global__ void clearKernel(unsigned char* grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= GRID_W || y >= GRID_H) return;
    grid[y * GRID_W + x] = 0;
}

// Count population kernel (reduction)
__global__ void countPopKernel(const unsigned char* grid, int* count) {
    __shared__ int sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = 0;
    while (i < GRID_W * GRID_H) {
        sdata[tid] += grid[i];
        i += blockDim.x * gridDim.x;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(count, sdata[0]);
}

// Parse rule string like "B3/S23"
void parseRule(const char* rule, SimState* state) {
    memset(state->birthRule, 0, sizeof(state->birthRule));
    memset(state->surviveRule, 0, sizeof(state->surviveRule));

    int* current = NULL;
    for (const char* p = rule; *p; p++) {
        if (*p == 'B' || *p == 'b') {
            current = state->birthRule;
        } else if (*p == 'S' || *p == 's') {
            current = state->surviveRule;
        } else if (*p >= '0' && *p <= '8' && current) {
            current[*p - '0'] = 1;
        }
    }
}

// Get rule string
void getRuleString(SimState* state, char* buf) {
    char* p = buf;
    *p++ = 'B';
    for (int i = 0; i <= 8; i++) {
        if (state->birthRule[i]) *p++ = '0' + i;
    }
    *p++ = '/';
    *p++ = 'S';
    for (int i = 0; i <= 8; i++) {
        if (state->surviveRule[i]) *p++ = '0' + i;
    }
    *p = '\0';
}

int main() {
    printf("=== Conway's Game of Life - CUDA Demo ===\n");
    printf("Grid: %dx%d cells\n", GRID_W, GRID_H);
    printf("\nControls:\n");
    printf("  SPACE     - Pause/Play\n");
    printf("  S         - Single step (when paused)\n");
    printf("  R         - Randomize\n");
    printf("  C         - Clear\n");
    printf("  W         - Toggle wrap edges\n");
    printf("  +/-       - Zoom in/out\n");
    printf("  Arrows    - Pan\n");
    printf("  1-5       - Brush size\n");
    printf("  [/]       - Speed down/up\n");
    printf("  Left click  - Paint alive\n");
    printf("  Right click - Paint dead\n");
    printf("  0-9       - Preset rules\n");
    printf("  H         - Show/hide help\n");
    printf("  Q/ESC     - Quit\n\n");

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
    attrs.event_mask = ExposureMask | KeyPressMask | ButtonPressMask |
                       ButtonReleaseMask | PointerMotionMask | ButtonMotionMask;

    Window window = XCreateWindow(display, root, 0, 0, WIDTH, HEIGHT, 0,
                                  vinfo.depth, InputOutput, vinfo.visual,
                                  CWColormap | CWBorderPixel | CWBackPixel | CWEventMask,
                                  &attrs);

    XStoreName(display, window, "Conway's Game of Life - CUDA");
    XMapWindow(display, window);

    GC gc = XCreateGC(display, window, 0, NULL);
    XImage* image = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                 (char*)malloc(WIDTH * HEIGHT * 4), WIDTH, HEIGHT, 32, 0);

    // Initialize simulation state
    SimState state;
    parseRule("B3/S23", &state);  // Classic Conway
    state.zoom = 4.0f;
    state.panX = GRID_W / 2 - WIDTH / (2 * state.zoom);
    state.panY = GRID_H / 2 - HEIGHT / (2 * state.zoom);
    state.paused = 1;
    state.speed = 1;
    state.wrapEdges = 1;
    state.brushSize = 2;
    state.generation = 0;
    state.population = 0;

    // Allocate GPU memory
    unsigned char *d_grid[2], *d_pixels;
    int *d_popCount;
    char *d_text;

    cudaMalloc(&d_grid[0], GRID_W * GRID_H);
    cudaMalloc(&d_grid[1], GRID_W * GRID_H);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_popCount, sizeof(int));
    cudaMalloc(&d_text, 256);

    // Clear grids
    cudaMemset(d_grid[0], 0, GRID_W * GRID_H);
    cudaMemset(d_grid[1], 0, GRID_W * GRID_H);

    // Copy rules to device
    cudaMemcpyToSymbol(d_birthRule, state.birthRule, sizeof(state.birthRule));
    cudaMemcpyToSymbol(d_surviveRule, state.surviveRule, sizeof(state.surviveRule));
    cudaMemcpyToSymbol(d_wrapEdges, &state.wrapEdges, sizeof(int));

    // Kernel launch configs
    dim3 gridBlock(16, 16);
    dim3 gridGrid((GRID_W + 15) / 16, (GRID_H + 15) / 16);
    dim3 pixelGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    int currentGrid = 0;
    int mouseDown = 0;
    int mouseButton = 0;
    int showHelp = 1;
    int frameCount = 0;

    // Preset rules
    const char* presetRules[] = {
        "B3/S23",     // 0: Conway's Life
        "B36/S23",    // 1: HighLife
        "B2/S",       // 2: Seeds
        "B3/S012345678", // 3: Life without Death
        "B1357/S1357",   // 4: Replicator
        "B368/S245",     // 5: Morley
        "B3678/S34678",  // 6: Day & Night
        "B35678/S5678",  // 7: Diamoeba
        "B4678/S35678",  // 8: Anneal
        "B3/S2345"       // 9: Long Life
    };
    const char* presetNames[] = {
        "Conway", "HighLife", "Seeds", "NoDelete", "Replicator",
        "Morley", "Day&Night", "Diamoeba", "Anneal", "LongLife"
    };
    int currentPreset = 0;

    // Start with some random cells
    srand(time(NULL));
    {
        dim3 rBlock(16, 16);
        dim3 rGrid((GRID_W + 15) / 16, (GRID_H + 15) / 16);
        randomizeKernel<<<rGrid, rBlock>>>(d_grid[0], rand(), 0.25f);
        cudaDeviceSynchronize();
    }

    // Main loop
    while (1) {
        // Handle events
        while (XPending(display)) {
            XEvent event;
            XNextEvent(display, &event);

            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);

                if (key == XK_q || key == XK_Q || key == XK_Escape) {
                    goto cleanup;
                }
                else if (key == XK_space) {
                    state.paused = !state.paused;
                }
                else if (key == XK_s && state.paused) {
                    // Single step
                    simulateKernel<<<gridGrid, gridBlock>>>(d_grid[currentGrid], d_grid[1-currentGrid]);
                    currentGrid = 1 - currentGrid;
                    state.generation++;
                }
                else if (key == XK_r) {
                    randomizeKernel<<<gridGrid, gridBlock>>>(d_grid[currentGrid], rand(), 0.25f);
                    state.generation = 0;
                }
                else if (key == XK_c) {
                    clearKernel<<<gridGrid, gridBlock>>>(d_grid[currentGrid]);
                    state.generation = 0;
                }
                else if (key == XK_w) {
                    state.wrapEdges = !state.wrapEdges;
                    cudaMemcpyToSymbol(d_wrapEdges, &state.wrapEdges, sizeof(int));
                }
                else if (key == XK_h) {
                    showHelp = !showHelp;
                }
                else if (key == XK_plus || key == XK_equal) {
                    state.zoom = fminf(state.zoom * 1.2f, 32.0f);
                }
                else if (key == XK_minus) {
                    state.zoom = fmaxf(state.zoom / 1.2f, 0.5f);
                }
                else if (key == XK_Left) {
                    state.panX -= 20.0f / state.zoom;
                }
                else if (key == XK_Right) {
                    state.panX += 20.0f / state.zoom;
                }
                else if (key == XK_Up) {
                    state.panY -= 20.0f / state.zoom;
                }
                else if (key == XK_Down) {
                    state.panY += 20.0f / state.zoom;
                }
                else if (key == XK_bracketleft) {
                    state.speed = (state.speed > 1) ? state.speed - 1 : 1;
                }
                else if (key == XK_bracketright) {
                    state.speed = (state.speed < 10) ? state.speed + 1 : 10;
                }
                else if (key >= XK_1 && key <= XK_5) {
                    state.brushSize = key - XK_1 + 1;
                }
                else if (key >= XK_0 && key <= XK_9) {
                    int preset = (key == XK_0) ? 0 : key - XK_0;
                    if (preset < 10) {
                        parseRule(presetRules[preset], &state);
                        cudaMemcpyToSymbol(d_birthRule, state.birthRule, sizeof(state.birthRule));
                        cudaMemcpyToSymbol(d_surviveRule, state.surviveRule, sizeof(state.surviveRule));
                        currentPreset = preset;
                    }
                }
            }
            else if (event.type == ButtonPress) {
                mouseDown = 1;
                mouseButton = event.xbutton.button;
            }
            else if (event.type == ButtonRelease) {
                mouseDown = 0;
            }
            else if ((event.type == MotionNotify || event.type == ButtonPress) && mouseDown) {
                int mx = (event.type == MotionNotify) ? event.xmotion.x : event.xbutton.x;
                int my = (event.type == MotionNotify) ? event.xmotion.y : event.xbutton.y;

                // Convert screen coords to grid coords
                int gx = (int)(state.panX + mx / state.zoom);
                int gy = (int)(state.panY + my / state.zoom);

                // Paint cells
                int brushDim = state.brushSize * 2 + 1;
                dim3 paintBlock(brushDim, brushDim);
                dim3 paintGrid(1, 1);
                int value = (mouseButton == 1) ? 1 : 0;  // Left=alive, Right=dead
                paintKernel<<<paintGrid, paintBlock>>>(d_grid[currentGrid], gx, gy, state.brushSize, value);
            }
        }

        // Simulation step (if not paused)
        if (!state.paused) {
            for (int i = 0; i < state.speed; i++) {
                simulateKernel<<<gridGrid, gridBlock>>>(d_grid[currentGrid], d_grid[1-currentGrid]);
                currentGrid = 1 - currentGrid;
                state.generation++;
            }
        }

        // Count population every 10 frames
        if (frameCount % 10 == 0) {
            cudaMemset(d_popCount, 0, sizeof(int));
            countPopKernel<<<256, 256>>>(d_grid[currentGrid], d_popCount);
            cudaMemcpy(&state.population, d_popCount, sizeof(int), cudaMemcpyDeviceToHost);
        }

        // Render grid
        renderKernel<<<pixelGrid, gridBlock>>>(d_grid[currentGrid], d_pixels,
                                                state.zoom, state.panX, state.panY);

        // Draw UI overlay
        if (showHelp) {
            // Semi-transparent background
            dim3 uiBlock(16, 16);
            dim3 uiGrid((200 + 15) / 16, (140 + 15) / 16);
            fillRectKernel<<<uiGrid, uiBlock>>>(d_pixels, 5, 5, 200, 140, 0, 0, 0, 180);

            // Draw status text
            char buf[64];

            sprintf(buf, "Gen: %d", state.generation);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 10, 255, 255, 255);

            sprintf(buf, "Pop: %d", state.population);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 25, 255, 255, 255);

            char ruleBuf[32];
            getRuleString(&state, ruleBuf);
            sprintf(buf, "Rule: %s", ruleBuf);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 40, 100, 255, 100);

            sprintf(buf, "(%s)", presetNames[currentPreset]);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 55, 100, 200, 255);

            sprintf(buf, "Zoom: %.1fx", state.zoom);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 70, 200, 200, 200);

            sprintf(buf, "Speed: %d", state.speed);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 85, 200, 200, 200);

            sprintf(buf, "Brush: %d", state.brushSize);
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 100, 200, 200, 200);

            sprintf(buf, "Wrap: %s", state.wrapEdges ? "ON" : "OFF");
            cudaMemcpy(d_text, buf, strlen(buf) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(buf) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(buf), 10, 115, 200, 200, 200);

            const char* statusText = state.paused ? "[PAUSED]" : "[RUNNING]";
            unsigned char sr = state.paused ? 255 : 100;
            unsigned char sg = state.paused ? 100 : 255;
            cudaMemcpy(d_text, statusText, strlen(statusText) + 1, cudaMemcpyHostToDevice);
            drawTextKernel<<<(strlen(statusText) * 48 + 255) / 256, 256>>>(d_pixels, d_text, strlen(statusText), 10, 130, sr, sg, 100);
        }

        // Copy to host and display
        cudaMemcpy(image->data, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(display);

        frameCount++;
        usleep(16666);  // ~60 FPS
    }

cleanup:
    cudaFree(d_grid[0]);
    cudaFree(d_grid[1]);
    cudaFree(d_pixels);
    cudaFree(d_popCount);
    cudaFree(d_text);

    XDestroyImage(image);
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);

    printf("Goodbye!\n");
    return 0;
}
