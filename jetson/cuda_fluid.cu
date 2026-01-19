/*
 * Jetson Nano CUDA 2D Fluid Simulation
 * Based on Jos Stam's "Stable Fluids" (SIGGRAPH 1999)
 *
 * Features:
 *   - Semi-Lagrangian advection
 *   - Jacobi iteration for diffusion & pressure solve
 *   - Divergence-free projection
 *   - Interactive mouse/keyboard input
 *   - Real-time density visualization
 *
 * Controls:
 *   Left Mouse  - Add density + velocity (drag to push fluid)
 *   Right Mouse - Add density only
 *   1-4         - Preset color schemes
 *   V           - Toggle velocity visualization
 *   C           - Clear simulation
 *   +/-         - Adjust viscosity
 *   [/]         - Adjust diffusion
 *   R           - Reset parameters
 *   Q/Escape    - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>
#include <math.h>

// Simulation grid size (power of 2 for efficiency)
#define SIM_WIDTH 256
#define SIM_HEIGHT 256

// Display size
#define DISP_WIDTH 768
#define DISP_HEIGHT 768

// Simulation parameters
#define JACOBI_ITERATIONS 20
#define VELOCITY_DISSIPATION 0.999f
#define DENSITY_DISSIPATION 0.995f

// Grid indexing macros
#define IX(x, y) ((y) * SIM_WIDTH + (x))
#define CLAMP(x, lo, hi) ((x) < (lo) ? (lo) : ((x) > (hi) ? (hi) : (x)))

// Bilinear interpolation
__device__ float bilerp(float* field, float x, float y) {
    x = CLAMP(x, 0.5f, SIM_WIDTH - 1.5f);
    y = CLAMP(y, 0.5f, SIM_HEIGHT - 1.5f);

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    float sx = x - x0;
    float sy = y - y0;

    float v00 = field[IX(x0, y0)];
    float v10 = field[IX(x1, y0)];
    float v01 = field[IX(x0, y1)];
    float v11 = field[IX(x1, y1)];

    return (1-sx)*(1-sy)*v00 + sx*(1-sy)*v10 + (1-sx)*sy*v01 + sx*sy*v11;
}

// ============== ADVECTION ==============
// Semi-Lagrangian: trace particle back in time, sample old value
__global__ void advectKernel(float* dst, float* src, float* velX, float* velY,
                             float dt, float dissipation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    // Trace back
    float px = x - dt * velX[IX(x, y)];
    float py = y - dt * velY[IX(x, y)];

    // Sample and dissipate
    dst[IX(x, y)] = dissipation * bilerp(src, px, py);
}

// Advect velocity field (self-advection)
__global__ void advectVelocityKernel(float* dstX, float* dstY,
                                      float* srcX, float* srcY,
                                      float dt, float dissipation) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    // Trace back using current velocity
    float px = x - dt * srcX[IX(x, y)];
    float py = y - dt * srcY[IX(x, y)];

    // Sample velocity at traced position
    dstX[IX(x, y)] = dissipation * bilerp(srcX, px, py);
    dstY[IX(x, y)] = dissipation * bilerp(srcY, px, py);
}

// ============== DIFFUSION ==============
// Jacobi iteration for diffusion: (I - α∇²)x = b
__global__ void diffuseJacobiKernel(float* dst, float* src, float* src0,
                                     float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    float left   = src[IX(x-1, y)];
    float right  = src[IX(x+1, y)];
    float bottom = src[IX(x, y-1)];
    float top    = src[IX(x, y+1)];
    float center = src0[IX(x, y)];

    dst[IX(x, y)] = (center + alpha * (left + right + bottom + top)) * beta;
}

// ============== PROJECTION ==============
// Step 1: Compute divergence of velocity field
__global__ void divergenceKernel(float* div, float* velX, float* velY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    float vL = velX[IX(x-1, y)];
    float vR = velX[IX(x+1, y)];
    float vB = velY[IX(x, y-1)];
    float vT = velY[IX(x, y+1)];

    div[IX(x, y)] = -0.5f * (vR - vL + vT - vB);
}

// Step 2: Jacobi iteration to solve pressure Poisson equation
__global__ void pressureJacobiKernel(float* dst, float* src, float* div) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    float pL = src[IX(x-1, y)];
    float pR = src[IX(x+1, y)];
    float pB = src[IX(x, y-1)];
    float pT = src[IX(x, y+1)];
    float d  = div[IX(x, y)];

    dst[IX(x, y)] = (d + pL + pR + pB + pT) * 0.25f;
}

// Step 3: Subtract pressure gradient from velocity
__global__ void gradientSubtractKernel(float* velX, float* velY, float* pressure) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= SIM_WIDTH-1 || y < 1 || y >= SIM_HEIGHT-1) return;

    float pL = pressure[IX(x-1, y)];
    float pR = pressure[IX(x+1, y)];
    float pB = pressure[IX(x, y-1)];
    float pT = pressure[IX(x, y+1)];

    velX[IX(x, y)] -= 0.5f * (pR - pL);
    velY[IX(x, y)] -= 0.5f * (pT - pB);
}

// ============== BOUNDARY CONDITIONS ==============
__global__ void setBoundaryKernel(float* field, int scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= SIM_WIDTH && i >= SIM_HEIGHT) return;

    // Left and right boundaries
    if (i < SIM_HEIGHT) {
        field[IX(0, i)] = scale * field[IX(1, i)];
        field[IX(SIM_WIDTH-1, i)] = scale * field[IX(SIM_WIDTH-2, i)];
    }

    // Top and bottom boundaries
    if (i < SIM_WIDTH) {
        field[IX(i, 0)] = scale * field[IX(i, 1)];
        field[IX(i, SIM_HEIGHT-1)] = scale * field[IX(i, SIM_HEIGHT-2)];
    }
}

// ============== SPLAT (USER INPUT) ==============
__global__ void splatKernel(float* field, int cx, int cy, float radius,
                            float amount, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= SIM_WIDTH || y >= SIM_HEIGHT) return;

    float dx = x - cx;
    float dy = y - cy;
    float dist2 = dx * dx + dy * dy;
    float r2 = radius * radius;

    if (dist2 < r2) {
        float factor = expf(-dist2 / (0.25f * r2)) * dt * amount;
        field[IX(x, y)] += factor;
    }
}

__global__ void splatVelocityKernel(float* velX, float* velY,
                                     int cx, int cy, float radius,
                                     float vx, float vy, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= SIM_WIDTH || y >= SIM_HEIGHT) return;

    float dx = x - cx;
    float dy = y - cy;
    float dist2 = dx * dx + dy * dy;
    float r2 = radius * radius;

    if (dist2 < r2) {
        float factor = expf(-dist2 / (0.25f * r2)) * dt;
        velX[IX(x, y)] += vx * factor;
        velY[IX(x, y)] += vy * factor;
    }
}

// ============== VISUALIZATION ==============
__global__ void renderKernel(unsigned char* pixels, float* density,
                             float* velX, float* velY,
                             int dispWidth, int dispHeight,
                             int colorScheme, int showVelocity) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= dispWidth || py >= dispHeight) return;

    // Map display pixel to simulation cell
    float sx = (float)px / dispWidth * SIM_WIDTH;
    float sy = (float)(dispHeight - 1 - py) / dispHeight * SIM_HEIGHT;  // Flip Y

    int x = (int)sx;
    int y = (int)sy;
    x = CLAMP(x, 0, SIM_WIDTH - 1);
    y = CLAMP(y, 0, SIM_HEIGHT - 1);

    float d = density[IX(x, y)];
    d = CLAMP(d, 0.0f, 1.0f);

    float r, g, b;

    // Different color schemes
    switch (colorScheme) {
        case 0:  // Fire/smoke
            r = d * 2.0f;
            g = d * d * 1.5f;
            b = d * d * d;
            break;
        case 1:  // Blue/cyan ink
            r = d * d * 0.3f;
            g = d * 0.8f;
            b = d * 1.2f + 0.1f * d * d;
            break;
        case 2:  // Green plasma
            r = d * d * 0.5f;
            g = d * 1.2f;
            b = d * d * d * 0.8f;
            break;
        case 3:  // Rainbow based on density
            {
                float h = d * 4.0f;  // Hue
                float s = 1.0f;
                float v = sqrtf(d);
                // HSV to RGB
                int hi = (int)h % 6;
                float f = h - (int)h;
                float p = v * (1 - s);
                float q = v * (1 - f * s);
                float t = v * (1 - (1 - f) * s);
                switch (hi) {
                    case 0: r = v; g = t; b = p; break;
                    case 1: r = q; g = v; b = p; break;
                    case 2: r = p; g = v; b = t; break;
                    case 3: r = p; g = q; b = v; break;
                    case 4: r = t; g = p; b = v; break;
                    default: r = v; g = p; b = q; break;
                }
            }
            break;
        default:
            r = g = b = d;
    }

    // Show velocity as color overlay
    if (showVelocity) {
        float vx = velX[IX(x, y)] * 0.05f;
        float vy = velY[IX(x, y)] * 0.05f;
        r += fabsf(vx);
        b += fabsf(vy);
    }

    // Background gradient
    float bg = 0.02f + 0.03f * ((float)py / dispHeight);
    r = fmaxf(r, bg);
    g = fmaxf(g, bg * 0.8f);
    b = fmaxf(b, bg * 1.2f);

    int idx = (py * dispWidth + px) * 4;
    pixels[idx + 0] = (unsigned char)(CLAMP(b, 0.0f, 1.0f) * 255);  // B
    pixels[idx + 1] = (unsigned char)(CLAMP(g, 0.0f, 1.0f) * 255);  // G
    pixels[idx + 2] = (unsigned char)(CLAMP(r, 0.0f, 1.0f) * 255);  // R
    pixels[idx + 3] = 255;
}

// ============== CLEAR ==============
__global__ void clearFieldKernel(float* field, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) field[i] = 0.0f;
}

// ============== HOST CODE ==============
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("=== Jetson Nano CUDA 2D Fluid Simulation ===\n");
    printf("Based on Jos Stam's \"Stable Fluids\"\n\n");
    printf("Controls:\n");
    printf("  Left Mouse  - Add density + velocity (drag to push)\n");
    printf("  Right Mouse - Add density only\n");
    printf("  1-4         - Color schemes\n");
    printf("  V           - Toggle velocity visualization\n");
    printf("  C           - Clear simulation\n");
    printf("  +/-         - Adjust viscosity\n");
    printf("  [/]         - Adjust diffusion rate\n");
    printf("  R           - Reset parameters\n");
    printf("  Q/Escape    - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Grid: %dx%d, Display: %dx%d\n\n", SIM_WIDTH, SIM_HEIGHT, DISP_WIDTH, DISP_HEIGHT);

    // Open X11
    Display* display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Cannot open X display\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);

    XSetWindowAttributes attrs;
    attrs.event_mask = ExposureMask | KeyPressMask | ButtonPressMask |
                       ButtonReleaseMask | PointerMotionMask | StructureNotifyMask;
    attrs.background_pixel = BlackPixel(display, screen);

    Window window = XCreateWindow(display, root,
        50, 50, DISP_WIDTH, DISP_HEIGHT, 0,
        CopyFromParent, InputOutput, CopyFromParent,
        CWEventMask | CWBackPixel, &attrs);

    XStoreName(display, window, "CUDA 2D Fluid Simulation - Stable Fluids");
    XMapWindow(display, window);

    XEvent event;
    while (1) {
        XNextEvent(display, &event);
        if (event.type == MapNotify) break;
    }

    // Allocate simulation fields
    int fieldSize = SIM_WIDTH * SIM_HEIGHT;
    size_t fieldBytes = fieldSize * sizeof(float);

    float *d_velX, *d_velY;           // Velocity field
    float *d_velX_prev, *d_velY_prev; // Previous velocity (for ping-pong)
    float *d_density, *d_density_prev; // Density field
    float *d_pressure, *d_pressure_prev; // Pressure field
    float *d_divergence;               // Divergence

    cudaMalloc(&d_velX, fieldBytes);
    cudaMalloc(&d_velY, fieldBytes);
    cudaMalloc(&d_velX_prev, fieldBytes);
    cudaMalloc(&d_velY_prev, fieldBytes);
    cudaMalloc(&d_density, fieldBytes);
    cudaMalloc(&d_density_prev, fieldBytes);
    cudaMalloc(&d_pressure, fieldBytes);
    cudaMalloc(&d_pressure_prev, fieldBytes);
    cudaMalloc(&d_divergence, fieldBytes);

    // Clear all fields
    cudaMemset(d_velX, 0, fieldBytes);
    cudaMemset(d_velY, 0, fieldBytes);
    cudaMemset(d_velX_prev, 0, fieldBytes);
    cudaMemset(d_velY_prev, 0, fieldBytes);
    cudaMemset(d_density, 0, fieldBytes);
    cudaMemset(d_density_prev, 0, fieldBytes);
    cudaMemset(d_pressure, 0, fieldBytes);
    cudaMemset(d_pressure_prev, 0, fieldBytes);
    cudaMemset(d_divergence, 0, fieldBytes);

    // Allocate display buffer
    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, DISP_WIDTH * DISP_HEIGHT * 4);
    cudaMalloc(&d_pixels, DISP_WIDTH * DISP_HEIGHT * 4);

    Visual* visual = DefaultVisual(display, screen);
    int depth = DefaultDepth(display, screen);
    XImage* image = XCreateImage(display, visual, depth, ZPixmap, 0,
        (char*)h_pixels, DISP_WIDTH, DISP_HEIGHT, 32, DISP_WIDTH * 4);

    GC gc = XCreateGC(display, window, 0, NULL);

    // Kernel configurations
    dim3 simBlock(16, 16);
    dim3 simGrid((SIM_WIDTH + 15) / 16, (SIM_HEIGHT + 15) / 16);
    dim3 dispBlock(16, 16);
    dim3 dispGrid((DISP_WIDTH + 15) / 16, (DISP_HEIGHT + 15) / 16);
    int boundaryThreads = max(SIM_WIDTH, SIM_HEIGHT);

    // Simulation parameters
    float viscosity = 0.0001f;
    float diffusion = 0.0001f;
    float dt = 0.1f;
    int colorScheme = 0;
    int showVelocity = 0;

    // Mouse state
    int mouseDown = 0;
    int mouseButton = 0;
    int lastMouseX = 0, lastMouseY = 0;

    double lastTime = getTime();
    double lastFpsTime = lastTime;
    int frameCount = 0;

    printf("Simulation running...\n");

    while (1) {
        // Handle events
        while (XPending(display)) {
            XNextEvent(display, &event);

            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);

                if (key == XK_Escape || key == XK_q) goto cleanup;
                if (key == XK_c) {
                    cudaMemset(d_density, 0, fieldBytes);
                    cudaMemset(d_velX, 0, fieldBytes);
                    cudaMemset(d_velY, 0, fieldBytes);
                    printf("Cleared!\n");
                }
                if (key == XK_v) {
                    showVelocity = !showVelocity;
                    printf("Velocity display: %s\n", showVelocity ? "ON" : "OFF");
                }
                if (key == XK_1) { colorScheme = 0; printf("Color: Fire\n"); }
                if (key == XK_2) { colorScheme = 1; printf("Color: Ink\n"); }
                if (key == XK_3) { colorScheme = 2; printf("Color: Plasma\n"); }
                if (key == XK_4) { colorScheme = 3; printf("Color: Rainbow\n"); }
                if (key == XK_plus || key == XK_equal) {
                    viscosity *= 2.0f;
                    printf("Viscosity: %.6f\n", viscosity);
                }
                if (key == XK_minus) {
                    viscosity *= 0.5f;
                    printf("Viscosity: %.6f\n", viscosity);
                }
                if (key == XK_bracketright) {
                    diffusion *= 2.0f;
                    printf("Diffusion: %.6f\n", diffusion);
                }
                if (key == XK_bracketleft) {
                    diffusion *= 0.5f;
                    printf("Diffusion: %.6f\n", diffusion);
                }
                if (key == XK_r) {
                    viscosity = 0.0001f;
                    diffusion = 0.0001f;
                    printf("Parameters reset!\n");
                }
            }

            if (event.type == ButtonPress) {
                mouseDown = 1;
                mouseButton = event.xbutton.button;
                lastMouseX = event.xbutton.x;
                lastMouseY = event.xbutton.y;
            }

            if (event.type == ButtonRelease) {
                mouseDown = 0;
            }

            if (event.type == MotionNotify && mouseDown) {
                int mx = event.xmotion.x;
                int my = event.xmotion.y;

                // Map to simulation coordinates
                int sx = mx * SIM_WIDTH / DISP_WIDTH;
                int sy = (DISP_HEIGHT - 1 - my) * SIM_HEIGHT / DISP_HEIGHT;

                // Calculate velocity from mouse movement
                float vx = (mx - lastMouseX) * 5.0f;
                float vy = -(my - lastMouseY) * 5.0f;

                // Add density
                splatKernel<<<simGrid, simBlock>>>(d_density, sx, sy, 15.0f, 0.8f, dt);

                // Add velocity (only for left button)
                if (mouseButton == Button1) {
                    splatVelocityKernel<<<simGrid, simBlock>>>(d_velX, d_velY,
                        sx, sy, 15.0f, vx, vy, dt);
                }

                lastMouseX = mx;
                lastMouseY = my;
            }

            if (event.type == DestroyNotify) goto cleanup;
        }

        double now = getTime();
        float frameDt = (float)(now - lastTime);
        if (frameDt > 0.05f) frameDt = 0.05f;
        lastTime = now;

        // ========== FLUID SIMULATION STEP ==========

        // --- 1. Add forces (already done via mouse input) ---

        // --- 2. Diffuse velocity ---
        if (viscosity > 0.0f) {
            float alpha = (dt * viscosity * SIM_WIDTH * SIM_HEIGHT);
            float beta = 1.0f / (1.0f + 4.0f * alpha);

            for (int i = 0; i < JACOBI_ITERATIONS; i++) {
                diffuseJacobiKernel<<<simGrid, simBlock>>>(d_velX_prev, d_velX, d_velX, alpha, beta);
                diffuseJacobiKernel<<<simGrid, simBlock>>>(d_velY_prev, d_velY, d_velY, alpha, beta);
                cudaDeviceSynchronize();
                // Swap
                float* tmp = d_velX; d_velX = d_velX_prev; d_velX_prev = tmp;
                tmp = d_velY; d_velY = d_velY_prev; d_velY_prev = tmp;
            }
            setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velX, -1);
            setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velY, -1);
        }

        // --- 3. Project (make divergence-free) ---
        // Compute divergence
        divergenceKernel<<<simGrid, simBlock>>>(d_divergence, d_velX, d_velY);
        cudaMemset(d_pressure, 0, fieldBytes);

        // Solve pressure Poisson equation
        for (int i = 0; i < JACOBI_ITERATIONS; i++) {
            pressureJacobiKernel<<<simGrid, simBlock>>>(d_pressure_prev, d_pressure, d_divergence);
            cudaDeviceSynchronize();
            float* tmp = d_pressure; d_pressure = d_pressure_prev; d_pressure_prev = tmp;
        }
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_pressure, 1);

        // Subtract gradient
        gradientSubtractKernel<<<simGrid, simBlock>>>(d_velX, d_velY, d_pressure);
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velX, -1);
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velY, -1);

        // --- 4. Advect velocity ---
        advectVelocityKernel<<<simGrid, simBlock>>>(d_velX_prev, d_velY_prev,
            d_velX, d_velY, dt * SIM_WIDTH, VELOCITY_DISSIPATION);
        cudaDeviceSynchronize();
        float* tmp = d_velX; d_velX = d_velX_prev; d_velX_prev = tmp;
        tmp = d_velY; d_velY = d_velY_prev; d_velY_prev = tmp;
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velX, -1);
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_velY, -1);

        // --- 5. Project again ---
        divergenceKernel<<<simGrid, simBlock>>>(d_divergence, d_velX, d_velY);
        cudaMemset(d_pressure, 0, fieldBytes);
        for (int i = 0; i < JACOBI_ITERATIONS; i++) {
            pressureJacobiKernel<<<simGrid, simBlock>>>(d_pressure_prev, d_pressure, d_divergence);
            cudaDeviceSynchronize();
            tmp = d_pressure; d_pressure = d_pressure_prev; d_pressure_prev = tmp;
        }
        gradientSubtractKernel<<<simGrid, simBlock>>>(d_velX, d_velY, d_pressure);

        // --- 6. Diffuse density ---
        if (diffusion > 0.0f) {
            float alpha = (dt * diffusion * SIM_WIDTH * SIM_HEIGHT);
            float beta = 1.0f / (1.0f + 4.0f * alpha);

            for (int i = 0; i < JACOBI_ITERATIONS / 2; i++) {
                diffuseJacobiKernel<<<simGrid, simBlock>>>(d_density_prev, d_density, d_density, alpha, beta);
                cudaDeviceSynchronize();
                tmp = d_density; d_density = d_density_prev; d_density_prev = tmp;
            }
            setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_density, 1);
        }

        // --- 7. Advect density ---
        advectKernel<<<simGrid, simBlock>>>(d_density_prev, d_density, d_velX, d_velY,
            dt * SIM_WIDTH, DENSITY_DISSIPATION);
        cudaDeviceSynchronize();
        tmp = d_density; d_density = d_density_prev; d_density_prev = tmp;
        setBoundaryKernel<<<(boundaryThreads+255)/256, 256>>>(d_density, 1);

        // ========== RENDER ==========
        renderKernel<<<dispGrid, dispBlock>>>(d_pixels, d_density, d_velX, d_velY,
            DISP_WIDTH, DISP_HEIGHT, colorScheme, showVelocity);

        cudaDeviceSynchronize();

        // Copy and display
        cudaMemcpy(h_pixels, d_pixels, DISP_WIDTH * DISP_HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, image, 0, 0, 0, 0, DISP_WIDTH, DISP_HEIGHT);
        XFlush(display);

        frameCount++;
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f\n", frameCount / (now - lastFpsTime));
            frameCount = 0;
            lastFpsTime = now;
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    XFreeGC(display, gc);
    image->data = NULL;
    XDestroyImage(image);
    XDestroyWindow(display, window);
    XCloseDisplay(display);

    cudaFree(d_velX);
    cudaFree(d_velY);
    cudaFree(d_velX_prev);
    cudaFree(d_velY_prev);
    cudaFree(d_density);
    cudaFree(d_density_prev);
    cudaFree(d_pressure);
    cudaFree(d_pressure_prev);
    cudaFree(d_divergence);
    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
