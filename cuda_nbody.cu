/*
 * Jetson Nano CUDA N-Body Gravity Simulation
 *
 * Real-time gravitational simulation with thousands of bodies
 * Uses tiled shared-memory for efficient O(N²) force calculation
 *
 * Features:
 *   - Tiled shared-memory acceleration
 *   - Multiple galaxy presets
 *   - Softened gravity (prevents singularities)
 *   - Trail rendering
 *   - Interactive camera
 *   - Mass-based coloring
 *
 * Controls:
 *   1-5         - Galaxy presets
 *   Arrow keys  - Rotate view
 *   W/S         - Zoom in/out
 *   +/-         - Adjust time step
 *   T           - Toggle trails
 *   Space       - Pause/resume
 *   R           - Reset current preset
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

#define WIDTH 900
#define HEIGHT 700

// Simulation parameters
#define MAX_BODIES 8192
#define TILE_SIZE 128       // Bodies per shared memory tile
#define SOFTENING 0.5f      // Softening factor to prevent singularities
#define G 0.5f              // Gravitational constant

// Body structure (SoA for coalesced memory access)
struct Bodies {
    float* x;
    float* y;
    float* z;
    float* vx;
    float* vy;
    float* vz;
    float* mass;
    int count;
};

// ============== FORCE COMPUTATION (TILED) ==============
__global__ void computeForcesKernel(
    float* ax, float* ay, float* az,
    const float* px, const float* py, const float* pz,
    const float* mass,
    int n, float softening2, float grav)
{
    // Shared memory tile for body positions and masses
    __shared__ float4 tile[TILE_SIZE];  // x, y, z, mass

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float accx = 0.0f, accy = 0.0f, accz = 0.0f;
    float myX, myY, myZ;

    if (i < n) {
        myX = px[i];
        myY = py[i];
        myZ = pz[i];
    }

    // Process all tiles
    int numTiles = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile_idx = 0; tile_idx < numTiles; tile_idx++) {
        // Load tile into shared memory
        int j = tile_idx * TILE_SIZE + threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = make_float4(px[j], py[j], pz[j], mass[j]);
        } else {
            tile[threadIdx.x] = make_float4(0, 0, 0, 0);
        }
        __syncthreads();

        // Compute forces from this tile
        if (i < n) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                float dx = tile[k].x - myX;
                float dy = tile[k].y - myY;
                float dz = tile[k].z - myZ;

                float dist2 = dx * dx + dy * dy + dz * dz + softening2;
                float invDist = rsqrtf(dist2);
                float invDist3 = invDist * invDist * invDist;

                float force = G * tile[k].w * invDist3;

                accx += dx * force;
                accy += dy * force;
                accz += dz * force;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        ax[i] = accx;
        ay[i] = accy;
        az[i] = accz;
    }
}

// ============== INTEGRATION (Leapfrog) ==============
__global__ void integrateKernel(
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    const float* ax, const float* ay, const float* az,
    int n, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Update velocity
    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    // Update position
    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}

// ============== RENDERING ==============
__device__ void hsv2rgb(float h, float s, float v, float* r, float* g, float* b) {
    int hi = (int)(h * 6.0f) % 6;
    float f = h * 6.0f - (int)(h * 6.0f);
    float p = v * (1.0f - s);
    float q = v * (1.0f - f * s);
    float t = v * (1.0f - (1.0f - f) * s);

    switch (hi) {
        case 0: *r = v; *g = t; *b = p; break;
        case 1: *r = q; *g = v; *b = p; break;
        case 2: *r = p; *g = v; *b = t; break;
        case 3: *r = p; *g = q; *b = v; break;
        case 4: *r = t; *g = p; *b = v; break;
        default: *r = v; *g = p; *b = q; break;
    }
}

__global__ void clearKernel(unsigned char* pixels, int width, int height, int useTrails) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    if (useTrails) {
        // Fade existing pixels for trail effect
        pixels[idx + 0] = (unsigned char)(pixels[idx + 0] * 0.92f);
        pixels[idx + 1] = (unsigned char)(pixels[idx + 1] * 0.92f);
        pixels[idx + 2] = (unsigned char)(pixels[idx + 2] * 0.92f);
    } else {
        // Dark background
        pixels[idx + 0] = 5;
        pixels[idx + 1] = 5;
        pixels[idx + 2] = 10;
    }
    pixels[idx + 3] = 255;
}

__global__ void renderBodiesKernel(
    unsigned char* pixels, int width, int height,
    const float* px, const float* py, const float* pz,
    const float* mass,
    int n,
    float camX, float camY, float camZ,
    float rotX, float rotY, float zoom)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Get body position relative to camera
    float x = px[i] - camX;
    float y = py[i] - camY;
    float z = pz[i] - camZ;

    // Rotate around Y axis
    float cosY = cosf(rotY), sinY = sinf(rotY);
    float rx = x * cosY + z * sinY;
    float rz = -x * sinY + z * cosY;

    // Rotate around X axis
    float cosX = cosf(rotX), sinX = sinf(rotX);
    float ry = y * cosX - rz * sinX;
    float rzFinal = y * sinX + rz * cosX;

    // Perspective projection
    float depth = rzFinal + zoom;
    if (depth < 1.0f) return;

    float scale = 400.0f / depth;
    int screenX = (int)(width / 2 + rx * scale);
    int screenY = (int)(height / 2 - ry * scale);

    if (screenX < 0 || screenX >= width || screenY < 0 || screenY >= height) return;

    // Color based on mass
    float m = mass[i];
    float hue = fmodf(0.6f - logf(m + 0.1f) * 0.15f, 1.0f);
    if (hue < 0) hue += 1.0f;
    float sat = 0.8f;
    float val = fminf(1.0f, 0.5f + m * 0.5f);

    float r, g, b;
    hsv2rgb(hue, sat, val, &r, &g, &b);

    // Draw body (brighter for higher mass)
    int idx = (screenY * width + screenX) * 4;

    // Additive blending for glow effect
    int newB = pixels[idx + 0] + (int)(b * 200);
    int newG = pixels[idx + 1] + (int)(g * 200);
    int newR = pixels[idx + 2] + (int)(r * 200);

    pixels[idx + 0] = (unsigned char)min(255, newB);
    pixels[idx + 1] = (unsigned char)min(255, newG);
    pixels[idx + 2] = (unsigned char)min(255, newR);

    // Draw larger point for massive bodies
    if (m > 1.5f && screenX > 0 && screenX < width-1 && screenY > 0 && screenY < height-1) {
        int offsets[] = {-1, 0, 1, 0, 0, -1, 0, 1};
        for (int k = 0; k < 8; k += 2) {
            int nx = screenX + offsets[k];
            int ny = screenY + offsets[k+1];
            int nidx = (ny * width + nx) * 4;
            pixels[nidx + 0] = (unsigned char)min(255, pixels[nidx + 0] + (int)(b * 100));
            pixels[nidx + 1] = (unsigned char)min(255, pixels[nidx + 1] + (int)(g * 100));
            pixels[nidx + 2] = (unsigned char)min(255, pixels[nidx + 2] + (int)(r * 100));
        }
    }
}

// ============== GALAXY PRESETS ==============

// Random float in range [0, 1)
float randf() {
    return (float)rand() / (float)RAND_MAX;
}

// Gaussian random
float gaussRand() {
    float u1 = randf() + 0.0001f;
    float u2 = randf();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
}

void initUniformSphere(Bodies* b, int n) {
    b->count = n;
    for (int i = 0; i < n; i++) {
        // Uniform distribution in sphere
        float theta = randf() * 2.0f * 3.14159f;
        float phi = acosf(2.0f * randf() - 1.0f);
        float r = powf(randf(), 1.0f/3.0f) * 30.0f;

        b->x[i] = r * sinf(phi) * cosf(theta);
        b->y[i] = r * sinf(phi) * sinf(theta);
        b->z[i] = r * cosf(phi);

        b->vx[i] = gaussRand() * 0.5f;
        b->vy[i] = gaussRand() * 0.5f;
        b->vz[i] = gaussRand() * 0.5f;

        b->mass[i] = 0.5f + randf() * 0.5f;
    }
}

void initRotatingDisk(Bodies* b, int n) {
    b->count = n;
    for (int i = 0; i < n; i++) {
        // Disk in XZ plane
        float theta = randf() * 2.0f * 3.14159f;
        float r = 5.0f + randf() * 25.0f;

        b->x[i] = r * cosf(theta);
        b->y[i] = gaussRand() * 1.0f;  // Thin disk
        b->z[i] = r * sinf(theta);

        // Circular velocity (Keplerian-ish)
        float v = sqrtf(G * n * 0.5f / r) * 0.3f;
        b->vx[i] = -v * sinf(theta) + gaussRand() * 0.2f;
        b->vy[i] = gaussRand() * 0.1f;
        b->vz[i] = v * cosf(theta) + gaussRand() * 0.2f;

        b->mass[i] = 0.3f + randf() * 0.4f;
    }
}

void initCollidingGalaxies(Bodies* b, int n) {
    b->count = n;
    int half = n / 2;

    // Galaxy 1
    for (int i = 0; i < half; i++) {
        float theta = randf() * 2.0f * 3.14159f;
        float r = 3.0f + randf() * 15.0f;

        b->x[i] = -25.0f + r * cosf(theta);
        b->y[i] = gaussRand() * 0.5f;
        b->z[i] = r * sinf(theta);

        float v = sqrtf(G * half * 0.3f / r) * 0.2f;
        b->vx[i] = 1.5f - v * sinf(theta);
        b->vy[i] = gaussRand() * 0.1f;
        b->vz[i] = v * cosf(theta);

        b->mass[i] = 0.3f + randf() * 0.3f;
    }

    // Galaxy 2
    for (int i = half; i < n; i++) {
        float theta = randf() * 2.0f * 3.14159f;
        float r = 3.0f + randf() * 15.0f;

        b->x[i] = 25.0f + r * cosf(theta);
        b->y[i] = 5.0f + gaussRand() * 0.5f;
        b->z[i] = r * sinf(theta);

        float v = sqrtf(G * half * 0.3f / r) * 0.2f;
        b->vx[i] = -1.5f + v * sinf(theta);
        b->vy[i] = -0.3f + gaussRand() * 0.1f;
        b->vz[i] = -v * cosf(theta);

        b->mass[i] = 0.3f + randf() * 0.3f;
    }
}

void initCentralMass(Bodies* b, int n) {
    b->count = n;

    // Central massive body
    b->x[0] = 0; b->y[0] = 0; b->z[0] = 0;
    b->vx[0] = 0; b->vy[0] = 0; b->vz[0] = 0;
    b->mass[0] = 50.0f;

    // Orbiting bodies
    for (int i = 1; i < n; i++) {
        float theta = randf() * 2.0f * 3.14159f;
        float phi = acosf(2.0f * randf() - 1.0f);
        float r = 10.0f + randf() * 30.0f;

        b->x[i] = r * sinf(phi) * cosf(theta);
        b->y[i] = r * sinf(phi) * sinf(theta) * 0.3f;  // Flattened
        b->z[i] = r * cosf(phi);

        // Orbital velocity
        float v = sqrtf(G * b->mass[0] / r) * 0.8f;
        // Velocity perpendicular to radius in XZ plane
        b->vx[i] = -v * sinf(theta) + gaussRand() * 0.3f;
        b->vy[i] = gaussRand() * 0.1f;
        b->vz[i] = v * cosf(theta) + gaussRand() * 0.3f;

        b->mass[i] = 0.1f + randf() * 0.2f;
    }
}

void initFigure8(Bodies* b, int n) {
    b->count = n;

    // Figure-8 three-body initial conditions (scaled up)
    // Plus many small particles
    float scale = 15.0f;
    float vscale = 1.5f;

    // Three main bodies
    b->x[0] = -0.97000436f * scale;
    b->y[0] = 0.24308753f * scale;
    b->z[0] = 0;
    b->vx[0] = 0.4662036850f * vscale;
    b->vy[0] = 0.4323657300f * vscale;
    b->vz[0] = 0;
    b->mass[0] = 5.0f;

    b->x[1] = 0.97000436f * scale;
    b->y[1] = -0.24308753f * scale;
    b->z[1] = 0;
    b->vx[1] = 0.4662036850f * vscale;
    b->vy[1] = 0.4323657300f * vscale;
    b->vz[1] = 0;
    b->mass[1] = 5.0f;

    b->x[2] = 0;
    b->y[2] = 0;
    b->z[2] = 0;
    b->vx[2] = -0.93240737f * vscale;
    b->vy[2] = -0.86473146f * vscale;
    b->vz[2] = 0;
    b->mass[2] = 5.0f;

    // Add small particles around
    for (int i = 3; i < n; i++) {
        float theta = randf() * 2.0f * 3.14159f;
        float r = 20.0f + randf() * 20.0f;

        b->x[i] = r * cosf(theta);
        b->y[i] = r * sinf(theta);
        b->z[i] = gaussRand() * 2.0f;

        b->vx[i] = gaussRand() * 0.3f;
        b->vy[i] = gaussRand() * 0.3f;
        b->vz[i] = gaussRand() * 0.1f;

        b->mass[i] = 0.05f + randf() * 0.1f;
    }
}

// ============== HOST CODE ==============

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("=== Jetson Nano CUDA N-Body Gravity Simulation ===\n\n");
    printf("Controls:\n");
    printf("  1       - Uniform sphere collapse\n");
    printf("  2       - Rotating disk galaxy\n");
    printf("  3       - Colliding galaxies\n");
    printf("  4       - Central mass + orbiters\n");
    printf("  5       - Figure-8 three-body\n");
    printf("  Arrows  - Rotate view\n");
    printf("  W/S     - Zoom in/out\n");
    printf("  +/-     - Time step\n");
    printf("  T       - Toggle trails\n");
    printf("  Space   - Pause/resume\n");
    printf("  R       - Reset current preset\n");
    printf("  Q/Esc   - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    // Open X11
    Display* display = XOpenDisplay(NULL);
    if (!display) {
        fprintf(stderr, "Cannot open X display\n");
        return 1;
    }

    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);

    XSetWindowAttributes attrs;
    attrs.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
    attrs.background_pixel = BlackPixel(display, screen);

    Window window = XCreateWindow(display, root,
        50, 50, WIDTH, HEIGHT, 0,
        CopyFromParent, InputOutput, CopyFromParent,
        CWEventMask | CWBackPixel, &attrs);

    XStoreName(display, window, "CUDA N-Body Gravity Simulation");
    XMapWindow(display, window);

    XEvent event;
    while (1) {
        XNextEvent(display, &event);
        if (event.type == MapNotify) break;
    }

    // Allocate host bodies
    Bodies h_bodies;
    cudaMallocHost(&h_bodies.x, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.y, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.z, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.vx, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.vy, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.vz, MAX_BODIES * sizeof(float));
    cudaMallocHost(&h_bodies.mass, MAX_BODIES * sizeof(float));

    // Allocate device memory
    float *d_x, *d_y, *d_z, *d_vx, *d_vy, *d_vz, *d_mass;
    float *d_ax, *d_ay, *d_az;

    cudaMalloc(&d_x, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_y, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_z, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_vx, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_vy, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_vz, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_mass, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_ax, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_ay, MAX_BODIES * sizeof(float));
    cudaMalloc(&d_az, MAX_BODIES * sizeof(float));

    // Allocate display buffer
    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);

    Visual* visual = DefaultVisual(display, screen);
    int depth = DefaultDepth(display, screen);
    XImage* image = XCreateImage(display, visual, depth, ZPixmap, 0,
        (char*)h_pixels, WIDTH, HEIGHT, 32, WIDTH * 4);

    GC gc = XCreateGC(display, window, 0, NULL);

    // Simulation state
    int numBodies = 4096;
    int preset = 2;  // Start with rotating disk
    float dt = 0.02f;
    int paused = 0;
    int showTrails = 1;
    float rotX = 0.3f, rotY = 0.0f;
    float zoom = 80.0f;
    float camX = 0, camY = 0, camZ = 0;

    srand(42);

    // Initialize with rotating disk
    initRotatingDisk(&h_bodies, numBodies);

    // Copy to device
    cudaMemcpy(d_x, h_bodies.x, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_bodies.y, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_bodies.z, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_bodies.vx, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_bodies.vy, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_bodies.vz, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_bodies.mass, numBodies * sizeof(float), cudaMemcpyHostToDevice);

    const char* presetNames[] = {"Sphere Collapse", "Rotating Disk", "Colliding Galaxies",
                                  "Central Mass", "Figure-8"};
    printf("Bodies: %d, Preset: %s\n", numBodies, presetNames[preset-1]);
    printf("Trails: ON\n");

    dim3 blockSize(TILE_SIZE);
    dim3 gridSize((numBodies + TILE_SIZE - 1) / TILE_SIZE);

    dim3 dispBlockSize(16, 16);
    dim3 dispGridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    double lastTime = getTime();
    double lastFpsTime = lastTime;
    int frameCount = 0;
    float softening2 = SOFTENING * SOFTENING;

    while (1) {
        // Handle events
        while (XPending(display)) {
            XNextEvent(display, &event);

            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);

                if (key == XK_Escape || key == XK_q) goto cleanup;

                // View controls
                if (key == XK_Left) rotY -= 0.1f;
                if (key == XK_Right) rotY += 0.1f;
                if (key == XK_Up) rotX -= 0.1f;
                if (key == XK_Down) rotX += 0.1f;
                if (key == XK_w) zoom -= 5.0f;
                if (key == XK_s) zoom += 5.0f;
                zoom = fmaxf(20.0f, fminf(200.0f, zoom));

                // Time step
                if (key == XK_plus || key == XK_equal) dt *= 1.2f;
                if (key == XK_minus) dt /= 1.2f;
                dt = fmaxf(0.001f, fminf(0.1f, dt));

                // Toggles
                if (key == XK_space) {
                    paused = !paused;
                    printf("%s\n", paused ? "Paused" : "Running");
                }
                if (key == XK_t) {
                    showTrails = !showTrails;
                    printf("Trails: %s\n", showTrails ? "ON" : "OFF");
                }

                // Presets
                int newPreset = 0;
                if (key == XK_1) newPreset = 1;
                if (key == XK_2) newPreset = 2;
                if (key == XK_3) newPreset = 3;
                if (key == XK_4) newPreset = 4;
                if (key == XK_5) newPreset = 5;
                if (key == XK_r) newPreset = preset;  // Reset current

                if (newPreset > 0) {
                    preset = newPreset;
                    srand(time(NULL));

                    switch (preset) {
                        case 1: initUniformSphere(&h_bodies, numBodies); break;
                        case 2: initRotatingDisk(&h_bodies, numBodies); break;
                        case 3: initCollidingGalaxies(&h_bodies, numBodies); break;
                        case 4: initCentralMass(&h_bodies, numBodies); break;
                        case 5: initFigure8(&h_bodies, numBodies); break;
                    }

                    cudaMemcpy(d_x, h_bodies.x, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_y, h_bodies.y, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_z, h_bodies.z, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_vx, h_bodies.vx, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_vy, h_bodies.vy, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_vz, h_bodies.vz, numBodies * sizeof(float), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_mass, h_bodies.mass, numBodies * sizeof(float), cudaMemcpyHostToDevice);

                    // Clear screen
                    cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);

                    printf("Preset: %s\n", presetNames[preset-1]);
                }
            }

            if (event.type == DestroyNotify) goto cleanup;
        }

        // Physics simulation
        if (!paused) {
            // Compute forces
            computeForcesKernel<<<gridSize, blockSize>>>(
                d_ax, d_ay, d_az,
                d_x, d_y, d_z, d_mass,
                numBodies, softening2, G);

            // Integrate
            integrateKernel<<<gridSize, blockSize>>>(
                d_x, d_y, d_z, d_vx, d_vy, d_vz,
                d_ax, d_ay, d_az,
                numBodies, dt);
        }

        // Render
        clearKernel<<<dispGridSize, dispBlockSize>>>(d_pixels, WIDTH, HEIGHT, showTrails);

        renderBodiesKernel<<<gridSize, blockSize>>>(
            d_pixels, WIDTH, HEIGHT,
            d_x, d_y, d_z, d_mass,
            numBodies,
            camX, camY, camZ,
            rotX, rotY, zoom);

        cudaDeviceSynchronize();

        // Display
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, image, 0, 0, 0, 0, WIDTH, HEIGHT);
        XFlush(display);

        frameCount++;
        double now = getTime();
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f | Bodies: %d | dt: %.4f\n",
                   frameCount / (now - lastFpsTime), numBodies, dt);
            frameCount = 0;
            lastFpsTime = now;
        }
        lastTime = now;
    }

cleanup:
    printf("\nCleaning up...\n");

    XFreeGC(display, gc);
    image->data = NULL;
    XDestroyImage(image);
    XDestroyWindow(display, window);
    XCloseDisplay(display);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_mass);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_pixels);

    cudaFreeHost(h_bodies.x); cudaFreeHost(h_bodies.y); cudaFreeHost(h_bodies.z);
    cudaFreeHost(h_bodies.vx); cudaFreeHost(h_bodies.vy); cudaFreeHost(h_bodies.vz);
    cudaFreeHost(h_bodies.mass);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
