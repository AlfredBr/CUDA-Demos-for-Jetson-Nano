/*
 * Windows CUDA SGI Pipes Screensaver Recreation
 *
 * A nostalgic recreation of the classic SGI "Pipes" screensaver
 * Pipes grow through 3D space, making right-angle turns with ball joints
 *
 * Features:
 *   - 3D pipe rendering with cylinders
 *   - Ball joints at connection points
 *   - Metallic shading with specular highlights
 *   - Multiple colored pipes
 *   - Auto-rotating camera
 *   - Configurable pipe count and speed
 *
 * Controls:
 *   Arrow keys  - Rotate view manually
 *   W/S         - Zoom in/out
 *   +/-         - Adjust growth speed
 *   A           - Toggle auto-rotate
 *   C           - Clear and restart
 *   P           - Add another pipe
 *   Space       - Pause/resume
 *   Q/Escape    - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "win32_display.h"

#define WIDTH 1024
#define HEIGHT 768

#define PI 3.14159265359f
#define TWO_PI 6.28318530718f

// Pipe system parameters
#define MAX_PIPES 16
#define MAX_SEGMENTS_PER_PIPE 500
#define GRID_SIZE 20           // Size of the virtual grid
#define PIPE_RADIUS 0.4f
#define JOINT_RADIUS 0.55f

// Direction vectors (6 directions: +X, -X, +Y, -Y, +Z, -Z)
__constant__ float3 directions[6] = {
    {1, 0, 0}, {-1, 0, 0},
    {0, 1, 0}, {0, -1, 0},
    {0, 0, 1}, {0, 0, -1}
};

// Pipe segment structure
struct PipeSegment {
    float3 start;
    float3 end;
    int direction;  // 0-5 for the 6 directions
    float progress;  // 0-1 for growth animation
};

// Pipe structure
struct Pipe {
    PipeSegment segments[MAX_SEGMENTS_PER_PIPE];
    int numSegments;
    float3 currentPos;
    int currentDir;
    float3 color;
    int active;
    float growthProgress;  // Progress of current segment growth
};

// Predefined metallic colors
__device__ __host__ float3 getPipeColor(int index) {
    float3 colors[] = {
        {0.8f, 0.2f, 0.2f},   // Red
        {0.2f, 0.8f, 0.2f},   // Green
        {0.2f, 0.4f, 0.9f},   // Blue
        {0.9f, 0.8f, 0.2f},   // Gold
        {0.8f, 0.2f, 0.8f},   // Magenta
        {0.2f, 0.8f, 0.8f},   // Cyan
        {0.9f, 0.5f, 0.2f},   // Orange
        {0.7f, 0.7f, 0.7f},   // Silver
    };
    return colors[index % 8];
}

// ============== 3D RENDERING HELPERS ==============

__device__ float3 normalize3(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0.0001f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return make_float3(0, 1, 0);
}

__device__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 reflect3(float3 v, float3 n) {
    float d = 2.0f * dot3(v, n);
    return make_float3(v.x - d * n.x, v.y - d * n.y, v.z - d * n.z);
}

// Ray-sphere intersection
__device__ float intersectSphere(float3 ro, float3 rd, float3 center, float radius) {
    float3 oc = make_float3(ro.x - center.x, ro.y - center.y, ro.z - center.z);
    float b = dot3(oc, rd);
    float c = dot3(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0f) return -1.0f;
    return -b - sqrtf(h);
}

// Ray-cylinder intersection (capped)
__device__ float intersectCylinder(float3 ro, float3 rd, float3 pa, float3 pb, float radius, float3* outNormal) {
    float3 ba = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
    float3 oc = make_float3(ro.x - pa.x, ro.y - pa.y, ro.z - pa.z);

    float baba = dot3(ba, ba);
    float bard = dot3(ba, rd);
    float baoc = dot3(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot3(oc, rd) - baoc * bard;
    float k0 = baba * dot3(oc, oc) - baoc * baoc - radius * radius * baba;

    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return -1.0f;

    h = sqrtf(h);
    float t = (-k1 - h) / k2;

    // Check if hit is within cylinder caps
    float y = baoc + t * bard;
    if (y > 0.0f && y < baba) {
        // Calculate normal
        float3 hitPoint = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
        float3 toHit = make_float3(hitPoint.x - pa.x, hitPoint.y - pa.y, hitPoint.z - pa.z);
        float proj = dot3(toHit, ba) / baba;
        float3 onAxis = make_float3(pa.x + proj * ba.x, pa.y + proj * ba.y, pa.z + proj * ba.z);
        *outNormal = normalize3(make_float3(hitPoint.x - onAxis.x, hitPoint.y - onAxis.y, hitPoint.z - onAxis.z));
        return t;
    }

    return -1.0f;
}

// ============== CUDA KERNELS ==============

__global__ void clearKernel(unsigned char* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    // Dark gradient background
    float gy = (float)y / height;
    int bg = (int)(10 + gy * 20);

    pixels[idx + 0] = bg;
    pixels[idx + 1] = bg;
    pixels[idx + 2] = bg + 5;
    pixels[idx + 3] = 255;
}

__global__ void renderPipesKernel(
    unsigned char* pixels, int width, int height,
    Pipe* pipes, int numPipes,
    float camX, float camY, float camZ,
    float rotX, float rotY, float zoom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Camera setup
    float aspectRatio = (float)width / height;
    float fovScale = tanf(0.5f * 1.0f);  // ~57 degree FOV

    // Normalized device coordinates
    float ndcX = (2.0f * x / width - 1.0f) * aspectRatio * fovScale;
    float ndcY = (1.0f - 2.0f * y / height) * fovScale;

    // Ray direction in camera space
    float3 rd = normalize3(make_float3(ndcX, ndcY, 1.0f));

    // Rotate ray direction
    float cosY = cosf(rotY), sinY = sinf(rotY);
    float cosX = cosf(rotX), sinX = sinf(rotX);

    // Rotate around Y
    float rx = rd.x * cosY + rd.z * sinY;
    float rz = -rd.x * sinY + rd.z * cosY;
    rd.x = rx;
    rd.z = rz;

    // Rotate around X
    float ry = rd.y * cosX - rd.z * sinX;
    rz = rd.y * sinX + rd.z * cosX;
    rd.y = ry;
    rd.z = rz;

    // Ray origin
    float3 ro = make_float3(camX, camY, camZ - zoom);

    // Rotate camera position
    float cox = ro.x * cosY + ro.z * sinY;
    float coz = -ro.x * sinY + ro.z * cosY;
    ro.x = cox;
    ro.z = coz;

    float coy = ro.y * cosX - ro.z * sinX;
    coz = ro.y * sinX + ro.z * cosX;
    ro.y = coy;
    ro.z = coz;

    // Find closest intersection
    float minT = 1e10f;
    float3 hitNormal;
    float3 hitColor;
    int hit = 0;

    // Light direction
    float3 lightDir = normalize3(make_float3(0.5f, 1.0f, 0.3f));

    // Test all pipes
    for (int p = 0; p < numPipes; p++) {
        // Render pipes that have segments (even if inactive/done growing)
        if (pipes[p].numSegments == 0) continue;

        float3 pipeColor = pipes[p].color;

        // Test all segments
        for (int s = 0; s < pipes[p].numSegments; s++) {
            PipeSegment* seg = &pipes[p].segments[s];

            // Calculate actual end point based on progress
            float3 actualEnd;
            if (s == pipes[p].numSegments - 1) {
                // Last segment - apply growth progress
                float prog = seg->progress;
                actualEnd.x = seg->start.x + (seg->end.x - seg->start.x) * prog;
                actualEnd.y = seg->start.y + (seg->end.y - seg->start.y) * prog;
                actualEnd.z = seg->start.z + (seg->end.z - seg->start.z) * prog;
            } else {
                actualEnd = seg->end;
            }

            // Test cylinder
            float3 cylNormal;
            float t = intersectCylinder(ro, rd, seg->start, actualEnd, PIPE_RADIUS, &cylNormal);
            if (t > 0.0f && t < minT) {
                minT = t;
                hitNormal = cylNormal;
                hitColor = pipeColor;
                hit = 1;
            }

            // Test joint sphere at start (except for first segment)
            if (s > 0) {
                t = intersectSphere(ro, rd, seg->start, JOINT_RADIUS);
                if (t > 0.0f && t < minT) {
                    minT = t;
                    float3 hitPos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
                    hitNormal = normalize3(make_float3(
                        hitPos.x - seg->start.x,
                        hitPos.y - seg->start.y,
                        hitPos.z - seg->start.z));
                    hitColor = pipeColor;
                    hit = 1;
                }
            }

            // Test cap sphere at end of last segment
            if (s == pipes[p].numSegments - 1) {
                t = intersectSphere(ro, rd, actualEnd, JOINT_RADIUS);
                if (t > 0.0f && t < minT) {
                    minT = t;
                    float3 hitPos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
                    hitNormal = normalize3(make_float3(
                        hitPos.x - actualEnd.x,
                        hitPos.y - actualEnd.y,
                        hitPos.z - actualEnd.z));
                    hitColor = pipeColor;
                    hit = 1;
                }
            }
        }
    }

    int idx = (y * width + x) * 4;

    if (hit) {
        // Phong shading
        float diffuse = fmaxf(0.0f, dot3(hitNormal, lightDir));

        // Specular
        float3 viewDir = normalize3(make_float3(-rd.x, -rd.y, -rd.z));
        float3 reflectDir = reflect3(make_float3(-lightDir.x, -lightDir.y, -lightDir.z), hitNormal);
        float spec = powf(fmaxf(0.0f, dot3(viewDir, reflectDir)), 32.0f);

        // Metallic shading
        float ambient = 0.15f;
        float3 finalColor;
        finalColor.x = hitColor.x * (ambient + diffuse * 0.6f) + spec * 0.8f;
        finalColor.y = hitColor.y * (ambient + diffuse * 0.6f) + spec * 0.8f;
        finalColor.z = hitColor.z * (ambient + diffuse * 0.6f) + spec * 0.8f;

        // Add environment reflection tint
        float fresnel = powf(1.0f - fmaxf(0.0f, dot3(viewDir, hitNormal)), 3.0f);
        finalColor.x += fresnel * 0.3f;
        finalColor.y += fresnel * 0.3f;
        finalColor.z += fresnel * 0.4f;

        pixels[idx + 0] = (unsigned char)fminf(255.0f, finalColor.z * 255.0f);
        pixels[idx + 1] = (unsigned char)fminf(255.0f, finalColor.y * 255.0f);
        pixels[idx + 2] = (unsigned char)fminf(255.0f, finalColor.x * 255.0f);
    }
}

// ============== HOST CODE ==============

// Random float [0, 1)
float randf() {
    return (float)rand() / (float)RAND_MAX;
}

// Initialize a new pipe
void initPipe(Pipe* pipe, int colorIndex) {
    pipe->numSegments = 0;
    pipe->active = 1;
    pipe->color = getPipeColor(colorIndex);

    // Start at random grid position
    pipe->currentPos.x = (float)((rand() % GRID_SIZE) - GRID_SIZE / 2);
    pipe->currentPos.y = (float)((rand() % GRID_SIZE) - GRID_SIZE / 2);
    pipe->currentPos.z = (float)((rand() % GRID_SIZE) - GRID_SIZE / 2);

    // Random initial direction
    pipe->currentDir = rand() % 6;
    pipe->growthProgress = 0.0f;
}

// Choose new direction (perpendicular to current)
int chooseNewDirection(int currentDir) {
    // Get perpendicular directions
    int perpDirs[4];
    int count = 0;

    for (int d = 0; d < 6; d++) {
        // Skip same axis (current and opposite)
        if (d / 2 != currentDir / 2) {
            perpDirs[count++] = d;
        }
    }

    return perpDirs[rand() % 4];
}

// Grow pipe by one step
void growPipe(Pipe* pipe) {
    if (pipe->numSegments >= MAX_SEGMENTS_PER_PIPE - 1) {
        pipe->active = 0;
        return;
    }

    // Create new segment
    PipeSegment* seg = &pipe->segments[pipe->numSegments];
    seg->start = pipe->currentPos;
    seg->direction = pipe->currentDir;
    seg->progress = 0.0f;

    // Calculate end position (1 unit in current direction)
    float3 dir = make_float3(
        (pipe->currentDir == 0) ? 1.0f : (pipe->currentDir == 1) ? -1.0f : 0.0f,
        (pipe->currentDir == 2) ? 1.0f : (pipe->currentDir == 3) ? -1.0f : 0.0f,
        (pipe->currentDir == 4) ? 1.0f : (pipe->currentDir == 5) ? -1.0f : 0.0f
    );

    float segmentLength = 1.0f + randf() * 2.0f;  // Variable length
    seg->end.x = seg->start.x + dir.x * segmentLength;
    seg->end.y = seg->start.y + dir.y * segmentLength;
    seg->end.z = seg->start.z + dir.z * segmentLength;

    pipe->numSegments++;

    // Update current position
    pipe->currentPos = seg->end;

    // Check bounds and maybe turn
    int outOfBounds = (fabsf(pipe->currentPos.x) > GRID_SIZE / 2) ||
                      (fabsf(pipe->currentPos.y) > GRID_SIZE / 2) ||
                      (fabsf(pipe->currentPos.z) > GRID_SIZE / 2);

    // Always turn, or if out of bounds, or random chance
    if (outOfBounds || randf() < 0.3f) {
        pipe->currentDir = chooseNewDirection(pipe->currentDir);

        // If out of bounds, reverse back into bounds
        if (outOfBounds) {
            if (pipe->currentPos.x > GRID_SIZE / 2) pipe->currentDir = 1;
            else if (pipe->currentPos.x < -GRID_SIZE / 2) pipe->currentDir = 0;
            else if (pipe->currentPos.y > GRID_SIZE / 2) pipe->currentDir = 3;
            else if (pipe->currentPos.y < -GRID_SIZE / 2) pipe->currentDir = 2;
            else if (pipe->currentPos.z > GRID_SIZE / 2) pipe->currentDir = 5;
            else if (pipe->currentPos.z < -GRID_SIZE / 2) pipe->currentDir = 4;
        }
    }
}

int main() {
    printf("=== Windows CUDA SGI Pipes Screensaver ===\n\n");
    printf("Controls:\n");
    printf("  Arrows  - Rotate view\n");
    printf("  W/S     - Zoom in/out\n");
    printf("  +/-     - Growth speed\n");
    printf("  A       - Toggle auto-rotate\n");
    printf("  C       - Clear and restart\n");
    printf("  P       - Add another pipe\n");
    printf("  Space   - Pause/resume\n");
    printf("  Q/Esc   - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA SGI Pipes", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate pipes
    Pipe* h_pipes = (Pipe*)malloc(MAX_PIPES * sizeof(Pipe));
    Pipe* d_pipes;
    cudaMalloc(&d_pipes, MAX_PIPES * sizeof(Pipe));

    // Allocate display buffer
    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Initialize pipes
    int numPipes = 3;
    for (int i = 0; i < MAX_PIPES; i++) {
        h_pipes[i].active = 0;
        h_pipes[i].numSegments = 0;
    }
    for (int i = 0; i < numPipes; i++) {
        initPipe(&h_pipes[i], i);
    }

    // Simulation state
    float rotX = 0.3f, rotY = 0.0f;
    float zoom = 35.0f;
    float camX = 0, camY = 0, camZ = 0;
    int paused = 0;
    int autoRotate = 1;
    float growthSpeed = 0.05f;
    float growthAccum = 0.0f;

    // Kernel dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Frame rate limiting
    const double TARGET_FRAME_TIME = 1.0 / 60.0;
    double lastTime = win32_get_time(display);
    double lastFpsTime = lastTime;
    int frameCount = 0;

    printf("Pipes: %d | Auto-rotate: ON\n", numPipes);

    while (!win32_should_close(display)) {
        double frameStart = win32_get_time(display);

        // Handle events
        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;

                if (key == XK_Escape || key == XK_q) goto cleanup;

                // View controls
                if (key == XK_Left) rotY -= 0.1f;
                if (key == XK_Right) rotY += 0.1f;
                if (key == XK_Up) rotX -= 0.1f;
                if (key == XK_Down) rotX += 0.1f;
                if (key == XK_w) zoom -= 2.0f;
                if (key == XK_s) zoom += 2.0f;
                zoom = fmaxf(15.0f, fminf(80.0f, zoom));

                // Speed controls
                if (key == XK_plus || key == XK_equal) growthSpeed *= 1.3f;
                if (key == XK_minus) growthSpeed /= 1.3f;
                growthSpeed = fmaxf(0.01f, fminf(0.3f, growthSpeed));

                // Toggles
                if (key == XK_space) {
                    paused = !paused;
                    printf("%s\n", paused ? "Paused" : "Running");
                }
                if (key == XK_a) {
                    autoRotate = !autoRotate;
                    printf("Auto-rotate: %s\n", autoRotate ? "ON" : "OFF");
                }

                // Add pipe
                if (key == XK_p && numPipes < MAX_PIPES) {
                    initPipe(&h_pipes[numPipes], numPipes);
                    numPipes++;
                    printf("Pipes: %d\n", numPipes);
                }

                // Clear and restart
                if (key == XK_c) {
                    numPipes = 3;
                    for (int i = 0; i < MAX_PIPES; i++) {
                        h_pipes[i].active = 0;
                        h_pipes[i].numSegments = 0;
                    }
                    for (int i = 0; i < numPipes; i++) {
                        initPipe(&h_pipes[i], i);
                    }
                    printf("Cleared - Pipes: %d\n", numPipes);
                }
            }

            if (event.type == WIN32_EVENT_CLOSE) goto cleanup;
        }

        // Update simulation
        if (!paused) {
            // Auto-rotate camera
            if (autoRotate) {
                rotY += 0.003f;
            }

            // Grow pipes
            growthAccum += growthSpeed;

            // Update segment growth progress
            for (int i = 0; i < numPipes; i++) {
                if (!h_pipes[i].active) continue;

                if (h_pipes[i].numSegments > 0) {
                    PipeSegment* lastSeg = &h_pipes[i].segments[h_pipes[i].numSegments - 1];
                    lastSeg->progress += growthSpeed * 2.0f;

                    if (lastSeg->progress >= 1.0f) {
                        lastSeg->progress = 1.0f;
                        // Start new segment
                        if (growthAccum >= 1.0f) {
                            growPipe(&h_pipes[i]);
                        }
                    }
                } else {
                    // First segment
                    growPipe(&h_pipes[i]);
                }
            }

            if (growthAccum >= 1.0f) {
                growthAccum = 0.0f;
            }

            // Respawn inactive pipes
            for (int i = 0; i < numPipes; i++) {
                if (!h_pipes[i].active && h_pipes[i].numSegments >= MAX_SEGMENTS_PER_PIPE - 1) {
                    // This pipe is done, could respawn it
                    // For now, leave it visible
                }
            }
        }

        // Copy pipes to device
        cudaMemcpy(d_pipes, h_pipes, MAX_PIPES * sizeof(Pipe), cudaMemcpyHostToDevice);

        // Render
        clearKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT);

        renderPipesKernel<<<gridSize, blockSize>>>(
            d_pixels, WIDTH, HEIGHT,
            d_pipes, numPipes,
            camX, camY, camZ,
            rotX, rotY, zoom);

        cudaDeviceSynchronize();

        // Display
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        // FPS counter
        frameCount++;
        double now = win32_get_time(display);
        if (now - lastFpsTime >= 1.0) {
            int totalSegments = 0;
            for (int i = 0; i < numPipes; i++) {
                totalSegments += h_pipes[i].numSegments;
            }
            printf("FPS: %.1f | Pipes: %d | Segments: %d\n",
                   frameCount / (now - lastFpsTime), numPipes, totalSegments);
            frameCount = 0;
            lastFpsTime = now;
        }
        lastTime = now;

        // Frame rate limiting
        double frameEnd = win32_get_time(display);
        double elapsed = frameEnd - frameStart;
        if (elapsed < TARGET_FRAME_TIME) {
            Sleep((DWORD)((TARGET_FRAME_TIME - elapsed) * 1000.0));
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);

    cudaFree(d_pipes);
    cudaFree(d_pixels);
    free(h_pipes);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
