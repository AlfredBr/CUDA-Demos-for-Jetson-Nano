/*
 * Windows CUDA 3D Demo - Bouncing Ball in Spinning Cube
 * Ported from Jetson Nano version
 *
 * Software 3D rendering entirely on the GPU!
 * Features:
 *   - 3D perspective projection
 *   - Wireframe transparent cube
 *   - Bouncing ball with physics
 *   - Real-time shadows
 *
 * Controls:
 *   Arrow keys - Rotate view
 *   +/-        - Zoom in/out
 *   Space      - Launch ball with random velocity
 *   R          - Reset
 *   Q/Escape   - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600
#define CUBE_SIZE 1.5f
#define BALL_RADIUS 0.2f

// 3D Vector operations
struct Vec3 {
    float x, y, z;
};

__device__ __host__ Vec3 make_vec3(float x, float y, float z) {
    Vec3 v; v.x = x; v.y = y; v.z = z;
    return v;
}

__device__ __host__ Vec3 vec3_add(Vec3 a, Vec3 b) {
    return make_vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ Vec3 vec3_sub(Vec3 a, Vec3 b) {
    return make_vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ Vec3 vec3_mul(Vec3 a, float s) {
    return make_vec3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ float vec3_dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ float vec3_length(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ __host__ Vec3 vec3_normalize(Vec3 v) {
    float len = vec3_length(v);
    if (len > 0.0001f) return vec3_mul(v, 1.0f / len);
    return v;
}

// Rotate point around Y axis
__device__ __host__ Vec3 rotateY(Vec3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return make_vec3(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

// Rotate point around X axis
__device__ __host__ Vec3 rotateX(Vec3 p, float angle) {
    float c = cosf(angle);
    float s = sinf(angle);
    return make_vec3(p.x, p.y * c - p.z * s, p.y * s + p.z * c);
}

// Project 3D point to 2D screen
__device__ __host__ void project(Vec3 p, float camDist, int* sx, int* sy, float* depth) {
    float z = p.z + camDist;
    if (z < 0.1f) z = 0.1f;
    float scale = 300.0f / z;
    *sx = (int)(WIDTH / 2 + p.x * scale);
    *sy = (int)(HEIGHT / 2 - p.y * scale);
    *depth = z;
}

// Draw a line using Bresenham's algorithm
__device__ void drawLine(unsigned char* pixels, int x0, int y0, int x1, int y1,
                         unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    int dx = abs(x1 - x0);
    int dy = abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;

    while (1) {
        if (x0 >= 0 && x0 < WIDTH && y0 >= 0 && y0 < HEIGHT) {
            int idx = (y0 * WIDTH + x0) * 4;
            float alpha = a / 255.0f;
            pixels[idx + 0] = (unsigned char)(b * alpha + pixels[idx + 0] * (1 - alpha));
            pixels[idx + 1] = (unsigned char)(g * alpha + pixels[idx + 1] * (1 - alpha));
            pixels[idx + 2] = (unsigned char)(r * alpha + pixels[idx + 2] * (1 - alpha));
            pixels[idx + 3] = 255;
        }

        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

// Clear framebuffer with gradient background
__global__ void clearKernel(unsigned char* pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    float t = (float)y / height;
    pixels[idx + 0] = (unsigned char)(20 + t * 30);
    pixels[idx + 1] = (unsigned char)(10 + t * 20);
    pixels[idx + 2] = (unsigned char)(30 + t * 20);
    pixels[idx + 3] = 255;
}

// Draw the ball as a filled circle with shading
__global__ void drawBallKernel(unsigned char* pixels, int width, int height,
                                int ballScreenX, int ballScreenY, float ballScreenRadius,
                                float depth, Vec3 ballColor, float rotY) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float dx = x - ballScreenX;
    float dy = y - ballScreenY;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < ballScreenRadius) {
        int idx = (y * width + x) * 4;

        float nx = dx / ballScreenRadius;
        float ny = -dy / ballScreenRadius;
        float nz = sqrtf(fmaxf(0.0f, 1.0f - nx * nx - ny * ny));

        Vec3 lightDir = make_vec3(0.5f, 0.7f, 0.5f);
        lightDir = rotateY(lightDir, -rotY);
        lightDir = vec3_normalize(lightDir);

        float diffuse = fmaxf(0.0f, nx * lightDir.x + ny * lightDir.y + nz * lightDir.z);
        float ambient = 0.3f;
        float light = ambient + diffuse * 0.7f;

        Vec3 viewDir = make_vec3(0, 0, 1);
        Vec3 normal = make_vec3(nx, ny, nz);
        Vec3 reflect = vec3_sub(vec3_mul(normal, 2.0f * vec3_dot(normal, lightDir)), lightDir);
        float spec = powf(fmaxf(0.0f, vec3_dot(reflect, viewDir)), 32.0f);

        float r = fminf(1.0f, ballColor.x * light + spec * 0.5f);
        float g = fminf(1.0f, ballColor.y * light + spec * 0.5f);
        float b = fminf(1.0f, ballColor.z * light + spec * 0.5f);

        float edge = 1.0f - nz;
        r += edge * 0.2f;
        g += edge * 0.3f;
        b += edge * 0.4f;

        pixels[idx + 2] = (unsigned char)(fminf(255.0f, r * 255));
        pixels[idx + 1] = (unsigned char)(fminf(255.0f, g * 255));
        pixels[idx + 0] = (unsigned char)(fminf(255.0f, b * 255));
    }
}

// Draw shadow on the floor
__global__ void drawShadowKernel(unsigned char* pixels, int width, int height,
                                  int shadowX, int shadowY, float shadowRadius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float dx = x - shadowX;
    float dy = (y - shadowY) * 2.0f;
    float dist = sqrtf(dx * dx + dy * dy);

    if (dist < shadowRadius) {
        int idx = (y * width + x) * 4;
        float alpha = 0.4f * (1.0f - dist / shadowRadius);

        pixels[idx + 0] = (unsigned char)(pixels[idx + 0] * (1.0f - alpha));
        pixels[idx + 1] = (unsigned char)(pixels[idx + 1] * (1.0f - alpha));
        pixels[idx + 2] = (unsigned char)(pixels[idx + 2] * (1.0f - alpha));
    }
}

// Draw cube edges
__global__ void drawCubeEdgesKernel(unsigned char* pixels,
                                     float rotX, float rotY, float camDist,
                                     unsigned char r, unsigned char g, unsigned char b) {
    Vec3 verts[8] = {
        make_vec3(-CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        make_vec3( CUBE_SIZE, -CUBE_SIZE, -CUBE_SIZE),
        make_vec3( CUBE_SIZE,  CUBE_SIZE, -CUBE_SIZE),
        make_vec3(-CUBE_SIZE,  CUBE_SIZE, -CUBE_SIZE),
        make_vec3(-CUBE_SIZE, -CUBE_SIZE,  CUBE_SIZE),
        make_vec3( CUBE_SIZE, -CUBE_SIZE,  CUBE_SIZE),
        make_vec3( CUBE_SIZE,  CUBE_SIZE,  CUBE_SIZE),
        make_vec3(-CUBE_SIZE,  CUBE_SIZE,  CUBE_SIZE)
    };

    int edges[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    int edgeIdx = blockIdx.x;
    if (edgeIdx >= 12) return;

    Vec3 v0 = verts[edges[edgeIdx][0]];
    Vec3 v1 = verts[edges[edgeIdx][1]];

    v0 = rotateY(v0, rotY);
    v0 = rotateX(v0, rotX);
    v1 = rotateY(v1, rotY);
    v1 = rotateX(v1, rotX);

    int x0, y0, x1, y1;
    float d0, d1;
    project(v0, camDist, &x0, &y0, &d0);
    project(v1, camDist, &x1, &y1, &d1);

    float avgDepth = (d0 + d1) / 2.0f;
    float alpha = 150.0f + 100.0f / avgDepth;
    if (alpha > 255) alpha = 255;

    drawLine(pixels, x0, y0, x1, y1, r, g, b, (unsigned char)alpha);
}

// Ball physics structure
struct Ball {
    Vec3 pos;
    Vec3 vel;
    Vec3 color;
};

int main() {
    printf("=== Windows CUDA 3D Demo ===\n");
    printf("Bouncing Ball in Spinning Cube\n\n");
    printf("Controls:\n");
    printf("  Arrow keys  - Rotate view\n");
    printf("  +/-         - Zoom in/out\n");
    printf("  Space       - Launch ball randomly\n");
    printf("  R           - Reset\n");
    printf("  Q/Escape    - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA 3D - Bouncing Ball in Cube", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate memory
    unsigned char* h_pixels;
    unsigned char* d_pixels;

    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    // Grid configs
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // Scene parameters
    float rotX = 0.3f;
    float rotY = 0.0f;
    float camDist = 6.0f;
    float autoRotSpeed = 0.5f;

    // Ball
    Ball ball;
    ball.pos = make_vec3(0, 0, 0);
    ball.vel = make_vec3(1.5f, 2.0f, 1.0f);
    ball.color = make_vec3(1.0f, 0.3f, 0.1f);

    float gravity = -4.0f;
    float bounce = 0.85f;
    float wallLimit = CUBE_SIZE - BALL_RADIUS;

    double lastTime = win32_get_time(display);
    double lastFpsTime = lastTime;
    int frameCount = 0;

    srand((unsigned int)time(NULL));

    while (!win32_should_close(display)) {
        // Process Windows messages
        if (win32_process_events(display)) break;

        // Handle events
        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                if (event.key == XK_Escape || event.key == XK_q) goto cleanup;
                if (event.key == XK_Left) rotY -= 0.2f;
                if (event.key == XK_Right) rotY += 0.2f;
                if (event.key == XK_Up) rotX -= 0.1f;
                if (event.key == XK_Down) rotX += 0.1f;
                if (event.key == XK_plus || event.key == XK_equal) camDist -= 0.5f;
                if (event.key == XK_minus) camDist += 0.5f;
                if (camDist < 3.0f) camDist = 3.0f;
                if (camDist > 15.0f) camDist = 15.0f;

                if (event.key == XK_space) {
                    ball.vel.x = ((rand() % 100) / 50.0f - 1.0f) * 3.0f;
                    ball.vel.y = ((rand() % 100) / 100.0f) * 4.0f + 2.0f;
                    ball.vel.z = ((rand() % 100) / 50.0f - 1.0f) * 3.0f;
                    ball.color.x = 0.3f + (rand() % 70) / 100.0f;
                    ball.color.y = 0.3f + (rand() % 70) / 100.0f;
                    ball.color.z = 0.3f + (rand() % 70) / 100.0f;
                }

                if (event.key == XK_r) {
                    ball.pos = make_vec3(0, 0, 0);
                    ball.vel = make_vec3(1.5f, 2.0f, 1.0f);
                    rotX = 0.3f;
                    rotY = 0.0f;
                    camDist = 6.0f;
                }
            }
        }

        double now = win32_get_time(display);
        float dt = (float)(now - lastTime);
        if (dt > 0.05f) dt = 0.05f;
        lastTime = now;

        // Auto-rotate
        rotY += autoRotSpeed * dt;

        // Update ball physics
        ball.vel.y += gravity * dt;
        ball.pos = vec3_add(ball.pos, vec3_mul(ball.vel, dt));

        // Bounce off cube walls
        if (ball.pos.x < -wallLimit) { ball.pos.x = -wallLimit; ball.vel.x = -ball.vel.x * bounce; }
        if (ball.pos.x > wallLimit) { ball.pos.x = wallLimit; ball.vel.x = -ball.vel.x * bounce; }
        if (ball.pos.y < -wallLimit) { ball.pos.y = -wallLimit; ball.vel.y = -ball.vel.y * bounce; }
        if (ball.pos.y > wallLimit) { ball.pos.y = wallLimit; ball.vel.y = -ball.vel.y * bounce; }
        if (ball.pos.z < -wallLimit) { ball.pos.z = -wallLimit; ball.vel.z = -ball.vel.z * bounce; }
        if (ball.pos.z > wallLimit) { ball.pos.z = wallLimit; ball.vel.z = -ball.vel.z * bounce; }

        // Clear
        clearKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT);

        // Transform ball position for rendering
        Vec3 ballTransformed = rotateY(ball.pos, rotY);
        ballTransformed = rotateX(ballTransformed, rotX);

        // Project ball to screen
        int ballSX, ballSY;
        float ballDepth;
        project(ballTransformed, camDist, &ballSX, &ballSY, &ballDepth);
        float ballScreenRadius = 300.0f * BALL_RADIUS / ballDepth;

        // Draw shadow
        Vec3 shadowPos = make_vec3(ball.pos.x, -CUBE_SIZE, ball.pos.z);
        shadowPos = rotateY(shadowPos, rotY);
        shadowPos = rotateX(shadowPos, rotX);
        int shadowSX, shadowSY;
        float shadowDepth;
        project(shadowPos, camDist, &shadowSX, &shadowSY, &shadowDepth);
        float shadowRadius = ballScreenRadius * (1.0f + (ball.pos.y + CUBE_SIZE) * 0.3f);

        drawShadowKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT,
            shadowSX, shadowSY, shadowRadius);

        // Draw cube (wireframe)
        drawCubeEdgesKernel<<<12, 1>>>(d_pixels, rotX, rotY, camDist, 100, 200, 255);

        // Draw ball
        drawBallKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT,
            ballSX, ballSY, ballScreenRadius, ballDepth, ball.color, rotY);

        cudaDeviceSynchronize();

        // Copy and display
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f | Ball pos: (%.2f, %.2f, %.2f)\n",
                   frameCount / (now - lastFpsTime),
                   ball.pos.x, ball.pos.y, ball.pos.z);
            frameCount = 0;
            lastFpsTime = now;
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);
    cudaFree(d_pixels);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
