/*
 * Windows CUDA Particle System Demo
 * Ported from Jetson Nano version
 *
 * Simulates thousands of particles with gravity, bouncing, and trails
 * All physics computed in parallel on the GPU!
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600
#define NUM_PARTICLES 50000
#define TRAIL_FADE 0.92f

struct Particle {
    float x, y;
    float vx, vy;
    float r, g, b;
    float life;
};

// Initialize random states for each particle
__global__ void initRandom(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < NUM_PARTICLES) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Spawn a particle at emitter position
__device__ void spawnParticle(Particle* p, curandState* state, float emitterX, float emitterY) {
    p->x = emitterX;
    p->y = emitterY;

    // Random velocity in a cone shape upward
    float angle = (curand_uniform(state) - 0.5f) * 2.0f;
    float speed = 100.0f + curand_uniform(state) * 200.0f;

    p->vx = angle * speed;
    p->vy = -speed * (0.5f + curand_uniform(state) * 0.5f);

    // Random warm color (fire-like)
    p->r = 0.8f + curand_uniform(state) * 0.2f;
    p->g = 0.2f + curand_uniform(state) * 0.6f;
    p->b = curand_uniform(state) * 0.3f;

    p->life = 0.5f + curand_uniform(state) * 2.0f;
}

// Update particle physics
__global__ void updateParticles(Particle* particles, curandState* states,
                                 float dt, float emitterX, float emitterY, float time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle* p = &particles[idx];
    curandState* state = &states[idx];

    // Respawn dead particles
    if (p->life <= 0) {
        spawnParticle(p, state, emitterX, emitterY);
        return;
    }

    // Gravity
    p->vy += 200.0f * dt;

    // Wind effect (sinusoidal)
    p->vx += sinf(time * 2.0f + p->y * 0.01f) * 50.0f * dt;

    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;

    // Bounce off walls
    if (p->x < 0) { p->x = 0; p->vx *= -0.6f; }
    if (p->x >= WIDTH) { p->x = WIDTH - 1; p->vx *= -0.6f; }
    if (p->y < 0) { p->y = 0; p->vy *= -0.6f; }
    if (p->y >= HEIGHT) { p->y = HEIGHT - 1; p->vy *= -0.8f; p->vx *= 0.95f; }

    // Age the particle
    p->life -= dt;

    // Fade color as particle ages
    float fade = p->life / 2.5f;
    if (fade > 1.0f) fade = 1.0f;
    p->r *= (0.99f + fade * 0.01f);
    p->g *= (0.97f + fade * 0.03f);
}

// Render particles to framebuffer
__global__ void renderParticles(unsigned char* pixels, Particle* particles, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_PARTICLES) return;

    Particle* p = &particles[idx];

    if (p->life <= 0) return;

    int px = (int)p->x;
    int py = (int)p->y;

    if (px < 0 || px >= width || py < 0 || py >= height) return;

    int pidx = (py * width + px) * 4;

    // Additive blending for glowing effect
    float intensity = p->life / 2.5f;
    if (intensity > 1.0f) intensity = 1.0f;

    int r = pixels[pidx + 2] + (int)(p->r * intensity * 100);
    int g = pixels[pidx + 1] + (int)(p->g * intensity * 100);
    int b = pixels[pidx + 0] + (int)(p->b * intensity * 100);

    pixels[pidx + 2] = (r > 255) ? 255 : r;
    pixels[pidx + 1] = (g > 255) ? 255 : g;
    pixels[pidx + 0] = (b > 255) ? 255 : b;
    pixels[pidx + 3] = 255;
}

// Fade/clear the framebuffer (creates trails)
__global__ void fadeFramebuffer(unsigned char* pixels, int width, int height, float fade) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    pixels[idx + 0] = (unsigned char)(pixels[idx + 0] * fade);
    pixels[idx + 1] = (unsigned char)(pixels[idx + 1] * fade);
    pixels[idx + 2] = (unsigned char)(pixels[idx + 2] * fade);
}

int main() {
    printf("=== Windows CUDA Particle System ===\n");
    printf("Particles: %d\n", NUM_PARTICLES);
    printf("Press Q or Escape to exit\n");
    printf("Move mouse to control emitter!\n\n");

    // CUDA device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA Particle System - Move Mouse!", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate memory
    unsigned char* h_pixels;
    unsigned char* d_pixels;
    Particle* d_particles;
    curandState* d_states;

    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    cudaMalloc(&d_states, NUM_PARTICLES * sizeof(curandState));

    // Clear framebuffer
    cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);

    // Initialize random states
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    initRandom<<<numBlocks, blockSize>>>(d_states, (unsigned long)time(NULL));

    // Initialize particles as dead (will respawn)
    Particle* h_particles = (Particle*)malloc(NUM_PARTICLES * sizeof(Particle));
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_particles[i].life = -1.0f;
    }
    cudaMemcpy(d_particles, h_particles, NUM_PARTICLES * sizeof(Particle), cudaMemcpyHostToDevice);
    free(h_particles);

    // Grid config for fade kernel
    dim3 fadeBlock(16, 16);
    dim3 fadeGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    double startTime = win32_get_time(display);
    double lastTime = startTime;
    int frameCount = 0;
    double lastFpsTime = startTime;

    float mouseX = WIDTH / 2.0f;
    float mouseY = HEIGHT / 2.0f;

    // Main loop
    while (!win32_should_close(display)) {
        // Process Windows messages
        if (win32_process_events(display)) break;

        // Handle events
        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                if (event.key == XK_Escape || event.key == XK_q) goto cleanup;
            }
            if (event.type == WIN32_EVENT_MOUSE_MOVE) {
                mouseX = (float)event.mouseX;
                mouseY = (float)event.mouseY;
            }
        }

        double now = win32_get_time(display);
        float dt = (float)(now - lastTime);
        if (dt > 0.05f) dt = 0.05f;
        lastTime = now;
        float time = (float)(now - startTime);

        // Fade the framebuffer (creates trails)
        fadeFramebuffer<<<fadeGrid, fadeBlock>>>(d_pixels, WIDTH, HEIGHT, TRAIL_FADE);

        // Update particles on GPU
        updateParticles<<<numBlocks, blockSize>>>(d_particles, d_states, dt, mouseX, mouseY, time);

        // Render particles on GPU
        renderParticles<<<numBlocks, blockSize>>>(d_pixels, d_particles, WIDTH, HEIGHT);

        // Copy to host
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);

        // Display
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f | Particles: %d | Mouse: (%.0f, %.0f)\n",
                   frameCount / (now - lastFpsTime), NUM_PARTICLES, mouseX, mouseY);
            frameCount = 0;
            lastFpsTime = now;
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);
    cudaFree(d_pixels);
    cudaFree(d_particles);
    cudaFree(d_states);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
