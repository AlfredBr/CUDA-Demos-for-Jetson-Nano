/*
 * CUDA Cornell Box Path Tracer for Windows
 *
 * Progressive path tracer with:
 *   - Möller-Trumbore ray-triangle intersection
 *   - Area light sampling
 *   - Cosine-weighted hemisphere sampling for diffuse
 *   - Accumulation buffer for progressive refinement
 *   - Classic Cornell box scene with color bleeding
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "win32_display.h"

#define WIDTH 512
#define HEIGHT 512
#define MAX_TRIANGLES 64
#define MAX_BOUNCES 6

// ============================================================================
// VECTOR MATH
// ============================================================================

struct Vec3 {
    float x, y, z;
    __device__ __host__ Vec3() : x(0), y(0), z(0) {}
    __device__ __host__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

__device__ __host__ Vec3 operator+(Vec3 a, Vec3 b) { return Vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __host__ Vec3 operator-(Vec3 a, Vec3 b) { return Vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __host__ Vec3 operator*(Vec3 a, float t) { return Vec3(a.x*t, a.y*t, a.z*t); }
__device__ __host__ Vec3 operator*(float t, Vec3 a) { return Vec3(a.x*t, a.y*t, a.z*t); }
__device__ __host__ Vec3 operator*(Vec3 a, Vec3 b) { return Vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ __host__ Vec3 operator/(Vec3 a, float t) { return Vec3(a.x/t, a.y/t, a.z/t); }
__device__ __host__ float dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __host__ Vec3 cross(Vec3 a, Vec3 b) {
    return Vec3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ __host__ float length(Vec3 v) { return sqrtf(dot(v,v)); }
__device__ __host__ Vec3 normalize(Vec3 v) {
    float len = length(v);
    return len > 0 ? v/len : Vec3(0,0,0);
}

// ============================================================================
// DATA STRUCTURES
// ============================================================================

struct Ray { Vec3 origin, dir; };

struct Material {
    float albedo[3];
    float emission[3];
};

struct Triangle {
    float v0[3], v1[3], v2[3];
    float normal[3];
    int materialId;
};

struct HitRecord {
    float t;
    Vec3 point, normal;
    int materialId;
    bool frontFace;
};

// Constant memory
__constant__ Triangle d_triangles[MAX_TRIANGLES];
__constant__ Material d_materials[8];
__constant__ int d_numTriangles;
__constant__ float d_lightCorner[3];
__constant__ float d_lightU[3];
__constant__ float d_lightV[3];
__constant__ float d_lightNormal[3];
__constant__ float d_lightArea;

// Helper to convert float[3] to Vec3
__device__ Vec3 toVec3(const float* f) { return Vec3(f[0], f[1], f[2]); }

// ============================================================================
// RAY-TRIANGLE INTERSECTION
// ============================================================================

__device__ bool intersectTriangle(const Ray& ray, const Triangle& tri,
                                   float tMin, float tMax, HitRecord& rec) {
    Vec3 v0 = toVec3(tri.v0), v1 = toVec3(tri.v1), v2 = toVec3(tri.v2);
    Vec3 edge1 = v1 - v0, edge2 = v2 - v0;
    Vec3 h = cross(ray.dir, edge2);
    float a = dot(edge1, h);

    if (fabsf(a) < 1e-8f) return false;

    float f = 1.0f / a;
    Vec3 s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    Vec3 q = cross(s, edge1);
    float v = f * dot(ray.dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = f * dot(edge2, q);
    if (t < tMin || t > tMax) return false;

    rec.t = t;
    rec.point = ray.origin + t * ray.dir;
    rec.normal = toVec3(tri.normal);
    rec.materialId = tri.materialId;
    rec.frontFace = dot(ray.dir, rec.normal) < 0;
    if (!rec.frontFace) rec.normal = rec.normal * -1.0f;

    return true;
}

__device__ bool intersectScene(const Ray& ray, float tMin, float tMax, HitRecord& rec) {
    bool hit = false;
    float closest = tMax;

    for (int i = 0; i < d_numTriangles; i++) {
        HitRecord temp;
        if (intersectTriangle(ray, d_triangles[i], tMin, closest, temp)) {
            hit = true;
            closest = temp.t;
            rec = temp;
        }
    }
    return hit;
}

// ============================================================================
// SAMPLING
// ============================================================================

__device__ Vec3 sampleHemisphereCosine(Vec3 normal, curandState* rng) {
    float u1 = curand_uniform(rng), u2 = curand_uniform(rng);
    float r = sqrtf(u1), theta = 6.28318530718f * u2;
    float x = r * cosf(theta), y = r * sinf(theta), z = sqrtf(1.0f - u1);

    Vec3 w = normal;
    Vec3 a = (fabsf(w.x) > 0.9f) ? Vec3(0,1,0) : Vec3(1,0,0);
    Vec3 u = normalize(cross(a, w));
    Vec3 v = cross(w, u);

    return normalize(u*x + v*y + w*z);
}

__device__ Vec3 sampleLight(curandState* rng) {
    float u = curand_uniform(rng), v = curand_uniform(rng);
    return toVec3(d_lightCorner) + u * toVec3(d_lightU) + v * toVec3(d_lightV);
}

// ============================================================================
// PATH TRACING
// ============================================================================

__device__ Vec3 tracePath(Ray ray, curandState* rng, int maxBounces) {
    Vec3 throughput(1,1,1), radiance(0,0,0);

    for (int bounce = 0; bounce <= maxBounces; bounce++) {
        HitRecord rec;

        if (!intersectScene(ray, 0.001f, 1e10f, rec)) {
            radiance = radiance + throughput * Vec3(0.01f, 0.01f, 0.02f);
            break;
        }

        Material mat = d_materials[rec.materialId];
        Vec3 albedo = toVec3(mat.albedo);
        Vec3 emission = toVec3(mat.emission);

        radiance = radiance + throughput * emission;

        if (emission.x > 0 || emission.y > 0 || emission.z > 0) break;

        // Russian roulette
        if (bounce > 2) {
            float p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
            if (curand_uniform(rng) > p) break;
            throughput = throughput / p;
        }

        // Direct lighting
        Vec3 lightPoint = sampleLight(rng);
        Vec3 toLight = lightPoint - rec.point;
        float lightDist = length(toLight);
        Vec3 lightDir = toLight / lightDist;

        float NdotL = dot(rec.normal, lightDir);
        if (NdotL > 0) {
            Ray shadowRay;
            shadowRay.origin = rec.point + rec.normal * 0.001f;
            shadowRay.dir = lightDir;

            HitRecord shadowRec;
            if (!intersectScene(shadowRay, 0.001f, lightDist - 0.001f, shadowRec)) {
                Vec3 ln = toVec3(d_lightNormal);
                float lightNdotL = fmaxf(0.0f, dot(ln * -1.0f, lightDir));
                if (lightNdotL > 0) {
                    float pdf = (lightDist * lightDist) / (d_lightArea * lightNdotL);
                    Vec3 lightE(15.0f, 15.0f, 15.0f);
                    Vec3 brdf = albedo / 3.14159265f;
                    radiance = radiance + throughput * brdf * lightE * NdotL / pdf;
                }
            }
        }

        Vec3 newDir = sampleHemisphereCosine(rec.normal, rng);
        throughput = throughput * albedo;
        ray.origin = rec.point + rec.normal * 0.001f;
        ray.dir = newDir;
    }

    return radiance;
}

// ============================================================================
// KERNELS
// ============================================================================

__global__ void renderKernel(float3* accumBuffer, unsigned int* sampleCount,
                              int width, int height, int maxBounces, unsigned int frameNum) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    curandState rng;
    curand_init(frameNum * width * height + idx, 0, 0, &rng);

    float fov = 0.698132f; // 40 degrees
    Vec3 camPos(278, 273, -800);
    Vec3 camDir = normalize(Vec3(0, 0, 1));
    Vec3 camRight = Vec3(1, 0, 0);
    Vec3 camUp = Vec3(0, 1, 0);

    float halfH = tanf(fov / 2.0f);
    float halfW = halfH; // Square aspect

    float u = ((float)x + curand_uniform(&rng)) / width;
    float v = ((float)y + curand_uniform(&rng)) / height;
    u = (2.0f * u - 1.0f) * halfW;
    v = (2.0f * v - 1.0f) * halfH;

    Ray ray;
    ray.origin = camPos;
    ray.dir = normalize(camDir + u * camRight + v * camUp);

    Vec3 color = tracePath(ray, &rng, maxBounces);

    float3 prev = accumBuffer[idx];
    accumBuffer[idx] = make_float3(prev.x + color.x, prev.y + color.y, prev.z + color.z);
    sampleCount[idx]++;
}

__global__ void tonemapKernel(unsigned char* pixels, const float3* accumBuffer,
                               const unsigned int* sampleCount, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int displayIdx = ((height - 1 - y) * width + x) * 4;

    float3 accum = accumBuffer[idx];
    unsigned int samples = sampleCount[idx];
    if (samples == 0) samples = 1;

    float r = accum.x / samples;
    float g = accum.y / samples;
    float b = accum.z / samples;

    // Reinhard + gamma
    r = powf(r / (1.0f + r), 0.4545f);
    g = powf(g / (1.0f + g), 0.4545f);
    b = powf(b / (1.0f + b), 0.4545f);

    pixels[displayIdx + 0] = (unsigned char)(fminf(b, 1.0f) * 255);
    pixels[displayIdx + 1] = (unsigned char)(fminf(g, 1.0f) * 255);
    pixels[displayIdx + 2] = (unsigned char)(fminf(r, 1.0f) * 255);
    pixels[displayIdx + 3] = 255;
}

// ============================================================================
// SCENE SETUP
// ============================================================================

void setVec3(float* dest, float x, float y, float z) {
    dest[0] = x; dest[1] = y; dest[2] = z;
}

void addQuad(Triangle* tris, int& n, float* v0, float* v1, float* v2, float* v3,
             float* normal, int mat) {
    memcpy(tris[n].v0, v0, 12); memcpy(tris[n].v1, v1, 12); memcpy(tris[n].v2, v2, 12);
    memcpy(tris[n].normal, normal, 12); tris[n].materialId = mat; n++;
    memcpy(tris[n].v0, v0, 12); memcpy(tris[n].v1, v2, 12); memcpy(tris[n].v2, v3, 12);
    memcpy(tris[n].normal, normal, 12); tris[n].materialId = mat; n++;
}

void buildCornellBox(Triangle* tris, int& n, Material* mats) {
    // Materials: 0=white, 1=red, 2=green, 3=light
    setVec3(mats[0].albedo, 0.73f, 0.73f, 0.73f); setVec3(mats[0].emission, 0, 0, 0);
    setVec3(mats[1].albedo, 0.65f, 0.05f, 0.05f); setVec3(mats[1].emission, 0, 0, 0);
    setVec3(mats[2].albedo, 0.12f, 0.45f, 0.15f); setVec3(mats[2].emission, 0, 0, 0);
    setVec3(mats[3].albedo, 0, 0, 0);             setVec3(mats[3].emission, 15, 15, 15);

    n = 0;
    float sz = 555.0f;
    float v[8][3], norm[3];

    // Floor
    setVec3(v[0], 0,0,0); setVec3(v[1], sz,0,0); setVec3(v[2], sz,0,sz); setVec3(v[3], 0,0,sz);
    setVec3(norm, 0,1,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Ceiling
    setVec3(v[0], 0,sz,0); setVec3(v[1], 0,sz,sz); setVec3(v[2], sz,sz,sz); setVec3(v[3], sz,sz,0);
    setVec3(norm, 0,-1,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Back wall
    setVec3(v[0], 0,0,sz); setVec3(v[1], sz,0,sz); setVec3(v[2], sz,sz,sz); setVec3(v[3], 0,sz,sz);
    setVec3(norm, 0,0,-1);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Left wall (red)
    setVec3(v[0], 0,0,0); setVec3(v[1], 0,0,sz); setVec3(v[2], 0,sz,sz); setVec3(v[3], 0,sz,0);
    setVec3(norm, 1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 1);

    // Right wall (green)
    setVec3(v[0], sz,0,0); setVec3(v[1], sz,sz,0); setVec3(v[2], sz,sz,sz); setVec3(v[3], sz,0,sz);
    setVec3(norm, -1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 2);

    // Area light on ceiling
    float ls = 130.0f, ly = sz - 1.0f;
    float lmin = (sz - ls) / 2.0f, lmax = (sz + ls) / 2.0f;
    setVec3(v[0], lmin,ly,lmin); setVec3(v[1], lmax,ly,lmin);
    setVec3(v[2], lmax,ly,lmax); setVec3(v[3], lmin,ly,lmax);
    setVec3(norm, 0,-1,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 3);

    // Short box (center: 185, 82.5, 169)
    float bh = 165.0f, bx = 82.5f, bcx = 185, bcz = 169;

    // Top
    setVec3(v[0], bcx-bx,bh,bcz-bx); setVec3(v[1], bcx+bx,bh,bcz-bx);
    setVec3(v[2], bcx+bx,bh,bcz+bx); setVec3(v[3], bcx-bx,bh,bcz+bx);
    setVec3(norm, 0,1,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Front
    setVec3(v[0], bcx-bx,0,bcz-bx); setVec3(v[1], bcx+bx,0,bcz-bx);
    setVec3(v[2], bcx+bx,bh,bcz-bx); setVec3(v[3], bcx-bx,bh,bcz-bx);
    setVec3(norm, 0,0,-1);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Back
    setVec3(v[0], bcx+bx,0,bcz+bx); setVec3(v[1], bcx-bx,0,bcz+bx);
    setVec3(v[2], bcx-bx,bh,bcz+bx); setVec3(v[3], bcx+bx,bh,bcz+bx);
    setVec3(norm, 0,0,1);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Left
    setVec3(v[0], bcx-bx,0,bcz+bx); setVec3(v[1], bcx-bx,0,bcz-bx);
    setVec3(v[2], bcx-bx,bh,bcz-bx); setVec3(v[3], bcx-bx,bh,bcz+bx);
    setVec3(norm, -1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Right
    setVec3(v[0], bcx+bx,0,bcz-bx); setVec3(v[1], bcx+bx,0,bcz+bx);
    setVec3(v[2], bcx+bx,bh,bcz+bx); setVec3(v[3], bcx+bx,bh,bcz-bx);
    setVec3(norm, 1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Tall box (center: 368, 165, 351)
    float th = 330.0f, tx = 82.5f, tcx = 368, tcz = 351;

    // Top
    setVec3(v[0], tcx-tx,th,tcz-tx); setVec3(v[1], tcx+tx,th,tcz-tx);
    setVec3(v[2], tcx+tx,th,tcz+tx); setVec3(v[3], tcx-tx,th,tcz+tx);
    setVec3(norm, 0,1,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Front
    setVec3(v[0], tcx-tx,0,tcz-tx); setVec3(v[1], tcx+tx,0,tcz-tx);
    setVec3(v[2], tcx+tx,th,tcz-tx); setVec3(v[3], tcx-tx,th,tcz-tx);
    setVec3(norm, 0,0,-1);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Back
    setVec3(v[0], tcx+tx,0,tcz+tx); setVec3(v[1], tcx-tx,0,tcz+tx);
    setVec3(v[2], tcx-tx,th,tcz+tx); setVec3(v[3], tcx+tx,th,tcz+tx);
    setVec3(norm, 0,0,1);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Left
    setVec3(v[0], tcx-tx,0,tcz+tx); setVec3(v[1], tcx-tx,0,tcz-tx);
    setVec3(v[2], tcx-tx,th,tcz-tx); setVec3(v[3], tcx-tx,th,tcz+tx);
    setVec3(norm, -1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    // Right
    setVec3(v[0], tcx+tx,0,tcz-tx); setVec3(v[1], tcx+tx,0,tcz+tx);
    setVec3(v[2], tcx+tx,th,tcz+tx); setVec3(v[3], tcx+tx,th,tcz-tx);
    setVec3(norm, 1,0,0);
    addQuad(tris, n, v[0], v[1], v[2], v[3], norm, 0);

    printf("Scene: %d triangles\n", n);
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    printf("=== CUDA Cornell Box Path Tracer ===\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);

    Triangle h_tris[MAX_TRIANGLES];
    Material h_mats[8];
    int numTris = 0;

    buildCornellBox(h_tris, numTris, h_mats);

    cudaMemcpyToSymbol(d_triangles, h_tris, numTris * sizeof(Triangle));
    cudaMemcpyToSymbol(d_materials, h_mats, 8 * sizeof(Material));
    cudaMemcpyToSymbol(d_numTriangles, &numTris, sizeof(int));

    // Light parameters
    float sz = 555.0f, ls = 130.0f, ly = sz - 1.0f;
    float lightCorner[3] = {(sz-ls)/2.0f, ly, (sz-ls)/2.0f};
    float lightU[3] = {ls, 0, 0};
    float lightV[3] = {0, 0, ls};
    float lightNormal[3] = {0, -1, 0};
    float lightArea = ls * ls;

    cudaMemcpyToSymbol(d_lightCorner, lightCorner, sizeof(lightCorner));
    cudaMemcpyToSymbol(d_lightU, lightU, sizeof(lightU));
    cudaMemcpyToSymbol(d_lightV, lightV, sizeof(lightV));
    cudaMemcpyToSymbol(d_lightNormal, lightNormal, sizeof(lightNormal));
    cudaMemcpyToSymbol(d_lightArea, &lightArea, sizeof(float));

    // Win32 setup
    Win32Display* display = win32_create_window("CUDA Cornell Box Path Tracer", WIDTH, HEIGHT);
    if (!display) { printf("Cannot create window\n"); return 1; }

    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);

    unsigned char* d_pixels;
    float3* d_accumBuffer;
    unsigned int* d_sampleCount;

    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_accumBuffer, WIDTH * HEIGHT * sizeof(float3));
    cudaMalloc(&d_sampleCount, WIDTH * HEIGHT * sizeof(unsigned int));

    cudaMemset(d_accumBuffer, 0, WIDTH * HEIGHT * sizeof(float3));
    cudaMemset(d_sampleCount, 0, WIDTH * HEIGHT * sizeof(unsigned int));

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    int maxBounces = 4;
    bool paused = false;
    unsigned int frameNum = 0, totalSamples = 0;

    double startTime = win32_get_time(display), lastFpsTime = startTime;
    int fpsFrameCount = 0;

    printf("Controls:\n");
    printf("  R     - Reset accumulation\n");
    printf("  +/-   - Adjust bounces (%d)\n", maxBounces);
    printf("  Space - Pause/resume\n");
    printf("  S     - Save image\n");
    printf("  Q     - Quit\n\n");
    printf("Rendering...\n");

    while (!win32_should_close(display)) {
        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;

                if (key == XK_Escape || key == XK_q) goto cleanup;

                if (key == XK_r) {
                    cudaMemset(d_accumBuffer, 0, WIDTH * HEIGHT * sizeof(float3));
                    cudaMemset(d_sampleCount, 0, WIDTH * HEIGHT * sizeof(unsigned int));
                    frameNum = totalSamples = 0;
                    startTime = win32_get_time(display);
                    printf("Reset\n");
                }

                if (key == XK_space) { paused = !paused; printf("%s\n", paused ? "Paused" : "Resumed"); }
                if (key == XK_plus || key == XK_equal) { maxBounces = (maxBounces < 6) ? maxBounces+1 : 6; printf("Bounces: %d\n", maxBounces); }
                if (key == XK_minus) { maxBounces = (maxBounces > 1) ? maxBounces-1 : 1; printf("Bounces: %d\n", maxBounces); }

                if (key == XK_s) {
                    cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
                    FILE* f = fopen("cornell.ppm", "wb");
                    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
                    for (int i = 0; i < WIDTH * HEIGHT; i++) {
                        fputc(h_pixels[i*4+2], f);
                        fputc(h_pixels[i*4+1], f);
                        fputc(h_pixels[i*4+0], f);
                    }
                    fclose(f);
                    printf("Saved cornell.ppm (%d spp)\n", totalSamples);
                }
            }
        }

        if (!paused) {
            renderKernel<<<gridSize, blockSize>>>(d_accumBuffer, d_sampleCount,
                                                   WIDTH, HEIGHT, maxBounces, frameNum);
            frameNum++;
            totalSamples++;
        }

        tonemapKernel<<<gridSize, blockSize>>>(d_pixels, d_accumBuffer, d_sampleCount, WIDTH, HEIGHT);

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        fpsFrameCount++;
        double currentTime = win32_get_time(display);
        if (currentTime - lastFpsTime >= 0.5) {
            float fps = fpsFrameCount / (float)(currentTime - lastFpsTime);
            float sps = totalSamples / (float)(currentTime - startTime);

            char title[256];
            snprintf(title, sizeof(title), "Cornell Box | %d spp | %.1f sps | %.1f FPS%s",
                totalSamples, sps, fps, paused ? " [PAUSED]" : "");
            SetWindowTextA(display->hwnd, title);

            fpsFrameCount = 0;
            lastFpsTime = currentTime;
        }
    }

cleanup:
    cudaFree(d_pixels);
    cudaFree(d_accumBuffer);
    cudaFree(d_sampleCount);

    free(h_pixels);
    win32_destroy_window(display);

    printf("\nFinal: %d samples\n", totalSamples);
    return 0;
}
