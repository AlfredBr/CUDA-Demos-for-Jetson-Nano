/*
 * Windows CUDA Ray Marcher - Procedural 3D Scene with SDFs
 * Ported from Jetson Nano version
 *
 * Signed Distance Field ray marching with:
 *   - Multiple primitives (sphere, box, torus, cylinder, plane)
 *   - CSG operations (union, intersection, subtraction, smooth blend)
 *   - Soft shadows and ambient occlusion
 *   - Phong lighting with specular highlights
 *   - Animated scene elements
 *   - Interactive camera
 *
 * Controls:
 *   Arrow keys   - Orbit camera
 *   W/S          - Move camera forward/back
 *   +/-          - Adjust FOV
 *   1-4          - Switch scenes
 *   Space        - Toggle animation
 *   P            - Toggle soft shadows
 *   O            - Toggle ambient occlusion
 *   R            - Reset camera
 *   Escape       - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600

#define MAX_STEPS 100
#define MAX_DIST 50.0f
#define SURF_DIST 0.001f
#define EPSILON 0.001f

// ============== VECTOR MATH ==============
// Use CUDA's built-in float3 with helper functions

__device__ __host__ inline float3 f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__device__ __host__ inline float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 sub3(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 mul3(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __host__ inline float3 mul3v(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ inline float3 neg3(float3 a) {
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ __host__ inline float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3 cross3(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ __host__ inline float len3(float3 v) {
    return sqrtf(dot3(v, v));
}

__device__ __host__ inline float3 norm3(float3 v) {
    float len = len3(v);
    return (len > 0.0001f) ? mul3(v, 1.0f / len) : v;
}

__device__ __host__ inline float3 reflect3(float3 I, float3 N) {
    return sub3(I, mul3(N, 2.0f * dot3(N, I)));
}

__device__ __host__ inline float3 abs3(float3 v) {
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__device__ __host__ inline float3 max3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __host__ inline float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __host__ inline float3 clamp3(float3 v, float lo, float hi) {
    return make_float3(clampf(v.x, lo, hi), clampf(v.y, lo, hi), clampf(v.z, lo, hi));
}

__device__ __host__ inline float mixf(float a, float b, float t) {
    return a * (1.0f - t) + b * t;
}

__device__ __host__ inline float3 mix3(float3 a, float3 b, float t) {
    return add3(mul3(a, 1.0f - t), mul3(b, t));
}

// ============== SDF PRIMITIVES ==============

__device__ float sdSphere(float3 p, float r) {
    return len3(p) - r;
}

__device__ float sdBox(float3 p, float3 b) {
    float3 q = sub3(abs3(p), b);
    float3 qmax = max3(q, f3(0, 0, 0));
    return len3(qmax) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f);
}

__device__ float sdRoundBox(float3 p, float3 b, float r) {
    float3 q = sub3(abs3(p), b);
    float3 qmax = max3(q, f3(0, 0, 0));
    return len3(qmax) + fminf(fmaxf(q.x, fmaxf(q.y, q.z)), 0.0f) - r;
}

__device__ float sdTorus(float3 p, float R, float r) {
    float qx = sqrtf(p.x * p.x + p.z * p.z) - R;
    return sqrtf(qx * qx + p.y * p.y) - r;
}

__device__ float sdCappedCylinder(float3 p, float h, float r) {
    float dx = sqrtf(p.x * p.x + p.z * p.z) - r;
    float dy = fabsf(p.y) - h;
    return fminf(fmaxf(dx, dy), 0.0f) + sqrtf(fmaxf(dx, 0.0f) * fmaxf(dx, 0.0f) +
                                               fmaxf(dy, 0.0f) * fmaxf(dy, 0.0f));
}

__device__ float sdPlane(float3 p, float h) {
    return p.y - h;
}

__device__ float sdOctahedron(float3 p, float s) {
    float3 ap = abs3(p);
    return (ap.x + ap.y + ap.z - s) * 0.57735027f;
}

// ============== CSG OPERATIONS ==============

__device__ float opUnion(float d1, float d2) {
    return fminf(d1, d2);
}

__device__ float opSubtract(float d1, float d2) {
    return fmaxf(-d1, d2);
}

__device__ float opIntersect(float d1, float d2) {
    return fmaxf(d1, d2);
}

__device__ float opSmoothUnion(float d1, float d2, float k) {
    float h = clampf(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return mixf(d2, d1, h) - k * h * (1.0f - h);
}

// ============== TRANSFORMATIONS ==============

__device__ float3 rotateX(float3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return f3(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
}

__device__ float3 rotateY(float3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return f3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
}

__device__ float3 rotateZ(float3 p, float a) {
    float c = cosf(a), s = sinf(a);
    return f3(c * p.x - s * p.y, s * p.x + c * p.y, p.z);
}

__device__ float3 opRep(float3 p, float3 c) {
    return f3(fmodf(p.x + 0.5f * c.x, c.x) - 0.5f * c.x,
              fmodf(p.y + 0.5f * c.y, c.y) - 0.5f * c.y,
              fmodf(p.z + 0.5f * c.z, c.z) - 0.5f * c.z);
}

// ============== SCENE DEFINITIONS ==============

struct SceneResult {
    float dist;
    int matId;
};

// Scene 1: Classic spheres on plane
__device__ SceneResult scene1(float3 p, float time) {
    SceneResult res;

    float plane = sdPlane(p, -1.0f);

    // Main bouncing sphere
    float3 sp1 = sub3(p, f3(0, 0.5f + 0.3f * sinf(time * 2.0f), 0));
    float sphere1 = sdSphere(sp1, 1.0f);

    // Orbiting spheres
    float a1 = time * 1.5f;
    float3 sp2 = sub3(p, f3(2.5f * cosf(a1), 0.0f, 2.5f * sinf(a1)));
    float sphere2 = sdSphere(sp2, 0.5f);

    float a2 = time * 1.5f + 2.094f;
    float3 sp3 = sub3(p, f3(2.5f * cosf(a2), 0.0f, 2.5f * sinf(a2)));
    float sphere3 = sdSphere(sp3, 0.5f);

    float a3 = time * 1.5f + 4.188f;
    float3 sp4 = sub3(p, f3(2.5f * cosf(a3), 0.0f, 2.5f * sinf(a3)));
    float sphere4 = sdSphere(sp4, 0.5f);

    res.dist = plane;
    res.matId = 0;

    if (sphere1 < res.dist) { res.dist = sphere1; res.matId = 1; }
    if (sphere2 < res.dist) { res.dist = sphere2; res.matId = 2; }
    if (sphere3 < res.dist) { res.dist = sphere3; res.matId = 3; }
    if (sphere4 < res.dist) { res.dist = sphere4; res.matId = 4; }

    return res;
}

// Scene 2: CSG operations
__device__ SceneResult scene2(float3 p, float time) {
    SceneResult res;

    float plane = sdPlane(p, -1.5f);

    // Sphere with box subtraction
    float3 p1 = sub3(p, f3(-2.0f, 0.5f, 0));
    p1 = rotateY(p1, time * 0.5f);
    float carved = opSubtract(sdBox(p1, f3(0.7f, 0.7f, 0.7f)), sdSphere(p1, 1.0f));

    // Smooth union metaballs
    float3 p2 = sub3(p, f3(2.0f, 0.5f, 0));
    float3 off1 = f3(0.5f * sinf(time), 0, 0);
    float3 off2 = f3(-0.5f * sinf(time), 0, 0);
    float3 off3 = f3(0, 0.5f * cosf(time * 1.3f), 0);
    float meta = sdSphere(sub3(p2, off1), 0.6f);
    meta = opSmoothUnion(meta, sdSphere(sub3(p2, off2), 0.6f), 0.3f);
    meta = opSmoothUnion(meta, sdSphere(sub3(p2, off3), 0.5f), 0.3f);

    // Intersection lens
    float3 p3 = sub3(p, f3(0, 0.5f, 2.5f));
    p3 = rotateZ(p3, time * 0.7f);
    float lens = opIntersect(sdSphere(sub3(p3, f3(0.3f, 0, 0)), 1.0f),
                             sdSphere(add3(p3, f3(0.3f, 0, 0)), 1.0f));

    res.dist = plane;
    res.matId = 0;

    if (carved < res.dist) { res.dist = carved; res.matId = 1; }
    if (meta < res.dist) { res.dist = meta; res.matId = 2; }
    if (lens < res.dist) { res.dist = lens; res.matId = 3; }

    return res;
}

// Scene 3: Geometric shapes
__device__ SceneResult scene3(float3 p, float time) {
    SceneResult res;

    float plane = sdPlane(p, -1.5f);

    // Torus
    float3 p1 = rotateX(sub3(p, f3(-2.0f, 0.5f, 0)), time * 0.8f);
    p1 = rotateZ(p1, time * 0.5f);
    float torus = sdTorus(p1, 0.8f, 0.3f);

    // Rounded box
    float3 p2 = rotateY(sub3(p, f3(0, 0.5f, 0)), time * 0.6f);
    p2 = rotateX(p2, time * 0.4f);
    float rbox = sdRoundBox(p2, f3(0.6f, 0.6f, 0.6f), 0.1f);

    // Octahedron
    float3 p3 = rotateY(sub3(p, f3(2.0f, 0.5f, 0)), time);
    float octa = sdOctahedron(p3, 1.0f);

    // Capped cylinder
    float3 p4 = rotateZ(sub3(p, f3(0, 0.5f, -2.5f)), time * 0.7f);
    float cyl = sdCappedCylinder(p4, 0.8f, 0.4f);

    res.dist = plane;
    res.matId = 0;

    if (torus < res.dist) { res.dist = torus; res.matId = 1; }
    if (rbox < res.dist) { res.dist = rbox; res.matId = 2; }
    if (octa < res.dist) { res.dist = octa; res.matId = 3; }
    if (cyl < res.dist) { res.dist = cyl; res.matId = 4; }

    return res;
}

// Scene 4: Infinite repetition
__device__ SceneResult scene4(float3 p, float time) {
    SceneResult res;

    float plane = sdPlane(p, -1.0f);

    float3 rep = f3(3.0f, 3.0f, 3.0f);
    float3 q = opRep(sub3(p, f3(0, 1.0f, 0)), rep);

    float wobble = 0.1f * sinf(p.x * 2.0f + time) * sinf(p.z * 2.0f + time * 1.3f);
    float spheres = sdSphere(q, 0.5f + wobble);

    res.dist = plane;
    res.matId = 0;

    if (spheres < res.dist) {
        res.dist = spheres;
        res.matId = 1 + ((int)(p.x / 3.0f + 100) + (int)(p.z / 3.0f + 100)) % 4;
    }

    return res;
}

__device__ SceneResult sceneSDF(float3 p, float time, int sceneId) {
    switch (sceneId) {
        case 0: return scene1(p, time);
        case 1: return scene2(p, time);
        case 2: return scene3(p, time);
        case 3: return scene4(p, time);
        default: return scene1(p, time);
    }
}

__device__ float sceneDistOnly(float3 p, float time, int sceneId) {
    return sceneSDF(p, time, sceneId).dist;
}

// ============== RAY MARCHING ==============

__device__ SceneResult rayMarch(float3 ro, float3 rd, float time, int sceneId) {
    SceneResult res;
    res.dist = 0.0f;
    res.matId = -1;

    float t = 0.0f;

    for (int i = 0; i < MAX_STEPS; i++) {
        float3 p = add3(ro, mul3(rd, t));
        SceneResult scene = sceneSDF(p, time, sceneId);

        if (scene.dist < SURF_DIST) {
            res.dist = t;
            res.matId = scene.matId;
            break;
        }

        if (t > MAX_DIST) break;

        t += scene.dist;
    }

    if (res.matId == -1) res.dist = MAX_DIST;

    return res;
}

// ============== NORMAL ESTIMATION ==============

__device__ float3 getNormal(float3 p, float time, int sceneId) {
    float d = sceneDistOnly(p, time, sceneId);
    float3 n = f3(
        d - sceneDistOnly(sub3(p, f3(EPSILON, 0, 0)), time, sceneId),
        d - sceneDistOnly(sub3(p, f3(0, EPSILON, 0)), time, sceneId),
        d - sceneDistOnly(sub3(p, f3(0, 0, EPSILON)), time, sceneId)
    );
    return norm3(n);
}

// ============== SOFT SHADOWS ==============

__device__ float softShadow(float3 ro, float3 rd, float mint, float maxt,
                            float k, float time, int sceneId) {
    float res = 1.0f;
    float t = mint;

    for (int i = 0; i < 32 && t < maxt; i++) {
        float h = sceneDistOnly(add3(ro, mul3(rd, t)), time, sceneId);
        if (h < 0.001f) return 0.0f;
        res = fminf(res, k * h / t);
        t += h;
    }

    return clampf(res, 0.0f, 1.0f);
}

// ============== AMBIENT OCCLUSION ==============

__device__ float ambientOcclusion(float3 p, float3 n, float time, int sceneId) {
    float occ = 0.0f;
    float sca = 1.0f;

    for (int i = 0; i < 5; i++) {
        float h = 0.01f + 0.12f * (float)i;
        float d = sceneDistOnly(add3(p, mul3(n, h)), time, sceneId);
        occ += (h - d) * sca;
        sca *= 0.95f;
    }

    return clampf(1.0f - 3.0f * occ, 0.0f, 1.0f);
}

// ============== MATERIAL COLORS ==============

__device__ float3 getMaterialColor(int matId, float3 p) {
    switch (matId) {
        case 0: {
            float check = fmodf(floorf(p.x) + floorf(p.z), 2.0f);
            return (check > 0.5f) ? f3(0.8f, 0.8f, 0.8f) : f3(0.2f, 0.2f, 0.2f);
        }
        case 1: return f3(1.0f, 0.2f, 0.2f);
        case 2: return f3(0.2f, 1.0f, 0.2f);
        case 3: return f3(0.2f, 0.2f, 1.0f);
        case 4: return f3(1.0f, 1.0f, 0.2f);
        case 5: return f3(1.0f, 0.2f, 1.0f);
        default: return f3(0.8f, 0.8f, 0.8f);
    }
}

// ============== RENDER KERNEL ==============

__global__ void renderKernel(unsigned char* pixels, int width, int height,
                              float3 camPos, float3 camTarget, float fov,
                              float time, int sceneId,
                              int enableShadows, int enableAO) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // Camera setup
    float3 forward = norm3(sub3(camTarget, camPos));
    float3 right = norm3(cross3(f3(0, 1, 0), forward));
    float3 up = cross3(forward, right);

    // Ray direction
    float aspect = (float)width / height;
    float fovScale = tanf(fov * 0.5f * 3.14159f / 180.0f);

    float u = ((float)px / width - 0.5f) * aspect * fovScale;
    float v = (0.5f - (float)py / height) * fovScale;

    float3 rd = norm3(add3(add3(forward, mul3(right, u)), mul3(up, v)));

    // Ray march
    SceneResult hit = rayMarch(camPos, rd, time, sceneId);

    float3 color;

    if (hit.matId >= 0) {
        float3 p = add3(camPos, mul3(rd, hit.dist));
        float3 n = getNormal(p, time, sceneId);

        float3 matColor = getMaterialColor(hit.matId, p);

        // Lighting
        float3 lightPos = f3(5.0f * sinf(time * 0.5f), 8.0f, 5.0f * cosf(time * 0.5f));
        float3 lightDir = norm3(sub3(lightPos, p));
        float3 viewDir = norm3(sub3(camPos, p));
        float3 halfDir = norm3(add3(lightDir, viewDir));

        float diff = fmaxf(dot3(n, lightDir), 0.0f);
        float spec = powf(fmaxf(dot3(n, halfDir), 0.0f), 32.0f);

        float shadow = 1.0f;
        if (enableShadows) {
            float3 shadowOrig = add3(p, mul3(n, 0.01f));
            shadow = softShadow(shadowOrig, lightDir, 0.02f, len3(sub3(lightPos, p)), 16.0f, time, sceneId);
        }

        float ao = 1.0f;
        if (enableAO) {
            ao = ambientOcclusion(p, n, time, sceneId);
        }

        float3 ambient = mul3(f3(0.15f, 0.15f, 0.2f), ao);
        float3 diffuse = mul3(matColor, diff * shadow);
        float3 specular = mul3(f3(1.0f, 1.0f, 1.0f), spec * shadow * 0.5f);

        color = add3(add3(ambient, diffuse), specular);

        // Fog
        float fog = 1.0f - expf(-hit.dist * 0.05f);
        color = mix3(color, f3(0.5f, 0.6f, 0.7f), fog);

    } else {
        float t = 0.5f * (rd.y + 1.0f);
        color = mix3(f3(0.5f, 0.6f, 0.7f), f3(0.1f, 0.2f, 0.4f), t);
    }

    // Gamma
    color = f3(powf(color.x, 0.4545f), powf(color.y, 0.4545f), powf(color.z, 0.4545f));

    int idx = (py * width + px) * 4;
    pixels[idx + 0] = (unsigned char)(clampf(color.z, 0.0f, 1.0f) * 255);
    pixels[idx + 1] = (unsigned char)(clampf(color.y, 0.0f, 1.0f) * 255);
    pixels[idx + 2] = (unsigned char)(clampf(color.x, 0.0f, 1.0f) * 255);
    pixels[idx + 3] = 255;
}

// ============== HOST CODE ==============

int main() {
    printf("=== Windows CUDA Ray Marcher ===\n");
    printf("Procedural 3D Scenes with Signed Distance Fields\n\n");
    printf("Controls:\n");
    printf("  Arrow keys - Orbit camera\n");
    printf("  W/S        - Zoom in/out\n");
    printf("  +/-        - Adjust FOV\n");
    printf("  1-4        - Switch scenes\n");
    printf("  Space      - Toggle animation\n");
    printf("  P          - Toggle soft shadows\n");
    printf("  O          - Toggle ambient occlusion\n");
    printf("  R          - Reset camera\n");
    printf("  Escape     - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA Ray Marcher - SDF Scenes", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    float camDist = 8.0f;
    float camAngleH = 0.5f;
    float camAngleV = 0.3f;
    float camHeight = 2.0f;
    float fov = 60.0f;

    int sceneId = 0;
    int animate = 1;
    int enableShadows = 1;
    int enableAO = 1;

    double startTime = win32_get_time(display);
    double lastTime = startTime;
    double lastFpsTime = startTime;
    int frameCount = 0;
    float animTime = 0.0f;

    const char* sceneNames[] = {"Orbiting Spheres", "CSG Operations", "Geometric Shapes", "Infinite Grid"};
    printf("Scene: %s\n", sceneNames[sceneId]);
    printf("Shadows: ON, AO: ON\n");

    while (!win32_should_close(display)) {
        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;

                if (key == XK_Escape || key == XK_q) goto cleanup;

                if (key == XK_Left) camAngleH -= 0.1f;
                if (key == XK_Right) camAngleH += 0.1f;
                if (key == XK_Up) camAngleV += 0.05f;
                if (key == XK_Down) camAngleV -= 0.05f;
                if (key == XK_w) camDist -= 0.5f;
                if (key == XK_s) camDist += 0.5f;

                camAngleV = fmaxf(-1.0f, fminf(1.0f, camAngleV));
                camDist = fmaxf(2.0f, fminf(20.0f, camDist));

                if (key == XK_plus || key == XK_equal) fov -= 5.0f;
                if (key == XK_minus) fov += 5.0f;
                fov = fmaxf(30.0f, fminf(120.0f, fov));

                if (key == XK_1) { sceneId = 0; printf("Scene: %s\n", sceneNames[0]); }
                if (key == XK_2) { sceneId = 1; printf("Scene: %s\n", sceneNames[1]); }
                if (key == XK_3) { sceneId = 2; printf("Scene: %s\n", sceneNames[2]); }
                if (key == XK_4) { sceneId = 3; printf("Scene: %s\n", sceneNames[3]); }

                if (key == XK_space) {
                    animate = !animate;
                    printf("Animation: %s\n", animate ? "ON" : "OFF");
                }
                if (key == XK_p) {
                    enableShadows = !enableShadows;
                    printf("Shadows: %s\n", enableShadows ? "ON" : "OFF");
                }
                if (key == 'O') {
                    enableAO = !enableAO;
                    printf("AO: %s\n", enableAO ? "ON" : "OFF");
                }

                if (key == XK_r) {
                    camDist = 8.0f;
                    camAngleH = 0.5f;
                    camAngleV = 0.3f;
                    camHeight = 2.0f;
                    fov = 60.0f;
                    printf("Camera reset\n");
                }
            }

            if (event.type == WIN32_EVENT_CLOSE) goto cleanup;
        }

        double now = win32_get_time(display);
        float dt = (float)(now - lastTime);
        lastTime = now;

        if (animate) {
            animTime += dt;
        }

        float3 camPos = make_float3(
            camDist * cosf(camAngleH) * cosf(camAngleV),
            camHeight + camDist * sinf(camAngleV),
            camDist * sinf(camAngleH) * cosf(camAngleV)
        );
        float3 camTarget = make_float3(0, 0.5f, 0);

        renderKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT,
            camPos, camTarget, fov, animTime, sceneId,
            enableShadows, enableAO);

        cudaDeviceSynchronize();

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f\n", frameCount / (now - lastFpsTime));
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
