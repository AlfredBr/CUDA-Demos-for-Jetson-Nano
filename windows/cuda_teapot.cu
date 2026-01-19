// Utah Teapot Renderer - 100% CUDA Software Rasterizer with Phong Shading
// No OpenGL - complete transform, rasterize, shade pipeline in CUDA
// For Windows

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "win32_display.h"

#define WIDTH 800
#define HEIGHT 600
#define MAX_VERTICES 8000
#define MAX_TRIANGLES 8000

// ============== Vector/Matrix Types ==============

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float a) : x(a), y(a), z(a) {}
    __host__ __device__ vec3(float a, float b, float c) : x(a), y(b), z(c) {}
};

struct vec4 {
    float x, y, z, w;
    __host__ __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    __host__ __device__ vec4(vec3 v, float w_) : x(v.x), y(v.y), z(v.z), w(w_) {}
};

// Vector operations
__host__ __device__ vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__host__ __device__ vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__host__ __device__ vec3 operator*(vec3 a, float b) { return vec3(a.x*b, a.y*b, a.z*b); }
__host__ __device__ vec3 operator*(float a, vec3 b) { return vec3(a*b.x, a*b.y, a*b.z); }
__host__ __device__ vec3 operator*(vec3 a, vec3 b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
__host__ __device__ vec3 operator-(vec3 a) { return vec3(-a.x, -a.y, -a.z); }

__host__ __device__ float dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ vec3 cross(vec3 a, vec3 b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__host__ __device__ float len(vec3 v) { return sqrtf(dot(v,v)); }
__host__ __device__ vec3 normalize(vec3 v) {
    float l = len(v);
    return l > 0.0001f ? v * (1.0f/l) : vec3(0);
}

// Matrix operations using float[16] - column-major
__host__ __device__ vec4 mulMV(const float* m, vec4 v) {
    return vec4(
        m[0]*v.x + m[4]*v.y + m[8]*v.z + m[12]*v.w,
        m[1]*v.x + m[5]*v.y + m[9]*v.z + m[13]*v.w,
        m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
        m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*v.w
    );
}

__host__ void mulMM(float* r, const float* a, const float* b) {
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        r[i+j*4] = 0;
        for(int k=0;k<4;k++) r[i+j*4] += a[i+k*4] * b[k+j*4];
    }
}

__host__ void identity(float* m) {
    for(int i=0;i<16;i++) m[i] = (i%5==0)?1.0f:0.0f;
}

__host__ void perspective(float* m, float fov, float aspect, float nearVal, float farVal) {
    for(int i=0;i<16;i++) m[i] = 0;
    float f = 1.0f / tanf(fov * 0.5f);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (farVal + nearVal) / (nearVal - farVal);
    m[11] = -1;
    m[14] = (2 * farVal * nearVal) / (nearVal - farVal);
}

__host__ void lookAt(float* m, vec3 eye, vec3 center, vec3 up) {
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    identity(m);
    m[0] = s.x; m[4] = s.y; m[8] = s.z;
    m[1] = u.x; m[5] = u.y; m[9] = u.z;
    m[2] = -f.x; m[6] = -f.y; m[10] = -f.z;
    m[12] = -dot(s, eye);
    m[13] = -dot(u, eye);
    m[14] = dot(f, eye);
    m[3] = m[7] = m[11] = 0;
}

__host__ void rotateY(float* m, float angle) {
    identity(m);
    float c = cosf(angle), s = sinf(angle);
    m[0] = c; m[8] = s;
    m[2] = -s; m[10] = c;
}

// ============== Triangle Data ==============

struct Triangle {
    int v0, v1, v2;
};

// Device constants - raw arrays to avoid constructor issues
__constant__ float d_mvp[16];
__constant__ float d_model[16];
__constant__ float d_modelIT[16];
__constant__ float d_lightPos[3];
__constant__ float d_viewPos[3];

// ============== Clear Kernels ==============

__global__ void clearFramebuffer(unsigned char* pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= WIDTH * HEIGHT) return;

    int y = idx / WIDTH;
    float t = (float)y / HEIGHT;
    unsigned char bg = (unsigned char)(20 + t * 30);

    int pidx = idx * 4;
    pixels[pidx + 0] = bg;
    pixels[pidx + 1] = bg;
    pixels[pidx + 2] = bg + 10;
    pixels[pidx + 3] = 255;
}

__global__ void clearDepth(float* depth) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= WIDTH * HEIGHT) return;
    depth[idx] = 1.0f;
}

// ============== Rasterization Kernel ==============

__device__ float edgeFunction(float ax, float ay, float bx, float by, float cx, float cy) {
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

__global__ void rasterizeTriangles(
    const vec3* vertices,
    const vec3* normals,
    const Triangle* triangles,
    int numTriangles,
    unsigned char* pixels,
    float* depth
) {
    int triIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (triIdx >= numTriangles) return;

    Triangle tri = triangles[triIdx];

    vec3 v0 = vertices[tri.v0];
    vec3 v1 = vertices[tri.v1];
    vec3 v2 = vertices[tri.v2];

    vec3 n0 = normals[tri.v0];
    vec3 n1 = normals[tri.v1];
    vec3 n2 = normals[tri.v2];

    // Transform to clip space
    vec4 clip0 = mulMV(d_mvp, vec4(v0, 1.0f));
    vec4 clip1 = mulMV(d_mvp, vec4(v1, 1.0f));
    vec4 clip2 = mulMV(d_mvp, vec4(v2, 1.0f));

    // Near-plane culling
    if (clip0.w < 0.1f || clip1.w < 0.1f || clip2.w < 0.1f) return;

    // Perspective divide -> NDC
    float invW0 = 1.0f / clip0.w;
    float invW1 = 1.0f / clip1.w;
    float invW2 = 1.0f / clip2.w;

    vec3 ndc0 = vec3(clip0.x * invW0, clip0.y * invW0, clip0.z * invW0);
    vec3 ndc1 = vec3(clip1.x * invW1, clip1.y * invW1, clip1.z * invW1);
    vec3 ndc2 = vec3(clip2.x * invW2, clip2.y * invW2, clip2.z * invW2);

    // NDC to screen coordinates
    float sx0 = (ndc0.x + 1.0f) * 0.5f * WIDTH;
    float sy0 = (1.0f - ndc0.y) * 0.5f * HEIGHT;
    float sz0 = (ndc0.z + 1.0f) * 0.5f;

    float sx1 = (ndc1.x + 1.0f) * 0.5f * WIDTH;
    float sy1 = (1.0f - ndc1.y) * 0.5f * HEIGHT;
    float sz1 = (ndc1.z + 1.0f) * 0.5f;

    float sx2 = (ndc2.x + 1.0f) * 0.5f * WIDTH;
    float sy2 = (1.0f - ndc2.y) * 0.5f * HEIGHT;
    float sz2 = (ndc2.z + 1.0f) * 0.5f;

    // Bounding box
    int minX = max(0, (int)floorf(fminf(sx0, fminf(sx1, sx2))));
    int maxX = min(WIDTH - 1, (int)ceilf(fmaxf(sx0, fmaxf(sx1, sx2))));
    int minY = max(0, (int)floorf(fminf(sy0, fminf(sy1, sy2))));
    int maxY = min(HEIGHT - 1, (int)ceilf(fmaxf(sy0, fmaxf(sy1, sy2))));

    float area = edgeFunction(sx0, sy0, sx1, sy1, sx2, sy2);
    if (fabsf(area) < 0.001f) return;
    if (area < 0) return;  // Back-face culling

    float invArea = 1.0f / area;

    // World-space positions for lighting
    vec4 world0 = mulMV(d_model, vec4(v0, 1.0f));
    vec4 world1 = mulMV(d_model, vec4(v1, 1.0f));
    vec4 world2 = mulMV(d_model, vec4(v2, 1.0f));
    vec3 wp0 = vec3(world0.x, world0.y, world0.z);
    vec3 wp1 = vec3(world1.x, world1.y, world1.z);
    vec3 wp2 = vec3(world2.x, world2.y, world2.z);

    // Transform normals
    vec4 wn0 = mulMV(d_modelIT, vec4(n0, 0.0f));
    vec4 wn1 = mulMV(d_modelIT, vec4(n1, 0.0f));
    vec4 wn2 = mulMV(d_modelIT, vec4(n2, 0.0f));
    vec3 worldN0 = normalize(vec3(wn0.x, wn0.y, wn0.z));
    vec3 worldN1 = normalize(vec3(wn1.x, wn1.y, wn1.z));
    vec3 worldN2 = normalize(vec3(wn2.x, wn2.y, wn2.z));

    // Material properties (copper)
    vec3 ambient = vec3(0.05f, 0.03f, 0.02f);
    vec3 diffuseColor = vec3(0.7f, 0.4f, 0.2f);
    vec3 specularColor = vec3(1.0f, 0.9f, 0.8f);
    float shininess = 32.0f;
    vec3 lightColor = vec3(1.0f, 0.95f, 0.9f);

    vec3 lightP = vec3(d_lightPos[0], d_lightPos[1], d_lightPos[2]);
    vec3 viewP = vec3(d_viewPos[0], d_viewPos[1], d_viewPos[2]);

    // Rasterize
    for (int py = minY; py <= maxY; py++) {
        for (int px = minX; px <= maxX; px++) {
            float x = px + 0.5f;
            float y = py + 0.5f;

            float w0 = edgeFunction(sx1, sy1, sx2, sy2, x, y) * invArea;
            float w1 = edgeFunction(sx2, sy2, sx0, sy0, x, y) * invArea;
            float w2 = edgeFunction(sx0, sy0, sx1, sy1, x, y) * invArea;

            if (w0 < 0 || w1 < 0 || w2 < 0) continue;

            // Perspective-correct interpolation
            float oneOverW = w0 * invW0 + w1 * invW1 + w2 * invW2;
            float corrW0 = w0 * invW0 / oneOverW;
            float corrW1 = w1 * invW1 / oneOverW;
            float corrW2 = w2 * invW2 / oneOverW;

            float z = sz0 * w0 + sz1 * w1 + sz2 * w2;

            int pixelIdx = py * WIDTH + px;

            // Depth test with atomic CAS
            float oldDepth = depth[pixelIdx];
            if (z >= oldDepth) continue;

            unsigned int assumed, old;
            do {
                assumed = __float_as_uint(oldDepth);
                old = atomicCAS((unsigned int*)&depth[pixelIdx], assumed, __float_as_uint(z));
                oldDepth = __uint_as_float(old);
            } while (oldDepth > z && old != assumed);

            if (oldDepth <= z) continue;

            // Interpolate world position and normal
            vec3 worldPos = wp0 * corrW0 + wp1 * corrW1 + wp2 * corrW2;
            vec3 normal = normalize(worldN0 * corrW0 + worldN1 * corrW1 + worldN2 * corrW2);

            // ====== PHONG SHADING ======
            vec3 L = normalize(lightP - worldPos);
            vec3 V = normalize(viewP - worldPos);

            // Diffuse
            float NdotL = fmaxf(dot(normal, L), 0.0f);
            vec3 diffuse = diffuseColor * lightColor * NdotL;

            // Specular (Blinn-Phong)
            vec3 H = normalize(L + V);
            float NdotH = fmaxf(dot(normal, H), 0.0f);
            float spec = powf(NdotH, shininess);
            vec3 specular = specularColor * lightColor * spec;

            vec3 color = ambient + diffuse + specular * 0.5f;

            // Tone mapping and gamma
            color.x = powf(color.x / (color.x + 1.0f), 0.45f);
            color.y = powf(color.y / (color.y + 1.0f), 0.45f);
            color.z = powf(color.z / (color.z + 1.0f), 0.45f);

            int outIdx = pixelIdx * 4;
            pixels[outIdx + 0] = (unsigned char)(fminf(color.z * 255.0f, 255.0f));
            pixels[outIdx + 1] = (unsigned char)(fminf(color.y * 255.0f, 255.0f));
            pixels[outIdx + 2] = (unsigned char)(fminf(color.x * 255.0f, 255.0f));
            pixels[outIdx + 3] = 255;
        }
    }
}

// ============== OBJ Loader ==============

void loadOBJ(const char* filename,
             vec3* vertices, int* numVertices,
             Triangle* triangles, int* numTriangles,
             vec3* normals) {

    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        exit(1);
    }

    *numVertices = 0;
    *numTriangles = 0;
    int* vertexFaceCount = (int*)calloc(MAX_VERTICES, sizeof(int));

    for (int i = 0; i < MAX_VERTICES; i++) {
        normals[i] = vec3(0, 0, 0);
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            vec3 v;
            sscanf(line + 2, "%f %f %f", &v.x, &v.y, &v.z);
            vertices[*numVertices] = v;
            (*numVertices)++;
        }
        else if (line[0] == 'f' && line[1] == ' ') {
            int v0, v1, v2;
            if (sscanf(line + 2, "%d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &v0, &v1, &v2) == 3 ||
                sscanf(line + 2, "%d//%*d %d//%*d %d//%*d", &v0, &v1, &v2) == 3 ||
                sscanf(line + 2, "%d/%*d %d/%*d %d/%*d", &v0, &v1, &v2) == 3 ||
                sscanf(line + 2, "%d %d %d", &v0, &v1, &v2) == 3) {

                v0--; v1--; v2--;

                triangles[*numTriangles].v0 = v0;
                triangles[*numTriangles].v1 = v1;
                triangles[*numTriangles].v2 = v2;

                vec3 e1 = vertices[v1] - vertices[v0];
                vec3 e2 = vertices[v2] - vertices[v0];
                vec3 fn = normalize(cross(e1, e2));

                normals[v0] = normals[v0] + fn;
                normals[v1] = normals[v1] + fn;
                normals[v2] = normals[v2] + fn;
                vertexFaceCount[v0]++;
                vertexFaceCount[v1]++;
                vertexFaceCount[v2]++;

                (*numTriangles)++;
            }
        }
    }

    fclose(f);

    for (int i = 0; i < *numVertices; i++) {
        if (vertexFaceCount[i] > 0) {
            normals[i] = normalize(normals[i]);
        }
    }

    free(vertexFaceCount);
    printf("Loaded: %d vertices, %d triangles\n", *numVertices, *numTriangles);
}

void normalizeMesh(vec3* vertices, int numVertices) {
    vec3 minV = vec3(FLT_MAX);
    vec3 maxV = vec3(-FLT_MAX);

    for (int i = 0; i < numVertices; i++) {
        minV.x = fminf(minV.x, vertices[i].x);
        minV.y = fminf(minV.y, vertices[i].y);
        minV.z = fminf(minV.z, vertices[i].z);
        maxV.x = fmaxf(maxV.x, vertices[i].x);
        maxV.y = fmaxf(maxV.y, vertices[i].y);
        maxV.z = fmaxf(maxV.z, vertices[i].z);
    }

    vec3 center = (minV + maxV) * 0.5f;
    vec3 size = maxV - minV;
    float maxSize = fmaxf(size.x, fmaxf(size.y, size.z));
    float scale = 2.0f / maxSize;

    for (int i = 0; i < numVertices; i++) {
        vertices[i] = (vertices[i] - center) * scale;
    }

    printf("Mesh normalized: center=(%.2f,%.2f,%.2f), scale=%.4f\n",
           center.x, center.y, center.z, scale);
}

// ============== Main ==============

int main() {
    printf("=== Utah Teapot - CUDA Software Rasterizer ===\n");
    printf("100%% CUDA: Transform -> Rasterize -> Phong Shading\n");
    printf("Resolution: %dx%d\n\n", WIDTH, HEIGHT);

    vec3* h_vertices = (vec3*)malloc(MAX_VERTICES * sizeof(vec3));
    vec3* h_normals = (vec3*)malloc(MAX_VERTICES * sizeof(vec3));
    Triangle* h_triangles = (Triangle*)malloc(MAX_TRIANGLES * sizeof(Triangle));
    int numVertices, numTriangles;

    loadOBJ("teapot.obj", h_vertices, &numVertices, h_triangles, &numTriangles, h_normals);
    normalizeMesh(h_vertices, numVertices);

    vec3 *d_vertices, *d_normals;
    Triangle* d_triangles;
    unsigned char* d_pixels;
    float* d_depth;

    cudaMalloc(&d_vertices, numVertices * sizeof(vec3));
    cudaMalloc(&d_normals, numVertices * sizeof(vec3));
    cudaMalloc(&d_triangles, numTriangles * sizeof(Triangle));
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_depth, WIDTH * HEIGHT * sizeof(float));

    cudaMemcpy(d_vertices, h_vertices, numVertices * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, h_normals, numVertices * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_triangles, h_triangles, numTriangles * sizeof(Triangle), cudaMemcpyHostToDevice);

    Win32Display* display = win32_create_window("Utah Teapot - CUDA Rasterizer", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);

    printf("Controls:\n");
    printf("  Left/Right - Rotate teapot\n");
    printf("  Up/Down    - Change light height\n");
    printf("  W/S        - Zoom in/out\n");
    printf("  Space      - Toggle auto-rotate\n");
    printf("  Q/ESC      - Quit\n\n");

    float angle = 0.0f;
    float lightAngle = 0.0f;
    float lightHeight = 3.0f;
    float camDist = 4.0f;
    int autoRotate = 1;
    int running = 1;

    float h_mvp[16], h_model[16], h_view[16], h_proj[16], h_temp[16];
    float h_lightPos[3], h_viewPos[3];

    while (running && !win32_should_close(display)) {
        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;
                if (key == XK_q || key == XK_Escape) {
                    running = 0;
                } else if (key == XK_Left) {
                    angle -= 0.1f;
                } else if (key == XK_Right) {
                    angle += 0.1f;
                } else if (key == XK_Up) {
                    lightHeight += 0.5f;
                } else if (key == XK_Down) {
                    lightHeight -= 0.5f;
                } else if (key == XK_w) {
                    camDist = fmaxf(2.0f, camDist - 0.2f);
                } else if (key == XK_s) {
                    camDist = fminf(10.0f, camDist + 0.2f);
                } else if (key == XK_space) {
                    autoRotate = !autoRotate;
                    printf("Auto-rotate: %s\n", autoRotate ? "ON" : "OFF");
                }
            }
        }

        if (autoRotate) {
            angle += 0.01f;
            lightAngle += 0.015f;
        }

        // Build matrices
        rotateY(h_model, angle);

        vec3 eye = vec3(cosf(0.3f) * camDist, 1.5f, sinf(0.3f) * camDist);
        vec3 center = vec3(0, 0, 0);
        vec3 up = vec3(0, 1, 0);
        lookAt(h_view, eye, center, up);

        perspective(h_proj, 45.0f * 3.14159f / 180.0f, (float)WIDTH / HEIGHT, 0.1f, 100.0f);

        mulMM(h_temp, h_view, h_model);
        mulMM(h_mvp, h_proj, h_temp);

        h_lightPos[0] = cosf(lightAngle) * 5.0f;
        h_lightPos[1] = lightHeight;
        h_lightPos[2] = sinf(lightAngle) * 5.0f;

        h_viewPos[0] = eye.x;
        h_viewPos[1] = eye.y;
        h_viewPos[2] = eye.z;

        cudaMemcpyToSymbol(d_mvp, h_mvp, sizeof(h_mvp));
        cudaMemcpyToSymbol(d_model, h_model, sizeof(h_model));
        cudaMemcpyToSymbol(d_modelIT, h_model, sizeof(h_model));  // Same for uniform scale
        cudaMemcpyToSymbol(d_lightPos, h_lightPos, sizeof(h_lightPos));
        cudaMemcpyToSymbol(d_viewPos, h_viewPos, sizeof(h_viewPos));

        clearFramebuffer<<<(WIDTH * HEIGHT + 255) / 256, 256>>>(d_pixels);
        clearDepth<<<(WIDTH * HEIGHT + 255) / 256, 256>>>(d_depth);

        int threadsPerBlock = 64;
        int blocks = (numTriangles + threadsPerBlock - 1) / threadsPerBlock;
        rasterizeTriangles<<<blocks, threadsPerBlock>>>(
            d_vertices, d_normals, d_triangles, numTriangles,
            d_pixels, d_depth
        );

        cudaDeviceSynchronize();

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        win32_sleep_ms(16);
    }

    cudaFree(d_vertices);
    cudaFree(d_normals);
    cudaFree(d_triangles);
    cudaFree(d_pixels);
    cudaFree(d_depth);

    free(h_vertices);
    free(h_normals);
    free(h_triangles);
    free(h_pixels);

    win32_destroy_window(display);

    printf("Goodbye!\n");
    return 0;
}
