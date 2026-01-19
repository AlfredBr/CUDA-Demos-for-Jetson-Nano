/*
 * Windows CUDA Solar System Simulation
 *
 * Real-time animation of our solar system with all planets and major moons
 * Uses Keplerian orbital mechanics for accurate motion
 *
 * Features:
 *   - Sun and all 8 planets with accurate colors
 *   - Major moons for each planet
 *   - Scaled orbital periods (adjustable time scale)
 *   - Interactive camera controls
 *   - Planet/moon labels
 *   - Orbital trails
 *
 * Controls:
 *   Arrow keys  - Rotate view
 *   W/S         - Zoom in/out
 *   +/-         - Adjust time scale
 *   T           - Toggle trails
 *   L           - Toggle labels
 *   1-9         - Focus on Sun/planets
 *   Space       - Pause/resume
 *   R           - Reset view
 *   Q/Escape    - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "win32_display.h"

#define WIDTH 1200
#define HEIGHT 900

#define PI 3.14159265359f
#define TWO_PI 6.28318530718f

// Maximum celestial bodies
#define MAX_BODIES 128
#define MAX_TRAIL_POINTS 500

// Body types
#define BODY_STAR 0
#define BODY_PLANET 1
#define BODY_MOON 2

// Celestial body structure
struct CelestialBody {
    char name[32];
    int type;                    // Star, planet, or moon
    int parentIdx;               // Index of parent body (-1 for Sun)

    // Orbital parameters
    float semiMajorAxis;         // Distance from parent (scaled)
    float eccentricity;          // Orbital eccentricity
    float inclination;           // Orbital inclination (radians)
    float orbitalPeriod;         // Orbital period in Earth days
    float meanAnomaly;           // Current mean anomaly

    // Physical properties
    float radius;                // Display radius (scaled)
    float r, g, b;               // Color

    // Current position (computed)
    float x, y, z;
};

// ============== ORBITAL MECHANICS ==============

__device__ __host__ float solveKepler(float M, float e, int iterations = 10) {
    // Solve Kepler's equation: M = E - e*sin(E)
    // Using Newton-Raphson iteration
    float E = M;
    for (int i = 0; i < iterations; i++) {
        E = E - (E - e * sinf(E) - M) / (1.0f - e * cosf(E));
    }
    return E;
}

__device__ __host__ void computeOrbitalPosition(
    float semiMajorAxis, float eccentricity, float inclination,
    float meanAnomaly, float* outX, float* outY, float* outZ)
{
    // Solve Kepler's equation for eccentric anomaly
    float E = solveKepler(meanAnomaly, eccentricity);

    // True anomaly
    float cosE = cosf(E);
    float sinE = sinf(E);
    float x = semiMajorAxis * (cosE - eccentricity);
    float y = semiMajorAxis * sqrtf(1.0f - eccentricity * eccentricity) * sinE;

    // Apply inclination (rotation around X axis)
    float cosI = cosf(inclination);
    float sinI = sinf(inclination);

    *outX = x;
    *outY = y * cosI;
    *outZ = y * sinI;
}

// ============== CUDA KERNELS ==============

__global__ void updateOrbitsKernel(
    CelestialBody* bodies, int numBodies, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    CelestialBody* b = &bodies[i];

    // Update mean anomaly based on orbital period
    if (b->orbitalPeriod > 0) {
        float angularVelocity = TWO_PI / b->orbitalPeriod;
        b->meanAnomaly += angularVelocity * dt;
        if (b->meanAnomaly > TWO_PI) b->meanAnomaly -= TWO_PI;
    }

    // Compute position relative to parent
    float localX, localY, localZ;
    computeOrbitalPosition(b->semiMajorAxis, b->eccentricity, b->inclination,
                          b->meanAnomaly, &localX, &localY, &localZ);

    // Get parent position
    float parentX = 0, parentY = 0, parentZ = 0;
    if (b->parentIdx >= 0 && b->parentIdx < numBodies) {
        parentX = bodies[b->parentIdx].x;
        parentY = bodies[b->parentIdx].y;
        parentZ = bodies[b->parentIdx].z;
    }

    // Update absolute position
    b->x = parentX + localX;
    b->y = parentY + localY;
    b->z = parentZ + localZ;
}

__global__ void clearKernel(unsigned char* pixels, int width, int height, int useTrails) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 4;

    if (useTrails) {
        // Fade for trail effect
        pixels[idx + 0] = (unsigned char)(pixels[idx + 0] * 0.96f);
        pixels[idx + 1] = (unsigned char)(pixels[idx + 1] * 0.96f);
        pixels[idx + 2] = (unsigned char)(pixels[idx + 2] * 0.96f);
    } else {
        // Space background with stars
        unsigned int seed = x * 1973 + y * 9277;
        seed = (seed * 1103515245 + 12345) & 0x7fffffff;
        if ((seed % 2000) == 0) {
            // Random star
            int brightness = 40 + (seed % 60);
            pixels[idx + 0] = brightness;
            pixels[idx + 1] = brightness;
            pixels[idx + 2] = brightness + 20;
        } else {
            pixels[idx + 0] = 2;
            pixels[idx + 1] = 2;
            pixels[idx + 2] = 8;
        }
    }
    pixels[idx + 3] = 255;
}

__device__ void drawFilledCircle(
    unsigned char* pixels, int width, int height,
    int cx, int cy, int radius,
    float r, float g, float b, float brightness)
{
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            if (dx * dx + dy * dy <= radius * radius) {
                int px = cx + dx;
                int py = cy + dy;
                if (px >= 0 && px < width && py >= 0 && py < height) {
                    int idx = (py * width + px) * 4;

                    // Simple shading - brighter toward center
                    float dist = sqrtf((float)(dx * dx + dy * dy)) / (float)radius;
                    float shade = brightness * (1.0f - dist * 0.5f);

                    // Additive blending
                    int newB = pixels[idx + 0] + (int)(b * shade * 255);
                    int newG = pixels[idx + 1] + (int)(g * shade * 255);
                    int newR = pixels[idx + 2] + (int)(r * shade * 255);

                    pixels[idx + 0] = (unsigned char)min(255, newB);
                    pixels[idx + 1] = (unsigned char)min(255, newG);
                    pixels[idx + 2] = (unsigned char)min(255, newR);
                }
            }
        }
    }
}

__device__ void drawRings(
    unsigned char* pixels, int width, int height,
    int cx, int cy, float scale, float rotX,
    float innerRadius, float outerRadius,
    float r, float g, float b)
{
    // Draw Saturn-like rings
    float cosX = cosf(rotX);

    for (float angle = 0; angle < TWO_PI; angle += 0.02f) {
        for (float rad = innerRadius; rad < outerRadius; rad += 0.3f) {
            float rx = rad * cosf(angle);
            float ry = rad * sinf(angle) * cosX * 0.3f;  // Flatten based on view angle

            int px = cx + (int)(rx * scale);
            int py = cy - (int)(ry * scale);

            if (px >= 0 && px < width && py >= 0 && py < height) {
                int idx = (py * width + px) * 4;

                // Ring brightness varies
                float brightness = 0.4f + 0.2f * sinf(rad * 2.0f);

                pixels[idx + 0] = (unsigned char)min(255, pixels[idx + 0] + (int)(b * brightness * 180));
                pixels[idx + 1] = (unsigned char)min(255, pixels[idx + 1] + (int)(g * brightness * 180));
                pixels[idx + 2] = (unsigned char)min(255, pixels[idx + 2] + (int)(r * brightness * 180));
            }
        }
    }
}

__global__ void renderBodiesKernel(
    unsigned char* pixels, int width, int height,
    CelestialBody* bodies, int numBodies,
    float camX, float camY, float camZ,
    float rotX, float rotY, float zoom,
    int focusIdx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numBodies) return;

    CelestialBody* b = &bodies[i];

    // Get position relative to camera focus
    float focusX = 0, focusY = 0, focusZ = 0;
    if (focusIdx >= 0 && focusIdx < numBodies) {
        focusX = bodies[focusIdx].x;
        focusY = bodies[focusIdx].y;
        focusZ = bodies[focusIdx].z;
    }

    float x = b->x - focusX - camX;
    float y = b->y - focusY - camY;
    float z = b->z - focusZ - camZ;

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

    float scale = 500.0f / depth;
    int screenX = (int)(width / 2 + rx * scale);
    int screenY = (int)(height / 2 - ry * scale);

    if (screenX < -100 || screenX >= width + 100 || screenY < -100 || screenY >= height + 100) return;

    // Calculate display radius
    int displayRadius = (int)(b->radius * scale);
    if (displayRadius < 1) displayRadius = 1;
    if (displayRadius > 100) displayRadius = 100;

    // Brightness based on type
    float brightness = 1.0f;
    if (b->type == BODY_STAR) brightness = 1.5f;
    else if (b->type == BODY_MOON) brightness = 0.8f;

    // Draw the body
    drawFilledCircle(pixels, width, height, screenX, screenY, displayRadius,
                    b->r, b->g, b->b, brightness);

    // Draw Saturn's rings (body index 6 is Saturn)
    if (i == 6) {
        drawRings(pixels, width, height, screenX, screenY, scale, rotX,
                 b->radius * 1.5f, b->radius * 2.5f,
                 0.85f, 0.75f, 0.55f);
    }

    // Draw glow for the Sun
    if (b->type == BODY_STAR) {
        for (int gr = displayRadius + 1; gr < displayRadius + 20; gr++) {
            float glowAlpha = 0.3f * (1.0f - (float)(gr - displayRadius) / 20.0f);
            for (float angle = 0; angle < TWO_PI; angle += 0.1f) {
                int gx = screenX + (int)(gr * cosf(angle));
                int gy = screenY + (int)(gr * sinf(angle));
                if (gx >= 0 && gx < width && gy >= 0 && gy < height) {
                    int idx = (gy * width + gx) * 4;
                    pixels[idx + 0] = (unsigned char)min(255, pixels[idx + 0] + (int)(50 * glowAlpha));
                    pixels[idx + 1] = (unsigned char)min(255, pixels[idx + 1] + (int)(180 * glowAlpha));
                    pixels[idx + 2] = (unsigned char)min(255, pixels[idx + 2] + (int)(255 * glowAlpha));
                }
            }
        }
    }
}

// ============== SOLAR SYSTEM INITIALIZATION ==============

void initSolarSystem(CelestialBody* bodies, int* numBodies) {
    int idx = 0;

    // Scale factors for visualization
    // Distances scaled logarithmically for visibility
    // Sizes exaggerated for visibility

    // ===== SUN =====
    strcpy(bodies[idx].name, "Sun");
    bodies[idx].type = BODY_STAR;
    bodies[idx].parentIdx = -1;
    bodies[idx].semiMajorAxis = 0;
    bodies[idx].eccentricity = 0;
    bodies[idx].inclination = 0;
    bodies[idx].orbitalPeriod = 0;
    bodies[idx].meanAnomaly = 0;
    bodies[idx].radius = 8.0f;
    bodies[idx].r = 1.0f; bodies[idx].g = 0.9f; bodies[idx].b = 0.3f;
    bodies[idx].x = bodies[idx].y = bodies[idx].z = 0;
    int sunIdx = idx++;

    // ===== MERCURY =====
    strcpy(bodies[idx].name, "Mercury");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 15.0f;
    bodies[idx].eccentricity = 0.206f;
    bodies[idx].inclination = 0.122f;
    bodies[idx].orbitalPeriod = 88.0f;
    bodies[idx].meanAnomaly = 1.2f;
    bodies[idx].radius = 1.2f;
    bodies[idx].r = 0.7f; bodies[idx].g = 0.7f; bodies[idx].b = 0.7f;  // Gray
    int mercuryIdx = idx++;

    // ===== VENUS =====
    strcpy(bodies[idx].name, "Venus");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 22.0f;
    bodies[idx].eccentricity = 0.007f;
    bodies[idx].inclination = 0.059f;
    bodies[idx].orbitalPeriod = 225.0f;
    bodies[idx].meanAnomaly = 2.5f;
    bodies[idx].radius = 1.8f;
    bodies[idx].r = 0.9f; bodies[idx].g = 0.75f; bodies[idx].b = 0.4f;  // Yellowish-tan
    int venusIdx = idx++;

    // ===== EARTH =====
    strcpy(bodies[idx].name, "Earth");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 30.0f;
    bodies[idx].eccentricity = 0.017f;
    bodies[idx].inclination = 0.0f;
    bodies[idx].orbitalPeriod = 365.25f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 2.0f;
    bodies[idx].r = 0.2f; bodies[idx].g = 0.5f; bodies[idx].b = 0.9f;  // Blue
    int earthIdx = idx++;

    // Moon
    strcpy(bodies[idx].name, "Moon");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = earthIdx;
    bodies[idx].semiMajorAxis = 4.0f;
    bodies[idx].eccentricity = 0.055f;
    bodies[idx].inclination = 0.09f;
    bodies[idx].orbitalPeriod = 27.3f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.6f;
    bodies[idx].r = 0.8f; bodies[idx].g = 0.8f; bodies[idx].b = 0.8f;
    idx++;

    // ===== MARS =====
    strcpy(bodies[idx].name, "Mars");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 42.0f;
    bodies[idx].eccentricity = 0.093f;
    bodies[idx].inclination = 0.032f;
    bodies[idx].orbitalPeriod = 687.0f;
    bodies[idx].meanAnomaly = 3.5f;
    bodies[idx].radius = 1.5f;
    bodies[idx].r = 0.9f; bodies[idx].g = 0.4f; bodies[idx].b = 0.2f;  // Red-orange
    int marsIdx = idx++;

    // Phobos
    strcpy(bodies[idx].name, "Phobos");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = marsIdx;
    bodies[idx].semiMajorAxis = 2.5f;
    bodies[idx].eccentricity = 0.015f;
    bodies[idx].inclination = 0.02f;
    bodies[idx].orbitalPeriod = 0.32f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.3f;
    bodies[idx].r = 0.6f; bodies[idx].g = 0.5f; bodies[idx].b = 0.4f;
    idx++;

    // Deimos
    strcpy(bodies[idx].name, "Deimos");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = marsIdx;
    bodies[idx].semiMajorAxis = 3.5f;
    bodies[idx].eccentricity = 0.0002f;
    bodies[idx].inclination = 0.03f;
    bodies[idx].orbitalPeriod = 1.26f;
    bodies[idx].meanAnomaly = 2.0f;
    bodies[idx].radius = 0.25f;
    bodies[idx].r = 0.65f; bodies[idx].g = 0.55f; bodies[idx].b = 0.45f;
    idx++;

    // ===== JUPITER =====
    strcpy(bodies[idx].name, "Jupiter");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 65.0f;
    bodies[idx].eccentricity = 0.049f;
    bodies[idx].inclination = 0.023f;
    bodies[idx].orbitalPeriod = 4333.0f;
    bodies[idx].meanAnomaly = 1.0f;
    bodies[idx].radius = 5.0f;
    bodies[idx].r = 0.9f; bodies[idx].g = 0.7f; bodies[idx].b = 0.5f;  // Orange-tan
    int jupiterIdx = idx++;

    // Io
    strcpy(bodies[idx].name, "Io");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = jupiterIdx;
    bodies[idx].semiMajorAxis = 7.0f;
    bodies[idx].eccentricity = 0.004f;
    bodies[idx].inclination = 0.04f;
    bodies[idx].orbitalPeriod = 1.77f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.5f;
    bodies[idx].r = 1.0f; bodies[idx].g = 0.9f; bodies[idx].b = 0.3f;  // Yellowish (volcanic)
    idx++;

    // Europa
    strcpy(bodies[idx].name, "Europa");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = jupiterIdx;
    bodies[idx].semiMajorAxis = 9.0f;
    bodies[idx].eccentricity = 0.009f;
    bodies[idx].inclination = 0.08f;
    bodies[idx].orbitalPeriod = 3.55f;
    bodies[idx].meanAnomaly = 1.5f;
    bodies[idx].radius = 0.45f;
    bodies[idx].r = 0.85f; bodies[idx].g = 0.8f; bodies[idx].b = 0.7f;  // Icy white-tan
    idx++;

    // Ganymede
    strcpy(bodies[idx].name, "Ganymede");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = jupiterIdx;
    bodies[idx].semiMajorAxis = 11.0f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.03f;
    bodies[idx].orbitalPeriod = 7.15f;
    bodies[idx].meanAnomaly = 3.0f;
    bodies[idx].radius = 0.6f;
    bodies[idx].r = 0.7f; bodies[idx].g = 0.65f; bodies[idx].b = 0.6f;  // Grayish
    idx++;

    // Callisto
    strcpy(bodies[idx].name, "Callisto");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = jupiterIdx;
    bodies[idx].semiMajorAxis = 14.0f;
    bodies[idx].eccentricity = 0.007f;
    bodies[idx].inclination = 0.05f;
    bodies[idx].orbitalPeriod = 16.69f;
    bodies[idx].meanAnomaly = 4.5f;
    bodies[idx].radius = 0.55f;
    bodies[idx].r = 0.5f; bodies[idx].g = 0.45f; bodies[idx].b = 0.4f;  // Dark
    idx++;

    // ===== SATURN =====
    strcpy(bodies[idx].name, "Saturn");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 95.0f;
    bodies[idx].eccentricity = 0.057f;
    bodies[idx].inclination = 0.043f;
    bodies[idx].orbitalPeriod = 10759.0f;
    bodies[idx].meanAnomaly = 2.0f;
    bodies[idx].radius = 4.5f;
    bodies[idx].r = 0.9f; bodies[idx].g = 0.8f; bodies[idx].b = 0.5f;  // Pale gold
    int saturnIdx = idx++;

    // Titan
    strcpy(bodies[idx].name, "Titan");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 10.0f;
    bodies[idx].eccentricity = 0.029f;
    bodies[idx].inclination = 0.33f;
    bodies[idx].orbitalPeriod = 15.95f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.6f;
    bodies[idx].r = 0.85f; bodies[idx].g = 0.65f; bodies[idx].b = 0.3f;  // Orange haze
    idx++;

    // Rhea
    strcpy(bodies[idx].name, "Rhea");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 7.5f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.35f;
    bodies[idx].orbitalPeriod = 4.52f;
    bodies[idx].meanAnomaly = 1.0f;
    bodies[idx].radius = 0.35f;
    bodies[idx].r = 0.8f; bodies[idx].g = 0.8f; bodies[idx].b = 0.8f;
    idx++;

    // Iapetus
    strcpy(bodies[idx].name, "Iapetus");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 14.0f;
    bodies[idx].eccentricity = 0.028f;
    bodies[idx].inclination = 0.27f;
    bodies[idx].orbitalPeriod = 79.32f;
    bodies[idx].meanAnomaly = 2.0f;
    bodies[idx].radius = 0.35f;
    bodies[idx].r = 0.6f; bodies[idx].g = 0.5f; bodies[idx].b = 0.4f;  // Two-toned
    idx++;

    // Dione
    strcpy(bodies[idx].name, "Dione");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 6.0f;
    bodies[idx].eccentricity = 0.002f;
    bodies[idx].inclination = 0.02f;
    bodies[idx].orbitalPeriod = 2.74f;
    bodies[idx].meanAnomaly = 3.0f;
    bodies[idx].radius = 0.3f;
    bodies[idx].r = 0.85f; bodies[idx].g = 0.85f; bodies[idx].b = 0.85f;
    idx++;

    // Tethys
    strcpy(bodies[idx].name, "Tethys");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 5.2f;
    bodies[idx].eccentricity = 0.0001f;
    bodies[idx].inclination = 0.02f;
    bodies[idx].orbitalPeriod = 1.89f;
    bodies[idx].meanAnomaly = 4.0f;
    bodies[idx].radius = 0.28f;
    bodies[idx].r = 0.9f; bodies[idx].g = 0.9f; bodies[idx].b = 0.9f;  // Icy
    idx++;

    // Enceladus
    strcpy(bodies[idx].name, "Enceladus");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 4.5f;
    bodies[idx].eccentricity = 0.005f;
    bodies[idx].inclination = 0.02f;
    bodies[idx].orbitalPeriod = 1.37f;
    bodies[idx].meanAnomaly = 5.0f;
    bodies[idx].radius = 0.25f;
    bodies[idx].r = 0.95f; bodies[idx].g = 0.95f; bodies[idx].b = 1.0f;  // Bright icy
    idx++;

    // Mimas
    strcpy(bodies[idx].name, "Mimas");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = saturnIdx;
    bodies[idx].semiMajorAxis = 3.8f;
    bodies[idx].eccentricity = 0.02f;
    bodies[idx].inclination = 0.03f;
    bodies[idx].orbitalPeriod = 0.94f;
    bodies[idx].meanAnomaly = 0.5f;
    bodies[idx].radius = 0.22f;
    bodies[idx].r = 0.8f; bodies[idx].g = 0.8f; bodies[idx].b = 0.78f;
    idx++;

    // ===== URANUS =====
    strcpy(bodies[idx].name, "Uranus");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 130.0f;
    bodies[idx].eccentricity = 0.046f;
    bodies[idx].inclination = 0.013f;
    bodies[idx].orbitalPeriod = 30687.0f;
    bodies[idx].meanAnomaly = 4.0f;
    bodies[idx].radius = 3.0f;
    bodies[idx].r = 0.5f; bodies[idx].g = 0.8f; bodies[idx].b = 0.9f;  // Cyan
    int uranusIdx = idx++;

    // Miranda
    strcpy(bodies[idx].name, "Miranda");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = uranusIdx;
    bodies[idx].semiMajorAxis = 4.0f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.07f;
    bodies[idx].orbitalPeriod = 1.41f;
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.25f;
    bodies[idx].r = 0.7f; bodies[idx].g = 0.7f; bodies[idx].b = 0.7f;
    idx++;

    // Ariel
    strcpy(bodies[idx].name, "Ariel");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = uranusIdx;
    bodies[idx].semiMajorAxis = 5.5f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.04f;
    bodies[idx].orbitalPeriod = 2.52f;
    bodies[idx].meanAnomaly = 1.0f;
    bodies[idx].radius = 0.3f;
    bodies[idx].r = 0.75f; bodies[idx].g = 0.75f; bodies[idx].b = 0.75f;
    idx++;

    // Umbriel
    strcpy(bodies[idx].name, "Umbriel");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = uranusIdx;
    bodies[idx].semiMajorAxis = 6.5f;
    bodies[idx].eccentricity = 0.004f;
    bodies[idx].inclination = 0.03f;
    bodies[idx].orbitalPeriod = 4.14f;
    bodies[idx].meanAnomaly = 2.0f;
    bodies[idx].radius = 0.28f;
    bodies[idx].r = 0.5f; bodies[idx].g = 0.5f; bodies[idx].b = 0.5f;  // Dark
    idx++;

    // Titania
    strcpy(bodies[idx].name, "Titania");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = uranusIdx;
    bodies[idx].semiMajorAxis = 8.0f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.08f;
    bodies[idx].orbitalPeriod = 8.71f;
    bodies[idx].meanAnomaly = 3.0f;
    bodies[idx].radius = 0.35f;
    bodies[idx].r = 0.7f; bodies[idx].g = 0.68f; bodies[idx].b = 0.65f;
    idx++;

    // Oberon
    strcpy(bodies[idx].name, "Oberon");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = uranusIdx;
    bodies[idx].semiMajorAxis = 10.0f;
    bodies[idx].eccentricity = 0.001f;
    bodies[idx].inclination = 0.07f;
    bodies[idx].orbitalPeriod = 13.46f;
    bodies[idx].meanAnomaly = 4.0f;
    bodies[idx].radius = 0.33f;
    bodies[idx].r = 0.6f; bodies[idx].g = 0.55f; bodies[idx].b = 0.5f;
    idx++;

    // ===== NEPTUNE =====
    strcpy(bodies[idx].name, "Neptune");
    bodies[idx].type = BODY_PLANET;
    bodies[idx].parentIdx = sunIdx;
    bodies[idx].semiMajorAxis = 160.0f;
    bodies[idx].eccentricity = 0.011f;
    bodies[idx].inclination = 0.031f;
    bodies[idx].orbitalPeriod = 60190.0f;
    bodies[idx].meanAnomaly = 5.0f;
    bodies[idx].radius = 2.8f;
    bodies[idx].r = 0.3f; bodies[idx].g = 0.4f; bodies[idx].b = 0.9f;  // Deep blue
    int neptuneIdx = idx++;

    // Triton
    strcpy(bodies[idx].name, "Triton");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = neptuneIdx;
    bodies[idx].semiMajorAxis = 6.0f;
    bodies[idx].eccentricity = 0.00002f;
    bodies[idx].inclination = 2.72f;  // Retrograde, high inclination
    bodies[idx].orbitalPeriod = -5.88f;  // Negative for retrograde
    bodies[idx].meanAnomaly = 0.0f;
    bodies[idx].radius = 0.4f;
    bodies[idx].r = 0.75f; bodies[idx].g = 0.7f; bodies[idx].b = 0.65f;
    idx++;

    // Nereid
    strcpy(bodies[idx].name, "Nereid");
    bodies[idx].type = BODY_MOON;
    bodies[idx].parentIdx = neptuneIdx;
    bodies[idx].semiMajorAxis = 12.0f;
    bodies[idx].eccentricity = 0.75f;  // Highly eccentric
    bodies[idx].inclination = 0.12f;
    bodies[idx].orbitalPeriod = 360.0f;
    bodies[idx].meanAnomaly = 2.0f;
    bodies[idx].radius = 0.25f;
    bodies[idx].r = 0.65f; bodies[idx].g = 0.65f; bodies[idx].b = 0.65f;
    idx++;

    *numBodies = idx;
}

// Get planet indices for focus
int getPlanetIndex(int focusKey) {
    // Returns the body index for focus key 0-8
    // 0=Sun, 1=Mercury, 2=Venus, 3=Earth, 4=Mars, 5=Jupiter, 6=Saturn, 7=Uranus, 8=Neptune
    int indices[] = {0, 1, 2, 3, 5, 8, 13, 21, 27};
    if (focusKey >= 0 && focusKey <= 8) return indices[focusKey];
    return 0;
}

const char* getPlanetName(int focusKey) {
    const char* names[] = {"Sun", "Mercury", "Venus", "Earth", "Mars",
                           "Jupiter", "Saturn", "Uranus", "Neptune"};
    if (focusKey >= 0 && focusKey <= 8) return names[focusKey];
    return "Sun";
}

// ============== HOST CODE ==============

int main() {
    printf("=== Windows CUDA Solar System Simulation ===\n\n");
    printf("Controls:\n");
    printf("  0       - Focus on Sun\n");
    printf("  1-8     - Focus on planets (Mercury-Neptune)\n");
    printf("  Arrows  - Rotate view\n");
    printf("  W/S     - Zoom in/out\n");
    printf("  +/-     - Time scale\n");
    printf("  T       - Toggle trails\n");
    printf("  Space   - Pause/resume\n");
    printf("  R       - Reset view\n");
    printf("  Q/Esc   - Quit\n\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    // Create Win32 window
    Win32Display* display = win32_create_window("CUDA Solar System Simulation", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate bodies
    CelestialBody* h_bodies = (CelestialBody*)malloc(MAX_BODIES * sizeof(CelestialBody));
    CelestialBody* d_bodies;
    cudaMalloc(&d_bodies, MAX_BODIES * sizeof(CelestialBody));

    // Allocate display buffer
    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);
    cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);

    // Initialize solar system
    int numBodies = 0;
    initSolarSystem(h_bodies, &numBodies);
    printf("Initialized %d celestial bodies\n", numBodies);

    // Copy to device
    cudaMemcpy(d_bodies, h_bodies, numBodies * sizeof(CelestialBody), cudaMemcpyHostToDevice);

    // Simulation state
    float timeScale = 5.0f;  // Days per frame
    int paused = 0;
    int showTrails = 0;
    float rotX = 0.5f, rotY = 0.0f;
    float zoom = 250.0f;
    float camX = 0, camY = 0, camZ = 0;
    int focusKey = 0;  // Focus on Sun
    int focusIdx = 0;

    // Kernel dimensions
    dim3 bodyBlock(128);
    dim3 bodyGrid((numBodies + 127) / 128);

    dim3 dispBlock(16, 16);
    dim3 dispGrid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    double lastTime = win32_get_time(display);
    double lastFpsTime = lastTime;
    int frameCount = 0;
    float simTime = 0;  // Simulation time in days

    printf("Focus: %s | Time scale: %.1f days/frame\n", getPlanetName(focusKey), timeScale);

    while (!win32_should_close(display)) {
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
                if (key == XK_w) zoom -= 10.0f;
                if (key == XK_s) zoom += 10.0f;
                zoom = fmaxf(30.0f, fminf(500.0f, zoom));

                // Time scale
                if (key == XK_plus || key == XK_equal) timeScale *= 1.5f;
                if (key == XK_minus) timeScale /= 1.5f;
                timeScale = fmaxf(0.1f, fminf(100.0f, timeScale));

                // Toggles
                if (key == XK_space) {
                    paused = !paused;
                    printf("%s\n", paused ? "Paused" : "Running");
                }
                if (key == XK_t) {
                    showTrails = !showTrails;
                    if (!showTrails) cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);
                    printf("Trails: %s\n", showTrails ? "ON" : "OFF");
                }

                // Focus controls
                int newFocus = -1;
                if (key == XK_0) newFocus = 0;
                if (key == XK_1) newFocus = 1;
                if (key == XK_2) newFocus = 2;
                if (key == XK_3) newFocus = 3;
                if (key == XK_4) newFocus = 4;
                if (key == XK_5) newFocus = 5;
                if (key == XK_6) newFocus = 6;
                if (key == XK_7) newFocus = 7;
                if (key == XK_8) newFocus = 8;

                if (newFocus >= 0) {
                    focusKey = newFocus;
                    focusIdx = getPlanetIndex(focusKey);

                    // Adjust zoom based on focus
                    if (focusKey == 0) zoom = 250.0f;  // Sun - full system view
                    else if (focusKey <= 4) zoom = 60.0f;  // Inner planets
                    else zoom = 100.0f;  // Outer planets

                    printf("Focus: %s\n", getPlanetName(focusKey));
                }

                // Reset
                if (key == XK_r) {
                    rotX = 0.5f; rotY = 0.0f;
                    zoom = 250.0f;
                    focusKey = 0;
                    focusIdx = 0;
                    timeScale = 5.0f;
                    cudaMemset(d_pixels, 0, WIDTH * HEIGHT * 4);
                    printf("View reset\n");
                }
            }

            if (event.type == WIN32_EVENT_CLOSE) goto cleanup;
        }

        // Update simulation
        if (!paused) {
            updateOrbitsKernel<<<bodyGrid, bodyBlock>>>(d_bodies, numBodies, timeScale);
            simTime += timeScale;
        }

        // Copy updated positions back for focus tracking
        cudaMemcpy(h_bodies, d_bodies, numBodies * sizeof(CelestialBody), cudaMemcpyDeviceToHost);

        // Render
        clearKernel<<<dispGrid, dispBlock>>>(d_pixels, WIDTH, HEIGHT, showTrails);

        renderBodiesKernel<<<bodyGrid, bodyBlock>>>(
            d_pixels, WIDTH, HEIGHT,
            d_bodies, numBodies,
            camX, camY, camZ,
            rotX, rotY, zoom,
            focusIdx);

        cudaDeviceSynchronize();

        // Display
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        double now = win32_get_time(display);
        if (now - lastFpsTime >= 1.0) {
            float years = simTime / 365.25f;
            printf("FPS: %.1f | Sim: %.2f years | Focus: %s | Time: %.1f days/frame\n",
                   frameCount / (now - lastFpsTime), years, getPlanetName(focusKey), timeScale);
            frameCount = 0;
            lastFpsTime = now;
        }
        lastTime = now;
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);

    cudaFree(d_bodies);
    cudaFree(d_pixels);
    free(h_bodies);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
