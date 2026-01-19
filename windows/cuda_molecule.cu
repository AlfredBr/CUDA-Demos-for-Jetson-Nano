/*
 * Windows CUDA Molecule Visualization
 *
 * Renders realistic 3D ball-and-stick molecular models
 * Uses CPK coloring convention for atoms
 *
 * Features:
 *   - Realistic atomic colors and sizes
 *   - Ball-and-stick representation
 *   - Multiple molecule presets (organic compounds, DNA bases, etc.)
 *   - Random molecule generation
 *   - Smooth camera rotation
 *   - Metallic shading with specular highlights
 *
 * Controls:
 *   1-9       - Select molecule preset
 *   R         - Random molecule
 *   Arrow keys - Rotate view
 *   W/S       - Zoom in/out
 *   A         - Toggle auto-rotate
 *   Space     - Pause/resume rotation
 *   Q/Escape  - Quit
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

// Molecule limits
#define MAX_ATOMS 200
#define MAX_BONDS 250

// Atom types (for coloring)
#define ATOM_H  0   // Hydrogen - white
#define ATOM_C  1   // Carbon - dark gray/black
#define ATOM_N  2   // Nitrogen - blue
#define ATOM_O  3   // Oxygen - red
#define ATOM_P  4   // Phosphorus - orange
#define ATOM_S  5   // Sulfur - yellow
#define ATOM_CL 6   // Chlorine - green
#define ATOM_BR 7   // Bromine - dark red
#define ATOM_F  8   // Fluorine - light green
#define ATOM_I  9   // Iodine - purple

// Atom structure
struct Atom {
    float x, y, z;
    int type;
    float radius;
};

// Bond structure
struct Bond {
    int atom1, atom2;
    int order;  // 1=single, 2=double, 3=triple
};

// Molecule structure
struct Molecule {
    Atom atoms[MAX_ATOMS];
    Bond bonds[MAX_BONDS];
    int numAtoms;
    int numBonds;
    char name[64];
};

// CPK colors for atoms (R, G, B)
__device__ __constant__ float3 atomColors[10] = {
    {0.95f, 0.95f, 0.95f},  // H - white
    {0.2f,  0.2f,  0.2f},   // C - dark gray
    {0.2f,  0.3f,  0.9f},   // N - blue
    {0.9f,  0.2f,  0.2f},   // O - red
    {1.0f,  0.5f,  0.0f},   // P - orange
    {0.9f,  0.8f,  0.2f},   // S - yellow
    {0.2f,  0.9f,  0.2f},   // Cl - green
    {0.6f,  0.1f,  0.1f},   // Br - dark red
    {0.5f,  0.9f,  0.5f},   // F - light green
    {0.5f,  0.1f,  0.5f},   // I - purple
};

// Atomic radii (van der Waals, scaled for visualization)
__device__ __constant__ float atomRadii[10] = {
    0.25f,  // H
    0.40f,  // C
    0.38f,  // N
    0.35f,  // O
    0.45f,  // P
    0.45f,  // S
    0.45f,  // Cl
    0.50f,  // Br
    0.35f,  // F
    0.55f,  // I
};

// ============== 3D MATH HELPERS ==============

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

__device__ float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
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

// Ray-cylinder intersection
__device__ float intersectCylinder(float3 ro, float3 rd, float3 pa, float3 pb, float radius, float3* outNormal) {
    float3 ba = make_float3(pb.x - pa.x, pb.y - pa.y, pb.z - pa.z);
    float3 oc = make_float3(ro.x - pa.x, ro.y - pa.y, ro.z - pa.z);

    float baba = dot3(ba, ba);
    float bard = dot3(ba, rd);
    float baoc = dot3(ba, oc);

    float k2 = baba - bard * bard;
    float k1 = baba * dot3(oc, rd) - baoc * bard;
    float k0 = baba * dot3(oc, oc) - baoc * baoc - radius * radius * baba;

    if (fabsf(k2) < 0.0001f) return -1.0f;

    float h = k1 * k1 - k2 * k0;
    if (h < 0.0f) return -1.0f;

    h = sqrtf(h);
    float t = (-k1 - h) / k2;

    float y = baoc + t * bard;
    if (y > 0.0f && y < baba && t > 0.0f) {
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

    // Gradient background
    float gy = (float)y / height;
    pixels[idx + 0] = (unsigned char)(20 + gy * 30);
    pixels[idx + 1] = (unsigned char)(25 + gy * 35);
    pixels[idx + 2] = (unsigned char)(35 + gy * 40);
    pixels[idx + 3] = 255;
}

// Simple 6x8 bitmap font for text rendering (covers ASCII 32-127)
// Each character is 6 pixels wide, stored in the lower 6 bits of each byte
__device__ unsigned char getPixel(int c, int px, int py) {
    // Simplified built-in font - returns 1 if pixel should be lit
    if (c < 32 || c > 127) return 0;

    // Font patterns for common characters (6 wide x 8 tall)
    static const unsigned char font[][8] = {
        {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 32 space
        {0x04,0x04,0x04,0x04,0x04,0x00,0x04,0x00}, // 33 !
        {0x0A,0x0A,0x0A,0x00,0x00,0x00,0x00,0x00}, // 34 "
        {0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A,0x00}, // 35 #
        {0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04,0x00}, // 36 $
        {0x18,0x19,0x02,0x04,0x08,0x13,0x03,0x00}, // 37 %
        {0x08,0x14,0x14,0x08,0x15,0x12,0x0D,0x00}, // 38 &
        {0x04,0x04,0x08,0x00,0x00,0x00,0x00,0x00}, // 39 '
        {0x02,0x04,0x08,0x08,0x08,0x04,0x02,0x00}, // 40 (
        {0x08,0x04,0x02,0x02,0x02,0x04,0x08,0x00}, // 41 )
        {0x00,0x04,0x15,0x0E,0x15,0x04,0x00,0x00}, // 42 *
        {0x00,0x04,0x04,0x1F,0x04,0x04,0x00,0x00}, // 43 +
        {0x00,0x00,0x00,0x00,0x00,0x04,0x04,0x08}, // 44 ,
        {0x00,0x00,0x00,0x1F,0x00,0x00,0x00,0x00}, // 45 -
        {0x00,0x00,0x00,0x00,0x00,0x00,0x04,0x00}, // 46 .
        {0x01,0x02,0x02,0x04,0x08,0x08,0x10,0x00}, // 47 /
        {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E,0x00}, // 48 0
        {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E,0x00}, // 49 1
        {0x0E,0x11,0x01,0x02,0x04,0x08,0x1F,0x00}, // 50 2
        {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E,0x00}, // 51 3
        {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02,0x00}, // 52 4
        {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E,0x00}, // 53 5
        {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E,0x00}, // 54 6
        {0x1F,0x01,0x02,0x04,0x08,0x08,0x08,0x00}, // 55 7
        {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E,0x00}, // 56 8
        {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C,0x00}, // 57 9
        {0x00,0x00,0x04,0x00,0x00,0x04,0x00,0x00}, // 58 :
        {0x00,0x00,0x04,0x00,0x00,0x04,0x04,0x08}, // 59 ;
        {0x02,0x04,0x08,0x10,0x08,0x04,0x02,0x00}, // 60 <
        {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00,0x00}, // 61 =
        {0x08,0x04,0x02,0x01,0x02,0x04,0x08,0x00}, // 62 >
        {0x0E,0x11,0x01,0x02,0x04,0x00,0x04,0x00}, // 63 ?
        {0x0E,0x11,0x17,0x15,0x17,0x10,0x0E,0x00}, // 64 @
        {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11,0x00}, // 65 A
        {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E,0x00}, // 66 B
        {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E,0x00}, // 67 C
        {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E,0x00}, // 68 D
        {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F,0x00}, // 69 E
        {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10,0x00}, // 70 F
        {0x0E,0x11,0x10,0x17,0x11,0x11,0x0F,0x00}, // 71 G
        {0x11,0x11,0x11,0x1F,0x11,0x11,0x11,0x00}, // 72 H
        {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E,0x00}, // 73 I
        {0x07,0x02,0x02,0x02,0x02,0x12,0x0C,0x00}, // 74 J
        {0x11,0x12,0x14,0x18,0x14,0x12,0x11,0x00}, // 75 K
        {0x10,0x10,0x10,0x10,0x10,0x10,0x1F,0x00}, // 76 L
        {0x11,0x1B,0x15,0x15,0x11,0x11,0x11,0x00}, // 77 M
        {0x11,0x19,0x15,0x13,0x11,0x11,0x11,0x00}, // 78 N
        {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E,0x00}, // 79 O
        {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10,0x00}, // 80 P
        {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D,0x00}, // 81 Q
        {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11,0x00}, // 82 R
        {0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E,0x00}, // 83 S
        {0x1F,0x04,0x04,0x04,0x04,0x04,0x04,0x00}, // 84 T
        {0x11,0x11,0x11,0x11,0x11,0x11,0x0E,0x00}, // 85 U
        {0x11,0x11,0x11,0x11,0x11,0x0A,0x04,0x00}, // 86 V
        {0x11,0x11,0x11,0x15,0x15,0x1B,0x11,0x00}, // 87 W
        {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11,0x00}, // 88 X
        {0x11,0x11,0x0A,0x04,0x04,0x04,0x04,0x00}, // 89 Y
        {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F,0x00}, // 90 Z
        {0x0E,0x08,0x08,0x08,0x08,0x08,0x0E,0x00}, // 91 [
        {0x10,0x08,0x08,0x04,0x02,0x02,0x01,0x00}, // 92 backslash
        {0x0E,0x02,0x02,0x02,0x02,0x02,0x0E,0x00}, // 93 ]
        {0x04,0x0A,0x11,0x00,0x00,0x00,0x00,0x00}, // 94 ^
        {0x00,0x00,0x00,0x00,0x00,0x00,0x1F,0x00}, // 95 _
        {0x08,0x04,0x02,0x00,0x00,0x00,0x00,0x00}, // 96 `
        {0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F,0x00}, // 97 a
        {0x10,0x10,0x1E,0x11,0x11,0x11,0x1E,0x00}, // 98 b
        {0x00,0x00,0x0E,0x11,0x10,0x11,0x0E,0x00}, // 99 c
        {0x01,0x01,0x0F,0x11,0x11,0x11,0x0F,0x00}, // 100 d
        {0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E,0x00}, // 101 e
        {0x02,0x05,0x04,0x0E,0x04,0x04,0x04,0x00}, // 102 f
        {0x00,0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E}, // 103 g
        {0x10,0x10,0x16,0x19,0x11,0x11,0x11,0x00}, // 104 h
        {0x04,0x00,0x0C,0x04,0x04,0x04,0x0E,0x00}, // 105 i
        {0x02,0x00,0x06,0x02,0x02,0x02,0x12,0x0C}, // 106 j
        {0x10,0x10,0x12,0x14,0x18,0x14,0x12,0x00}, // 107 k
        {0x0C,0x04,0x04,0x04,0x04,0x04,0x0E,0x00}, // 108 l
        {0x00,0x00,0x1A,0x15,0x15,0x11,0x11,0x00}, // 109 m
        {0x00,0x00,0x16,0x19,0x11,0x11,0x11,0x00}, // 110 n
        {0x00,0x00,0x0E,0x11,0x11,0x11,0x0E,0x00}, // 111 o
        {0x00,0x00,0x1E,0x11,0x11,0x1E,0x10,0x10}, // 112 p
        {0x00,0x00,0x0F,0x11,0x11,0x0F,0x01,0x01}, // 113 q
        {0x00,0x00,0x16,0x19,0x10,0x10,0x10,0x00}, // 114 r
        {0x00,0x00,0x0E,0x10,0x0E,0x01,0x1E,0x00}, // 115 s
        {0x04,0x04,0x0E,0x04,0x04,0x05,0x02,0x00}, // 116 t
        {0x00,0x00,0x11,0x11,0x11,0x13,0x0D,0x00}, // 117 u
        {0x00,0x00,0x11,0x11,0x11,0x0A,0x04,0x00}, // 118 v
        {0x00,0x00,0x11,0x11,0x15,0x15,0x0A,0x00}, // 119 w
        {0x00,0x00,0x11,0x0A,0x04,0x0A,0x11,0x00}, // 120 x
        {0x00,0x00,0x11,0x11,0x11,0x0F,0x01,0x0E}, // 121 y
        {0x00,0x00,0x1F,0x02,0x04,0x08,0x1F,0x00}, // 122 z
        {0x02,0x04,0x04,0x08,0x04,0x04,0x02,0x00}, // 123 {
        {0x04,0x04,0x04,0x04,0x04,0x04,0x04,0x00}, // 124 |
        {0x08,0x04,0x04,0x02,0x04,0x04,0x08,0x00}, // 125 }
        {0x00,0x00,0x08,0x15,0x02,0x00,0x00,0x00}, // 126 ~
        {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // 127
    };

    int idx = c - 32;
    if (idx < 0 || idx >= 96) return 0;
    if (py < 0 || py >= 8) return 0;
    if (px < 0 || px >= 6) return 0;

    unsigned char row = font[idx][py];
    return (row >> (4 - px)) & 1;
}

// Text rendering kernel - renders scaled bitmap text with subscript support for chemical formulas
// Subscripts: digits that follow letters (element symbols) are rendered smaller and lower
__device__ bool isElementChar(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

__device__ bool isDigit(char c) {
    return c >= '0' && c <= '9';
}

// Check if character at position should be subscripted (digit following element symbol)
__device__ bool shouldSubscript(const char* text, int textLen, int charIndex) {
    if (charIndex <= 0 || charIndex >= textLen) return false;
    char current = text[charIndex];
    char prev = text[charIndex - 1];

    // Digit following a letter (element symbol) should be subscript
    if (isDigit(current) && isElementChar(prev)) return true;

    // Digit following another subscript digit (e.g., "10" in H10)
    if (isDigit(current) && isDigit(prev) && charIndex >= 2) {
        // Check if the digit sequence started after a letter
        int i = charIndex - 1;
        while (i >= 0 && isDigit(text[i])) i--;
        if (i >= 0 && isElementChar(text[i])) return true;
    }

    return false;
}

__global__ void renderTextKernel(
    unsigned char* pixels, int width, int height,
    const char* text, int textLen, int startX, int startY, int scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate character positions with subscript handling
    int charHeight = 8 * scale;
    int subScale = (scale * 2) / 3;  // Subscript is 2/3 size
    if (subScale < 1) subScale = 1;
    int subCharWidth = 6 * subScale;
    int subCharHeight = 8 * subScale;
    int subOffsetY = charHeight - subCharHeight;  // Subscript lowered

    // Calculate total text width and find which character we're in
    int curX = startX;
    int hitChar = -1;
    int charStartX = 0;
    bool hitIsSubscript = false;

    for (int i = 0; i < textLen; i++) {
        bool isSub = shouldSubscript(text, textLen, i);
        int cw = isSub ? subCharWidth : (6 * scale);

        if (x >= curX && x < curX + cw) {
            hitChar = i;
            charStartX = curX;
            hitIsSubscript = isSub;
            break;
        }
        curX += cw;
    }

    if (hitChar < 0) return;

    // Check Y bounds
    int charY, charH, charScale;
    if (hitIsSubscript) {
        charY = startY + subOffsetY;
        charH = subCharHeight;
        charScale = subScale;
    } else {
        charY = startY;
        charH = charHeight;
        charScale = scale;
    }

    if (y < charY || y >= charY + charH) return;

    // Find pixel within character
    int relX = x - charStartX;
    int relY = y - charY;
    int pixelX = relX / charScale;
    int pixelY = relY / charScale;

    unsigned char c = text[hitChar];
    if (getPixel(c, pixelX, pixelY)) {
        int idx = (y * width + x) * 4;
        // White text with slight transparency for subscripts
        pixels[idx + 0] = 255;  // B
        pixels[idx + 1] = 255;  // G
        pixels[idx + 2] = 255;  // R
        pixels[idx + 3] = 255;  // A
    }
}

__global__ void renderMoleculeKernel(
    unsigned char* pixels, int width, int height,
    Atom* atoms, int numAtoms,
    Bond* bonds, int numBonds,
    float rotX, float rotY, float zoom)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Camera setup
    float aspectRatio = (float)width / height;
    float fovScale = tanf(0.5f * 0.8f);

    float ndcX = (2.0f * x / width - 1.0f) * aspectRatio * fovScale;
    float ndcY = (1.0f - 2.0f * y / height) * fovScale;

    // Ray direction
    float3 rd = normalize3(make_float3(ndcX, ndcY, 1.0f));

    // Rotate ray
    float cosY = cosf(rotY), sinY = sinf(rotY);
    float cosX = cosf(rotX), sinX = sinf(rotX);

    float rx = rd.x * cosY + rd.z * sinY;
    float rz = -rd.x * sinY + rd.z * cosY;
    rd.x = rx; rd.z = rz;

    float ry = rd.y * cosX - rd.z * sinX;
    rz = rd.y * sinX + rd.z * cosX;
    rd.y = ry; rd.z = rz;

    // Ray origin
    float3 ro = make_float3(0, 0, -zoom);

    // Rotate camera
    float cox = ro.x * cosY + ro.z * sinY;
    float coz = -ro.x * sinY + ro.z * cosY;
    ro.x = cox; ro.z = coz;

    float coy = ro.y * cosX - ro.z * sinX;
    coz = ro.y * sinX + ro.z * cosX;
    ro.y = coy; ro.z = coz;

    // Lighting
    float3 lightDir = normalize3(make_float3(0.5f, 0.8f, -0.3f));
    float3 lightDir2 = normalize3(make_float3(-0.3f, 0.2f, 0.5f));

    float minT = 1e10f;
    float3 hitNormal;
    float3 hitColor;
    int hitType = 0;  // 0=none, 1=atom, 2=bond

    // Test atoms (spheres)
    for (int i = 0; i < numAtoms; i++) {
        float3 center = make_float3(atoms[i].x, atoms[i].y, atoms[i].z);
        float radius = atoms[i].radius;

        float t = intersectSphere(ro, rd, center, radius);
        if (t > 0.0f && t < minT) {
            minT = t;
            float3 hitPos = make_float3(ro.x + t * rd.x, ro.y + t * rd.y, ro.z + t * rd.z);
            hitNormal = normalize3(make_float3(hitPos.x - center.x, hitPos.y - center.y, hitPos.z - center.z));
            hitColor = atomColors[atoms[i].type];
            hitType = 1;
        }
    }

    // Test bonds (cylinders)
    float bondRadius = 0.08f;
    for (int i = 0; i < numBonds; i++) {
        int a1 = bonds[i].atom1;
        int a2 = bonds[i].atom2;

        float3 p1 = make_float3(atoms[a1].x, atoms[a1].y, atoms[a1].z);
        float3 p2 = make_float3(atoms[a2].x, atoms[a2].y, atoms[a2].z);

        // For double/triple bonds, offset cylinders
        int order = bonds[i].order;
        float3 bondDir = normalize3(make_float3(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z));

        // Find perpendicular vector
        float3 perp;
        if (fabsf(bondDir.x) < 0.9f) {
            perp = normalize3(make_float3(0, -bondDir.z, bondDir.y));
        } else {
            perp = normalize3(make_float3(-bondDir.z, 0, bondDir.x));
        }

        for (int b = 0; b < order; b++) {
            float3 offset = make_float3(0, 0, 0);
            if (order == 2) {
                float off = (b == 0) ? -0.08f : 0.08f;
                offset = make_float3(perp.x * off, perp.y * off, perp.z * off);
            } else if (order == 3) {
                float off = (b - 1) * 0.1f;
                offset = make_float3(perp.x * off, perp.y * off, perp.z * off);
            }

            float3 bp1 = make_float3(p1.x + offset.x, p1.y + offset.y, p1.z + offset.z);
            float3 bp2 = make_float3(p2.x + offset.x, p2.y + offset.y, p2.z + offset.z);

            float3 cylNormal;
            float t = intersectCylinder(ro, rd, bp1, bp2, bondRadius, &cylNormal);
            if (t > 0.0f && t < minT) {
                minT = t;
                hitNormal = cylNormal;
                hitColor = make_float3(0.6f, 0.6f, 0.6f);  // Gray bonds
                hitType = 2;
            }
        }
    }

    int idx = (y * width + x) * 4;

    if (hitType > 0) {
        // Phong shading with two lights
        float diffuse1 = fmaxf(0.0f, dot3(hitNormal, lightDir));
        float diffuse2 = fmaxf(0.0f, dot3(hitNormal, lightDir2)) * 0.3f;

        float3 viewDir = normalize3(make_float3(-rd.x, -rd.y, -rd.z));
        float3 reflectDir = reflect3(make_float3(-lightDir.x, -lightDir.y, -lightDir.z), hitNormal);
        float spec = powf(fmaxf(0.0f, dot3(viewDir, reflectDir)), 40.0f);

        float ambient = 0.15f;
        float3 finalColor;

        if (hitType == 1) {
            // Atom - glossy plastic look
            finalColor.x = hitColor.x * (ambient + diffuse1 * 0.7f + diffuse2) + spec * 0.6f;
            finalColor.y = hitColor.y * (ambient + diffuse1 * 0.7f + diffuse2) + spec * 0.6f;
            finalColor.z = hitColor.z * (ambient + diffuse1 * 0.7f + diffuse2) + spec * 0.6f;
        } else {
            // Bond - more metallic
            finalColor.x = hitColor.x * (ambient + diffuse1 * 0.5f + diffuse2) + spec * 0.4f;
            finalColor.y = hitColor.y * (ambient + diffuse1 * 0.5f + diffuse2) + spec * 0.4f;
            finalColor.z = hitColor.z * (ambient + diffuse1 * 0.5f + diffuse2) + spec * 0.4f;
        }

        // Fresnel rim lighting
        float fresnel = powf(1.0f - fmaxf(0.0f, dot3(viewDir, hitNormal)), 3.0f);
        finalColor.x += fresnel * 0.15f;
        finalColor.y += fresnel * 0.15f;
        finalColor.z += fresnel * 0.2f;

        pixels[idx + 0] = (unsigned char)fminf(255.0f, finalColor.z * 255.0f);
        pixels[idx + 1] = (unsigned char)fminf(255.0f, finalColor.y * 255.0f);
        pixels[idx + 2] = (unsigned char)fminf(255.0f, finalColor.x * 255.0f);
    }
}

// ============== MOLECULE BUILDERS ==============

void addAtom(Molecule* mol, float x, float y, float z, int type) {
    if (mol->numAtoms >= MAX_ATOMS) return;
    Atom* a = &mol->atoms[mol->numAtoms];
    a->x = x;
    a->y = y;
    a->z = z;
    a->type = type;

    // Set radius based on type
    float radii[] = {0.25f, 0.40f, 0.38f, 0.35f, 0.45f, 0.45f, 0.45f, 0.50f, 0.35f, 0.55f};
    a->radius = radii[type];

    mol->numAtoms++;
}

void addBond(Molecule* mol, int a1, int a2, int order) {
    if (mol->numBonds >= MAX_BONDS) return;
    mol->bonds[mol->numBonds].atom1 = a1;
    mol->bonds[mol->numBonds].atom2 = a2;
    mol->bonds[mol->numBonds].order = order;
    mol->numBonds++;
}

// Center molecule at origin
void centerMolecule(Molecule* mol) {
    float cx = 0, cy = 0, cz = 0;
    for (int i = 0; i < mol->numAtoms; i++) {
        cx += mol->atoms[i].x;
        cy += mol->atoms[i].y;
        cz += mol->atoms[i].z;
    }
    cx /= mol->numAtoms;
    cy /= mol->numAtoms;
    cz /= mol->numAtoms;

    for (int i = 0; i < mol->numAtoms; i++) {
        mol->atoms[i].x -= cx;
        mol->atoms[i].y -= cy;
        mol->atoms[i].z -= cz;
    }
}

// Build Water (H2O)
void buildWater(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Water (H2O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.76f, 0.59f, 0.0f, ATOM_H);
    addAtom(mol, -0.76f, 0.59f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);

    centerMolecule(mol);
}

// Build Methane (CH4)
void buildMethane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Methane (CH4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.63f, 0.63f, 0.63f, ATOM_H);
    addAtom(mol, -0.63f, -0.63f, 0.63f, ATOM_H);
    addAtom(mol, -0.63f, 0.63f, -0.63f, ATOM_H);
    addAtom(mol, 0.63f, -0.63f, -0.63f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);

    centerMolecule(mol);
}

// Build Benzene (C6H6)
void buildBenzene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Benzene (C6H6)");

    float r = 1.4f;
    float rH = 2.2f;

    // Carbon ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // Hydrogen atoms
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // C-C bonds (alternating single/double for aromatic)
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }

    // C-H bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, i + 6, 1);
    }

    centerMolecule(mol);
}

// Build Ethanol (C2H5OH)
void buildEthanol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ethanol (C2H5OH)");

    // Carbons
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // C2

    // Oxygen
    addAtom(mol, 2.2f, 1.1f, 0.0f, ATOM_O);      // O

    // Hydrogens on C1
    addAtom(mol, -0.5f, 0.9f, 0.3f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.8f, ATOM_H);
    addAtom(mol, -0.5f, -0.4f, -0.9f, ATOM_H);

    // Hydrogens on C2
    addAtom(mol, 2.0f, -0.5f, 0.85f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, -0.85f, ATOM_H);

    // Hydrogen on O
    addAtom(mol, 3.1f, 1.0f, 0.0f, ATOM_H);

    // Bonds
    addBond(mol, 0, 1, 1);  // C-C
    addBond(mol, 1, 2, 1);  // C-O
    addBond(mol, 0, 3, 1);  // C-H
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 2, 8, 1);  // O-H

    centerMolecule(mol);
}

// Build Caffeine (C8H10N4O2)
void buildCaffeine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Caffeine (C8H10N4O2)");

    // Purine ring system (approximate coordinates)
    // Imidazole ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);       // N1
    addAtom(mol, 1.2f, 0.5f, 0.0f, ATOM_C);       // C2
    addAtom(mol, 1.2f, 1.9f, 0.0f, ATOM_N);       // N3
    addAtom(mol, 0.0f, 2.4f, 0.0f, ATOM_C);       // C4
    addAtom(mol, -0.8f, 1.2f, 0.0f, ATOM_C);      // C5

    // Pyrimidine ring
    addAtom(mol, -0.8f, 3.6f, 0.0f, ATOM_N);      // N6
    addAtom(mol, 0.0f, 4.8f, 0.0f, ATOM_C);       // C7
    addAtom(mol, 1.4f, 4.6f, 0.0f, ATOM_N);       // N8
    addAtom(mol, 1.8f, 3.3f, 0.0f, ATOM_C);       // C9

    // Carbonyl oxygens
    addAtom(mol, 2.3f, -0.2f, 0.0f, ATOM_O);      // O1
    addAtom(mol, -0.4f, 5.9f, 0.0f, ATOM_O);      // O2

    // Methyl groups
    addAtom(mol, -0.5f, -1.3f, 0.0f, ATOM_C);     // CH3 on N1
    addAtom(mol, -2.0f, 3.8f, 0.0f, ATOM_C);      // CH3 on N6
    addAtom(mol, 2.0f, 5.8f, 0.0f, ATOM_C);       // CH3 on N8

    // Hydrogens on imidazole
    addAtom(mol, -1.8f, 1.0f, 0.0f, ATOM_H);      // H on C5
    addAtom(mol, 3.0f, 3.2f, 0.0f, ATOM_H);       // H on C9

    // Hydrogens on methyl groups (simplified - 3 each)
    addAtom(mol, -1.5f, -1.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.1f, -1.8f, 0.8f, ATOM_H);
    addAtom(mol, -0.1f, -1.8f, -0.8f, ATOM_H);

    addAtom(mol, -2.3f, 4.8f, 0.0f, ATOM_H);
    addAtom(mol, -2.5f, 3.3f, 0.8f, ATOM_H);
    addAtom(mol, -2.5f, 3.3f, -0.8f, ATOM_H);

    addAtom(mol, 1.5f, 6.7f, 0.0f, ATOM_H);
    addAtom(mol, 2.5f, 5.8f, 0.9f, ATOM_H);
    addAtom(mol, 2.5f, 5.8f, -0.9f, ATOM_H);

    // Bonds - rings
    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 4, 0, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 2, 1);

    // Carbonyl bonds
    addBond(mol, 1, 9, 2);
    addBond(mol, 6, 10, 2);

    // Methyl bonds
    addBond(mol, 0, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 7, 13, 1);

    // C-H bonds
    addBond(mol, 4, 14, 1);
    addBond(mol, 8, 15, 1);

    // Methyl H bonds
    addBond(mol, 11, 16, 1);
    addBond(mol, 11, 17, 1);
    addBond(mol, 11, 18, 1);
    addBond(mol, 12, 19, 1);
    addBond(mol, 12, 20, 1);
    addBond(mol, 12, 21, 1);
    addBond(mol, 13, 22, 1);
    addBond(mol, 13, 23, 1);
    addBond(mol, 13, 24, 1);

    centerMolecule(mol);
}

// Build Adenine (DNA base)
void buildAdenine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Adenine (DNA base)");

    // Purine ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);       // N1
    addAtom(mol, 1.3f, 0.3f, 0.0f, ATOM_C);       // C2
    addAtom(mol, 1.8f, 1.6f, 0.0f, ATOM_N);       // N3
    addAtom(mol, 0.9f, 2.6f, 0.0f, ATOM_C);       // C4
    addAtom(mol, -0.4f, 2.3f, 0.0f, ATOM_C);      // C5
    addAtom(mol, -0.8f, 1.0f, 0.0f, ATOM_C);      // C6

    // Imidazole ring
    addAtom(mol, 1.2f, 3.9f, 0.0f, ATOM_N);       // N7
    addAtom(mol, 0.0f, 4.5f, 0.0f, ATOM_C);       // C8
    addAtom(mol, -1.0f, 3.5f, 0.0f, ATOM_N);      // N9

    // Amino group
    addAtom(mol, -2.1f, 0.7f, 0.0f, ATOM_N);      // NH2

    // Hydrogens
    addAtom(mol, 2.0f, -0.4f, 0.0f, ATOM_H);      // H on C2
    addAtom(mol, -0.2f, 5.5f, 0.0f, ATOM_H);      // H on C8
    addAtom(mol, -1.9f, 3.8f, 0.0f, ATOM_H);      // H on N9
    addAtom(mol, -2.6f, 1.5f, 0.0f, ATOM_H);      // H on NH2
    addAtom(mol, -2.6f, -0.1f, 0.0f, ATOM_H);     // H on NH2

    // Bonds
    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 4, 1);
    addBond(mol, 5, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 8, 12, 1);
    addBond(mol, 9, 13, 1);
    addBond(mol, 9, 14, 1);

    centerMolecule(mol);
}

// Build Glucose (C6H12O6)
void buildGlucose(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Glucose (C6H12O6)");

    // Pyranose ring (chair conformation, simplified)
    float r = 1.4f;

    // Ring carbons and oxygen
    addAtom(mol, r, 0.0f, 0.3f, ATOM_C);          // C1
    addAtom(mol, r * 0.5f, r * 0.866f, -0.3f, ATOM_C);   // C2
    addAtom(mol, -r * 0.5f, r * 0.866f, 0.3f, ATOM_C);   // C3
    addAtom(mol, -r, 0.0f, -0.3f, ATOM_C);        // C4
    addAtom(mol, -r * 0.5f, -r * 0.866f, 0.3f, ATOM_C);  // C5
    addAtom(mol, r * 0.5f, -r * 0.866f, -0.3f, ATOM_O);  // Ring O

    // C6 (CH2OH group)
    addAtom(mol, -r * 0.9f, -r * 1.5f, 0.0f, ATOM_C);    // C6

    // OH groups
    addAtom(mol, r * 1.5f, 0.3f, 1.0f, ATOM_O);          // O on C1
    addAtom(mol, r * 0.9f, r * 1.4f, -1.0f, ATOM_O);     // O on C2
    addAtom(mol, -r * 0.9f, r * 1.4f, 1.0f, ATOM_O);     // O on C3
    addAtom(mol, -r * 1.5f, 0.0f, -1.0f, ATOM_O);        // O on C4
    addAtom(mol, -r * 0.5f, -r * 2.3f, 0.0f, ATOM_O);    // O on C6

    // Hydrogens (simplified - one per carbon/oxygen)
    addAtom(mol, r * 1.3f, -0.5f, -0.5f, ATOM_H);        // H on C1
    addAtom(mol, r * 0.8f, r * 0.5f, 0.5f, ATOM_H);      // H on C2
    addAtom(mol, -r * 0.2f, r * 1.1f, -0.5f, ATOM_H);    // H on C3
    addAtom(mol, -r * 0.7f, -0.3f, 0.5f, ATOM_H);        // H on C4
    addAtom(mol, -r * 0.8f, -r * 0.6f, -0.5f, ATOM_H);   // H on C5
    addAtom(mol, -r * 1.5f, -r * 1.3f, 0.7f, ATOM_H);    // H on C6
    addAtom(mol, -r * 1.3f, -r * 1.5f, -0.8f, ATOM_H);   // H on C6

    // OH hydrogens
    addAtom(mol, r * 2.2f, 0.0f, 1.3f, ATOM_H);
    addAtom(mol, r * 1.5f, r * 1.2f, -1.5f, ATOM_H);
    addAtom(mol, -r * 1.5f, r * 1.2f, 1.5f, ATOM_H);
    addAtom(mol, -r * 2.2f, 0.0f, -1.3f, ATOM_H);
    addAtom(mol, -r * 0.9f, -r * 2.8f, 0.5f, ATOM_H);

    // Ring bonds
    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);

    // C6 bond
    addBond(mol, 4, 6, 1);

    // C-O bonds
    addBond(mol, 0, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 3, 10, 1);
    addBond(mol, 6, 11, 1);

    // C-H bonds
    addBond(mol, 0, 12, 1);
    addBond(mol, 1, 13, 1);
    addBond(mol, 2, 14, 1);
    addBond(mol, 3, 15, 1);
    addBond(mol, 4, 16, 1);
    addBond(mol, 6, 17, 1);
    addBond(mol, 6, 18, 1);

    // O-H bonds
    addBond(mol, 7, 19, 1);
    addBond(mol, 8, 20, 1);
    addBond(mol, 9, 21, 1);
    addBond(mol, 10, 22, 1);
    addBond(mol, 11, 23, 1);

    centerMolecule(mol);
}

// Build Aspirin (C9H8O4)
void buildAspirin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Aspirin/Bayer (C9H8O4)");

    float r = 1.4f;

    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // Carboxylic acid group (COOH)
    addAtom(mol, 2.5f, 0.5f, 0.0f, ATOM_C);       // C7
    addAtom(mol, 3.2f, 1.5f, 0.0f, ATOM_O);       // O1 (=O)
    addAtom(mol, 3.0f, -0.6f, 0.0f, ATOM_O);      // O2 (OH)

    // Acetyl group (OCOCH3)
    addAtom(mol, r * cosf(PI/3.0f) - 0.8f, r * sinf(PI/3.0f) + 0.8f, 0.0f, ATOM_O);  // O3
    addAtom(mol, r * cosf(PI/3.0f) - 1.0f, r * sinf(PI/3.0f) + 2.2f, 0.0f, ATOM_C);  // C8
    addAtom(mol, r * cosf(PI/3.0f) - 2.3f, r * sinf(PI/3.0f) + 2.8f, 0.0f, ATOM_O);  // O4 (=O)
    addAtom(mol, r * cosf(PI/3.0f) + 0.2f, r * sinf(PI/3.0f) + 3.2f, 0.0f, ATOM_C);  // C9 (CH3)

    // Hydrogens on benzene (4 of them, positions 2,3,4,5)
    float rH = 2.4f;
    addAtom(mol, rH * cosf(2*PI/3.0f), rH * sinf(2*PI/3.0f), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(PI), rH * sinf(PI), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(4*PI/3.0f), rH * sinf(4*PI/3.0f), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(5*PI/3.0f), rH * sinf(5*PI/3.0f), 0.0f, ATOM_H);

    // H on COOH
    addAtom(mol, 3.8f, -0.5f, 0.0f, ATOM_H);

    // H on CH3
    addAtom(mol, r * cosf(PI/3.0f) + 0.0f, r * sinf(PI/3.0f) + 4.1f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(PI/3.0f) + 0.9f, r * sinf(PI/3.0f) + 2.8f, 0.7f, ATOM_H);
    addAtom(mol, r * cosf(PI/3.0f) + 0.9f, r * sinf(PI/3.0f) + 2.8f, -0.7f, ATOM_H);

    // Benzene bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }

    // COOH bonds
    addBond(mol, 0, 6, 1);    // C1-C7
    addBond(mol, 6, 7, 2);    // C7=O1
    addBond(mol, 6, 8, 1);    // C7-O2

    // Acetyl bonds
    addBond(mol, 1, 9, 1);    // C2-O3
    addBond(mol, 9, 10, 1);   // O3-C8
    addBond(mol, 10, 11, 2);  // C8=O4
    addBond(mol, 10, 12, 1);  // C8-C9

    // C-H bonds
    addBond(mol, 2, 13, 1);
    addBond(mol, 3, 14, 1);
    addBond(mol, 4, 15, 1);
    addBond(mol, 5, 16, 1);
    addBond(mol, 8, 17, 1);
    addBond(mol, 12, 18, 1);
    addBond(mol, 12, 19, 1);
    addBond(mol, 12, 20, 1);

    centerMolecule(mol);
}

// Build Ammonia (NH3)
void buildAmmonia(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ammonia (NH3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 0.94f, 0.0f, 0.34f, ATOM_H);
    addAtom(mol, -0.47f, 0.81f, 0.34f, ATOM_H);
    addAtom(mol, -0.47f, -0.81f, 0.34f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);

    centerMolecule(mol);
}

// Build Carbon Dioxide (CO2)
void buildCarbonDioxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Carbon Dioxide (CO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -1.16f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 1.16f, 0.0f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 2);

    centerMolecule(mol);
}

// Build Formaldehyde (CH2O)
void buildFormaldehyde(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Formaldehyde (CH2O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_O);
    addAtom(mol, 0.93f, -0.54f, 0.0f, ATOM_H);
    addAtom(mol, -0.93f, -0.54f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);

    centerMolecule(mol);
}

// Build Acetone (C3H6O)
void buildAcetone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetone (C3H6O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_O);      // =O
    addAtom(mol, -1.4f, -0.5f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 1.4f, -0.5f, 0.0f, ATOM_C);     // CH3

    // Hydrogens on CH3 groups
    addAtom(mol, -1.4f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, -0.1f, 0.8f, ATOM_H);
    addAtom(mol, -2.1f, -0.1f, -0.8f, ATOM_H);
    addAtom(mol, 1.4f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, 2.1f, -0.1f, 0.8f, ATOM_H);
    addAtom(mol, 2.1f, -0.1f, -0.8f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 3, 7, 1);
    addBond(mol, 3, 8, 1);
    addBond(mol, 3, 9, 1);

    centerMolecule(mol);
}

// Build Acetic Acid (CH3COOH)
void buildAceticAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetic Acid (CH3COOH)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 2.1f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.1f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 3.0f, -1.1f, 0.0f, ATOM_H);     // H on OH
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 1, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 0, 7, 1);

    centerMolecule(mol);
}

// Build Propane (C3H8)
void buildPropane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Propane (C3H8)");

    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);

    // H on C1
    addAtom(mol, -2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, -0.87f, ATOM_H);
    // H on C2
    addAtom(mol, 0.0f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 0.0f, 0.6f, -0.9f, ATOM_H);
    // H on C3
    addAtom(mol, 2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 2, 10, 1);

    centerMolecule(mol);
}

// Build Butane (C4H10)
void buildButane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Butane (C4H10)");

    addAtom(mol, -2.25f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -0.75f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.75f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.25f, 0.0f, 0.0f, ATOM_C);

    // Hydrogens
    addAtom(mol, -2.75f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.75f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.75f, -0.5f, -0.87f, ATOM_H);
    addAtom(mol, -0.75f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, -0.75f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 0.75f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 0.75f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 2.75f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 2.75f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 2.75f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 2, 10, 1);
    addBond(mol, 3, 11, 1);
    addBond(mol, 3, 12, 1);
    addBond(mol, 3, 13, 1);

    centerMolecule(mol);
}

// Build Cyclohexane (C6H12)
void buildCyclohexane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cyclohexane (C6H12)");

    float r = 1.4f;
    // Chair conformation
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        float z = (i % 2 == 0) ? 0.3f : -0.3f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), z, ATOM_C);
    }

    // Axial and equatorial hydrogens
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        float z = (i % 2 == 0) ? 0.3f : -0.3f;
        // Axial H
        addAtom(mol, r * cosf(angle) * 0.6f, r * sinf(angle) * 0.6f, z + ((i % 2 == 0) ? 1.0f : -1.0f), ATOM_H);
        // Equatorial H
        addAtom(mol, (r + 0.9f) * cosf(angle), (r + 0.9f) * sinf(angle), z, ATOM_H);
    }

    // C-C bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, 1);
    }
    // C-H bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, 6 + i * 2, 1);
        addBond(mol, i, 6 + i * 2 + 1, 1);
    }

    centerMolecule(mol);
}

// Build Naphthalene (C10H8)
void buildNaphthalene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Naphthalene (C10H8)");

    // Two fused benzene rings
    float dx = 1.2f;
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, dx, 0.7f, 0.0f, ATOM_C);        // C2
    addAtom(mol, 2*dx, 0.0f, 0.0f, ATOM_C);      // C3
    addAtom(mol, 2*dx, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, dx, -2.1f, 0.0f, ATOM_C);       // C5
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6
    // Second ring
    addAtom(mol, 3*dx, 0.7f, 0.0f, ATOM_C);      // C7
    addAtom(mol, 4*dx, 0.0f, 0.0f, ATOM_C);      // C8
    addAtom(mol, 4*dx, -1.4f, 0.0f, ATOM_C);     // C9
    addAtom(mol, 3*dx, -2.1f, 0.0f, ATOM_C);     // C10

    // Hydrogens
    addAtom(mol, -0.9f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, dx, 1.7f, 0.0f, ATOM_H);
    addAtom(mol, dx, -3.1f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.9f, 0.0f, ATOM_H);
    addAtom(mol, 3*dx, 1.7f, 0.0f, ATOM_H);
    addAtom(mol, 4*dx + 0.9f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 4*dx + 0.9f, -1.9f, 0.0f, ATOM_H);
    addAtom(mol, 3*dx, -3.1f, 0.0f, ATOM_H);

    // Bonds - first ring
    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    // Second ring
    addBond(mol, 2, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 3, 1);
    // C-H bonds
    addBond(mol, 0, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 5, 13, 1);
    addBond(mol, 6, 14, 1);
    addBond(mol, 7, 15, 1);
    addBond(mol, 8, 16, 1);
    addBond(mol, 9, 17, 1);

    centerMolecule(mol);
}

// Build Urea (CH4N2O)
void buildUrea(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Urea (CH4N2O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 1.25f, 0.0f, ATOM_O);
    addAtom(mol, -1.2f, -0.6f, 0.0f, ATOM_N);
    addAtom(mol, 1.2f, -0.6f, 0.0f, ATOM_N);
    addAtom(mol, -1.3f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.1f, 0.0f, ATOM_H);
    addAtom(mol, 1.3f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -0.1f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 3, 7, 1);

    centerMolecule(mol);
}

// Build Glycine (C2H5NO2) - simplest amino acid
void buildGlycine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Glycine (C2H5NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha carbon
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // Amino
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // Carboxyl C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -0.6f, 0.9f, ATOM_H);
    addAtom(mol, 0.0f, -0.6f, -0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 4, 9, 1);

    centerMolecule(mol);
}

// Build Alanine (C3H7NO2)
void buildAlanine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Alanine (C3H7NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha carbon
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // Amino
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // Carboxyl C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, -2.1f, 0.9f, ATOM_H);
    addAtom(mol, 0.87f, -2.1f, -0.45f, ATOM_H);
    addAtom(mol, -0.87f, -2.1f, -0.45f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 5, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 5, 12, 1);

    centerMolecule(mol);
}

// Build Thymine (DNA base)
void buildThymine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Thymine (DNA base)");

    // Pyrimidine ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_N);      // N3
    addAtom(mol, 0.0f, 2.8f, 0.0f, ATOM_C);      // C4
    addAtom(mol, -1.2f, 2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_C);     // C6

    // Carbonyls
    addAtom(mol, 2.3f, 0.1f, 0.0f, ATOM_O);      // O on C2
    addAtom(mol, 0.0f, 4.1f, 0.0f, ATOM_O);      // O on C4

    // Methyl on C5
    addAtom(mol, -2.4f, 2.8f, 0.0f, ATOM_C);     // CH3

    // Hydrogens
    addAtom(mol, 0.0f, -1.0f, 0.0f, ATOM_H);     // H on N1
    addAtom(mol, 2.1f, 2.5f, 0.0f, ATOM_H);      // H on N3
    addAtom(mol, -2.1f, 0.1f, 0.0f, ATOM_H);     // H on C6
    addAtom(mol, -2.4f, 3.9f, 0.0f, ATOM_H);
    addAtom(mol, -3.1f, 2.4f, 0.8f, ATOM_H);
    addAtom(mol, -3.1f, 2.4f, -0.8f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 1, 6, 2);
    addBond(mol, 3, 7, 2);
    addBond(mol, 4, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 2, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 8, 12, 1);
    addBond(mol, 8, 13, 1);
    addBond(mol, 8, 14, 1);

    centerMolecule(mol);
}

// Build Cytosine (DNA base)
void buildCytosine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cytosine (DNA base)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_N);      // N3
    addAtom(mol, 0.0f, 2.8f, 0.0f, ATOM_C);      // C4
    addAtom(mol, -1.2f, 2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_C);     // C6

    addAtom(mol, 2.3f, 0.1f, 0.0f, ATOM_O);      // O on C2
    addAtom(mol, 0.0f, 4.1f, 0.0f, ATOM_N);      // NH2 on C4

    addAtom(mol, 0.0f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 2.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.1f, 0.0f, ATOM_H);
    addAtom(mol, -0.8f, 4.6f, 0.0f, ATOM_H);
    addAtom(mol, 0.8f, 4.6f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 1, 6, 2);
    addBond(mol, 3, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 5, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 7, 12, 1);

    centerMolecule(mol);
}

// Build Guanine (DNA base)
void buildGuanine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Guanine (DNA base)");

    // Purine ring system
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.3f, 0.3f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 1.8f, 1.6f, 0.0f, ATOM_N);      // N3
    addAtom(mol, 0.9f, 2.6f, 0.0f, ATOM_C);      // C4
    addAtom(mol, -0.4f, 2.3f, 0.0f, ATOM_C);     // C5
    addAtom(mol, -0.8f, 1.0f, 0.0f, ATOM_C);     // C6

    addAtom(mol, 1.2f, 3.9f, 0.0f, ATOM_N);      // N7
    addAtom(mol, 0.0f, 4.5f, 0.0f, ATOM_C);      // C8
    addAtom(mol, -1.0f, 3.5f, 0.0f, ATOM_N);     // N9

    addAtom(mol, 2.0f, -0.6f, 0.0f, ATOM_N);     // NH2 on C2
    addAtom(mol, -2.0f, 0.6f, 0.0f, ATOM_O);     // O on C6

    addAtom(mol, -0.3f, -0.9f, 0.0f, ATOM_H);
    addAtom(mol, -0.2f, 5.5f, 0.0f, ATOM_H);
    addAtom(mol, -1.9f, 3.8f, 0.0f, ATOM_H);
    addAtom(mol, 1.6f, -1.5f, 0.0f, ATOM_H);
    addAtom(mol, 2.9f, -0.4f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 4, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 5, 10, 2);
    addBond(mol, 0, 11, 1);
    addBond(mol, 7, 12, 1);
    addBond(mol, 8, 13, 1);
    addBond(mol, 9, 14, 1);
    addBond(mol, 9, 15, 1);

    centerMolecule(mol);
}

// Build Dopamine (C8H11NO2)
void buildDopamine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Dopamine (C8H11NO2)");

    float r = 1.4f;
    // Catechol ring (benzene with 2 OH)
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // OH groups on C3 and C4
    addAtom(mol, r * cosf(PI) - 0.8f, r * sinf(PI) + 0.6f, 0.0f, ATOM_O);
    addAtom(mol, r * cosf(4*PI/3) - 0.6f, r * sinf(4*PI/3) - 0.8f, 0.0f, ATOM_O);

    // Ethylamine chain from C1
    addAtom(mol, 2.5f, 0.5f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 3.8f, -0.2f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 5.0f, 0.5f, 0.0f, ATOM_N);      // NH2

    // Hydrogens
    float rH = 2.4f;
    addAtom(mol, rH * cosf(PI/3), rH * sinf(PI/3), 0.0f, ATOM_H);   // H on C2
    addAtom(mol, rH * cosf(5*PI/3), rH * sinf(5*PI/3), 0.0f, ATOM_H);// H on C6
    addAtom(mol, r * cosf(PI) - 1.6f, r * sinf(PI) + 0.3f, 0.0f, ATOM_H);  // H on OH
    addAtom(mol, r * cosf(4*PI/3) - 0.3f, r * sinf(4*PI/3) - 1.6f, 0.0f, ATOM_H);  // H on OH
    addAtom(mol, 2.5f, 1.1f, 0.9f, ATOM_H);
    addAtom(mol, 2.5f, 1.1f, -0.9f, ATOM_H);
    addAtom(mol, 3.8f, -0.8f, 0.9f, ATOM_H);
    addAtom(mol, 3.8f, -0.8f, -0.9f, ATOM_H);
    addAtom(mol, 5.0f, 1.1f, 0.8f, ATOM_H);
    addAtom(mol, 5.8f, 0.0f, 0.0f, ATOM_H);

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    // OH bonds
    addBond(mol, 3, 6, 1);
    addBond(mol, 4, 7, 1);
    // Chain bonds
    addBond(mol, 0, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 1);
    // H bonds
    addBond(mol, 1, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 6, 13, 1);
    addBond(mol, 7, 14, 1);
    addBond(mol, 8, 15, 1);
    addBond(mol, 8, 16, 1);
    addBond(mol, 9, 17, 1);
    addBond(mol, 9, 18, 1);
    addBond(mol, 10, 19, 1);
    addBond(mol, 10, 20, 1);

    centerMolecule(mol);
}

// Build Serotonin (C10H12N2O)
void buildSerotonin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Serotonin (C10H12N2O)");

    // Indole ring system
    float r = 1.3f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f + PI/6;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    // Pyrrole ring (fused)
    addAtom(mol, 2.3f, 0.7f, 0.0f, ATOM_C);      // C7
    addAtom(mol, 2.8f, -0.5f, 0.0f, ATOM_C);     // C8
    addAtom(mol, 1.9f, -1.3f, 0.0f, ATOM_N);     // N

    // OH on C5 position
    addAtom(mol, r * cosf(5*PI/3 + PI/6) + 0.9f, r * sinf(5*PI/3 + PI/6) - 0.5f, 0.0f, ATOM_O);

    // Ethylamine chain
    addAtom(mol, 4.1f, -0.8f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 5.0f, 0.3f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 6.3f, 0.0f, 0.0f, ATOM_N);      // NH2

    // Hydrogens (simplified)
    addAtom(mol, r * cosf(PI/3 + PI/6) * 1.7f, r * sinf(PI/3 + PI/6) * 1.7f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/3 + PI/6) * 1.7f, r * sinf(2*PI/3 + PI/6) * 1.7f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(PI + PI/6) * 1.7f, r * sinf(PI + PI/6) * 1.7f, 0.0f, ATOM_H);
    addAtom(mol, 2.5f, 1.7f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -2.3f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(5*PI/3 + PI/6) + 1.7f, r * sinf(5*PI/3 + PI/6) - 0.2f, 0.0f, ATOM_H);
    addAtom(mol, 4.3f, -1.4f, 0.9f, ATOM_H);
    addAtom(mol, 4.3f, -1.4f, -0.9f, ATOM_H);
    addAtom(mol, 4.8f, 0.9f, 0.9f, ATOM_H);
    addAtom(mol, 4.8f, 0.9f, -0.9f, ATOM_H);
    addAtom(mol, 6.5f, -0.5f, 0.8f, ATOM_H);
    addAtom(mol, 7.0f, 0.7f, 0.0f, ATOM_H);

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 5, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 7, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 1);
    // H bonds
    addBond(mol, 1, 13, 1);
    addBond(mol, 2, 14, 1);
    addBond(mol, 3, 15, 1);
    addBond(mol, 6, 16, 1);
    addBond(mol, 8, 17, 1);
    addBond(mol, 9, 18, 1);
    addBond(mol, 10, 19, 1);
    addBond(mol, 10, 20, 1);
    addBond(mol, 11, 21, 1);
    addBond(mol, 11, 22, 1);
    addBond(mol, 12, 23, 1);
    addBond(mol, 12, 24, 1);

    centerMolecule(mol);
}

// Build Nitric Oxide (NO)
void buildNitricOxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nitric Oxide (NO)");

    addAtom(mol, -0.58f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 0.58f, 0.0f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);

    centerMolecule(mol);
}

// Build Hydrogen Peroxide (H2O2)
void buildHydrogenPeroxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Hydrogen Peroxide (H2O2)");

    addAtom(mol, -0.7f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.7f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, -1.2f, 0.8f, 0.3f, ATOM_H);
    addAtom(mol, 1.2f, -0.8f, -0.3f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 1, 3, 1);

    centerMolecule(mol);
}

// Build Sulfuric Acid (H2SO4)
void buildSulfuricAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Sulfuric Acid (H2SO4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_S);
    addAtom(mol, 0.0f, 1.4f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, -1.4f, 0.0f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 2.1f, 0.7f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, -0.7f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 4, 6, 1);

    centerMolecule(mol);
}

// Build Phosphoric Acid (H3PO4)
void buildPhosphoricAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Phosphoric Acid (H3PO4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_P);
    addAtom(mol, 0.0f, 1.5f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 1.3f, -0.75f, 0.0f, ATOM_O);    // OH
    addAtom(mol, -1.3f, -0.75f, 0.0f, ATOM_O);   // OH
    addAtom(mol, 0.0f, -0.5f, 1.4f, ATOM_O);     // OH
    addAtom(mol, 2.0f, -0.3f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.3f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, 0.0f, 2.1f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 4, 7, 1);

    centerMolecule(mol);
}

// Build Toluene (C7H8) - methylbenzene
void buildToluene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Toluene (C7H8)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // Methyl group on C1
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // H on methyl
    addAtom(mol, 3.2f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, -0.45f, 0.8f, ATOM_H);
    addAtom(mol, 3.2f, -0.45f, -0.8f, ATOM_H);

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    // C-H on ring
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 6 + i, 1);
    }
    // Methyl H
    addBond(mol, 6, 12, 1);
    addBond(mol, 6, 13, 1);
    addBond(mol, 6, 14, 1);

    centerMolecule(mol);
}

// Build Phenol (C6H5OH)
void buildPhenol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Phenol (C6H5OH)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // OH group on C1
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 3.1f, 0.7f, 0.0f, ATOM_H);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 7 + i, 1);
    }

    centerMolecule(mol);
}

// Build Acetylene (C2H2)
void buildAcetylene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetylene (C2H2)");

    addAtom(mol, -0.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -1.65f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 1.65f, 0.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 3);
    addBond(mol, 0, 2, 1);
    addBond(mol, 1, 3, 1);

    centerMolecule(mol);
}

// Build Ethylene (C2H4)
void buildEthylene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ethylene (C2H4)");

    addAtom(mol, -0.67f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.67f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -1.23f, 0.92f, 0.0f, ATOM_H);
    addAtom(mol, -1.23f, -0.92f, 0.0f, ATOM_H);
    addAtom(mol, 1.23f, 0.92f, 0.0f, ATOM_H);
    addAtom(mol, 1.23f, -0.92f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 1, 4, 1);
    addBond(mol, 1, 5, 1);

    centerMolecule(mol);
}

// Build Hydrogen Cyanide (HCN)
void buildHydrogenCyanide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Hydrogen Cyanide (HCN)");

    addAtom(mol, -1.15f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.08f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.08f, 0.0f, 0.0f, ATOM_N);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 3);

    centerMolecule(mol);
}

// Build Hydrogen Sulfide (H2S)
void buildHydrogenSulfide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Hydrogen Sulfide (H2S)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_S);
    addAtom(mol, 0.96f, 0.62f, 0.0f, ATOM_H);
    addAtom(mol, -0.96f, 0.62f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);

    centerMolecule(mol);
}

// Build Chloroform (CHCl3)
void buildChloroform(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Chloroform (CHCl3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 1.07f, 0.0f, ATOM_H);
    addAtom(mol, 1.45f, -0.53f, 0.0f, ATOM_CL);
    addAtom(mol, -0.73f, -0.53f, 1.26f, ATOM_CL);
    addAtom(mol, -0.73f, -0.53f, -1.26f, ATOM_CL);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);

    centerMolecule(mol);
}

// Build Iodine (I2)
void buildIodine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Iodine (I2)");

    addAtom(mol, -1.33f, 0.0f, 0.0f, ATOM_I);
    addAtom(mol, 1.33f, 0.0f, 0.0f, ATOM_I);

    addBond(mol, 0, 1, 1);

    centerMolecule(mol);
}

// Build Ozone (O3)
void buildOzone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ozone (O3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 1.1f, 0.6f, 0.0f, ATOM_O);
    addAtom(mol, -1.1f, 0.6f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);

    centerMolecule(mol);
}

// Build Oxygen (O2)
void buildOxygen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Oxygen (O2)");

    addAtom(mol, -0.6f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.6f, 0.0f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);

    centerMolecule(mol);
}

// Build Nitrogen (N2)
void buildNitrogen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nitrogen (N2)");

    addAtom(mol, -0.55f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 0.55f, 0.0f, 0.0f, ATOM_N);

    addBond(mol, 0, 1, 3);

    centerMolecule(mol);
}

// Build Hydrogen (H2)
void buildHydrogen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Hydrogen (H2)");

    addAtom(mol, -0.37f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 0.37f, 0.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);

    centerMolecule(mol);
}

// Build Carbon Monoxide (CO)
void buildCarbonMonoxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Carbon Monoxide (CO)");

    addAtom(mol, -0.56f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.56f, 0.0f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 3);

    centerMolecule(mol);
}

// Build Nitrous Oxide (N2O)
void buildNitrousOxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nitrous Oxide (N2O)");

    addAtom(mol, -1.13f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 1.19f, 0.0f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 2);

    centerMolecule(mol);
}

// Build Sulfur Dioxide (SO2)
void buildSulfurDioxide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Sulfur Dioxide (SO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_S);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_O);
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 2);

    centerMolecule(mol);
}

// Build Hydrogen Chloride (HCl)
void buildHydrogenChloride(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Hydrogen Chloride (HCl)");

    addAtom(mol, -0.64f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 0.64f, 0.0f, 0.0f, ATOM_CL);

    addBond(mol, 0, 1, 1);

    centerMolecule(mol);
}

// Build Nitric Acid (HNO3)
void buildNitricAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nitric Acid (HNO3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 1.2f, 0.5f, 0.0f, ATOM_O);      // =O
    addAtom(mol, -1.0f, 0.8f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 0.3f, -1.2f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 1.1f, -1.5f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);

    centerMolecule(mol);
}

// Build Methanol (CH3OH)
void buildMethanol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Methanol (CH3OH)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.1f, 0.8f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);

    centerMolecule(mol);
}

// Build Ethane (C2H6)
void buildEthane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ethane (C2H6)");

    addAtom(mol, -0.77f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.77f, 0.0f, 0.0f, ATOM_C);
    // H on C1
    addAtom(mol, -1.15f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -1.15f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -1.15f, -0.5f, -0.87f, ATOM_H);
    // H on C2
    addAtom(mol, 1.15f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 1.15f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 1.15f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 1, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);

    centerMolecule(mol);
}

// Build Propene (C3H6)
void buildPropene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Propene (C3H6)");

    addAtom(mol, -1.3f, 0.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH=
    addAtom(mol, 1.3f, 0.0f, 0.0f, ATOM_C);      // =CH2
    // H on CH3
    addAtom(mol, -1.7f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -1.7f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -1.7f, -0.5f, -0.87f, ATOM_H);
    // H on middle C
    addAtom(mol, 0.0f, 1.1f, 0.0f, ATOM_H);
    // H on =CH2
    addAtom(mol, 1.9f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -0.9f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 2, 8, 1);

    centerMolecule(mol);
}

// Build Isopropanol / 2-Propanol (C3H8O)
void buildIsopropanol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Isopropanol (C3H8O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C with OH
    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 0.0f, 1.4f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, 2.2f, 0.0f, ATOM_H);      // H on OH
    addAtom(mol, 0.0f, -1.1f, 0.0f, ATOM_H);     // H on central C
    // H on CH3 (left)
    addAtom(mol, -2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, -0.87f, ATOM_H);
    // H on CH3 (right)
    addAtom(mol, 2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 2, 10, 1);
    addBond(mol, 2, 11, 1);

    centerMolecule(mol);
}

// Build Ethylene Glycol (C2H6O2)
void buildEthyleneGlycol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ethylene Glycol (C2H6O2)");

    addAtom(mol, -0.75f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.75f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -1.5f, 1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 1.5f, 1.1f, 0.0f, ATOM_O);      // OH
    addAtom(mol, -2.3f, 1.0f, 0.0f, ATOM_H);     // H on OH
    addAtom(mol, 2.3f, 1.0f, 0.0f, ATOM_H);      // H on OH
    // H on carbons
    addAtom(mol, -1.1f, -0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.1f, -0.5f, -0.9f, ATOM_H);
    addAtom(mol, 1.1f, -0.5f, 0.9f, ATOM_H);
    addAtom(mol, 1.1f, -0.5f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 1, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 1, 9, 1);

    centerMolecule(mol);
}

// Build Glycerol (C3H8O3)
void buildGlycerol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Glycerol (C3H8O3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Middle C
    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // Left C
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // Right C
    addAtom(mol, 0.0f, 1.4f, 0.0f, ATOM_O);      // Middle OH
    addAtom(mol, -2.3f, 1.0f, 0.0f, ATOM_O);     // Left OH
    addAtom(mol, 2.3f, 1.0f, 0.0f, ATOM_O);      // Right OH
    addAtom(mol, 0.0f, 2.2f, 0.0f, ATOM_H);
    addAtom(mol, -3.1f, 0.8f, 0.0f, ATOM_H);
    addAtom(mol, 3.1f, 0.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, -1.1f, 0.0f, ATOM_H);
    addAtom(mol, -1.9f, -0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.9f, -0.5f, -0.9f, ATOM_H);
    addAtom(mol, 1.9f, -0.5f, 0.9f, ATOM_H);
    addAtom(mol, 1.9f, -0.5f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 1, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 4, 7, 1);
    addBond(mol, 5, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 2, 12, 1);
    addBond(mol, 2, 13, 1);

    centerMolecule(mol);
}

// Build Acetaldehyde / Ethanal (C2H4O)
void buildAcetaldehyde(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetaldehyde (C2H4O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CHO
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, -0.6f, 1.1f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -0.6f, -1.0f, 0.0f, ATOM_H);    // H on CHO
    addAtom(mol, 2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 2.0f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 1, 4, 1);
    addBond(mol, 1, 5, 1);
    addBond(mol, 1, 6, 1);

    centerMolecule(mol);
}

// Build Formic Acid (HCOOH)
void buildFormicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Formic Acid (HCOOH)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -0.6f, 1.1f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 0.8f, -1.0f, 0.0f, ATOM_O);     // OH
    addAtom(mol, -0.9f, -0.6f, 0.0f, ATOM_H);    // H on C
    addAtom(mol, 1.6f, -0.6f, 0.0f, ATOM_H);     // H on OH

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);

    centerMolecule(mol);
}

// Build Lactic Acid (C3H6O3)
void buildLacticAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Lactic Acid (C3H6O3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C with OH
    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // COOH
    addAtom(mol, 0.0f, 1.4f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 2.1f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.1f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 0.0f, 2.2f, 0.0f, ATOM_H);
    addAtom(mol, 2.9f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, -1.1f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 2);
    addBond(mol, 2, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);

    centerMolecule(mol);
}

// Build Ethyl Acetate (C4H8O2)
void buildEthylAcetate(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ethyl Acetate (C4H8O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // CH3 (acetyl)
    addAtom(mol, 0.6f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 0.8f, -1.2f, 0.0f, ATOM_O);     // O-
    addAtom(mol, 2.2f, -1.2f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 3.0f, -2.4f, 0.0f, ATOM_C);     // CH3 (ethyl)
    // H on acetyl CH3
    addAtom(mol, -2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, -0.87f, ATOM_H);
    // H on CH2
    addAtom(mol, 2.6f, -0.7f, 0.9f, ATOM_H);
    addAtom(mol, 2.6f, -0.7f, -0.9f, ATOM_H);
    // H on ethyl CH3
    addAtom(mol, 4.1f, -2.3f, 0.0f, ATOM_H);
    addAtom(mol, 2.7f, -2.9f, 0.9f, ATOM_H);
    addAtom(mol, 2.7f, -2.9f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 5, 13, 1);

    centerMolecule(mol);
}

// Build Acetonitrile (C2H3N)
void buildAcetonitrile(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetonitrile (C2H3N)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 1.46f, 0.0f, 0.0f, ATOM_C);     // C
    addAtom(mol, 2.62f, 0.0f, 0.0f, ATOM_N);     // N
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 3);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);

    centerMolecule(mol);
}

// Build DMSO / Dimethyl Sulfoxide (C2H6OS)
void buildDMSO(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "DMSO (C2H6OS)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_S);
    addAtom(mol, 0.0f, 1.5f, 0.0f, ATOM_O);
    addAtom(mol, -1.5f, -0.5f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 1.5f, -0.5f, 0.0f, ATOM_C);     // CH3
    // H on CH3 (left)
    addAtom(mol, -2.0f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -1.0f, -0.87f, ATOM_H);
    // H on CH3 (right)
    addAtom(mol, 2.0f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, 2.0f, -1.0f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 3, 7, 1);
    addBond(mol, 3, 8, 1);
    addBond(mol, 3, 9, 1);

    centerMolecule(mol);
}

// Build Dichloromethane (CH2Cl2)
void buildDichloromethane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Dichloromethane (CH2Cl2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_CL);
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_CL);
    addAtom(mol, 0.0f, -0.6f, 0.9f, ATOM_H);
    addAtom(mol, 0.0f, -0.6f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);

    centerMolecule(mol);
}

// Build Chlorobenzene (C6H5Cl)
void buildChlorobenzene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Chlorobenzene (C6H5Cl)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // Chlorine on C1
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_CL);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 6 + i, 1);
    }

    centerMolecule(mol);
}

// Build Nitrobenzene (C6H5NO2)
void buildNitrobenzene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nitrobenzene (C6H5NO2)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // NO2 group on C1
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 3.3f, 1.0f, 0.0f, ATOM_O);
    addAtom(mol, 3.3f, -1.0f, 0.0f, ATOM_O);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 2);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 8 + i, 1);
    }

    centerMolecule(mol);
}

// Build Aniline (C6H5NH2)
void buildAniline(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Aniline (C6H5NH2)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // NH2 group on C1
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_N);
    addAtom(mol, 3.1f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, 3.1f, -0.9f, 0.0f, ATOM_H);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 6, 8, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 8 + i, 1);
    }

    centerMolecule(mol);
}

// Build Styrene (C8H8)
void buildStyrene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Styrene (C8H8)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // Vinyl group (-CH=CH2)
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // =CH-
    addAtom(mol, 3.9f, 0.0f, 0.0f, ATOM_C);      // =CH2

    // H on vinyl
    addAtom(mol, 2.6f, 1.1f, 0.0f, ATOM_H);
    addAtom(mol, 4.5f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, 4.5f, -0.9f, 0.0f, ATOM_H);

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 7, 9, 1);
    addBond(mol, 7, 10, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 10 + i, 1);
    }

    centerMolecule(mol);
}

// Build Benzoic Acid (C7H6O2)
void buildBenzoicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Benzoic Acid (C7H6O2)");

    float r = 1.4f;
    // Benzene ring
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }

    // COOH group
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 3.2f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 3.2f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 4.0f, -1.0f, 0.0f, ATOM_H);     // H on OH

    // H on benzene (5 positions)
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 9 + i, 1);
    }

    centerMolecule(mol);
}

// Build Valine (C5H11NO2)
void buildValine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Valine (C5H11NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // Side chain CH
    addAtom(mol, -1.3f, -2.2f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 1.3f, -2.2f, 0.0f, ATOM_C);     // CH3
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);      // H on alpha C
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);     // H on NH2
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);     // H on NH2
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);      // H on OH
    addAtom(mol, 0.0f, -2.1f, 0.9f, ATOM_H);     // H on side chain CH
    // H on CH3 (left)
    addAtom(mol, -1.3f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -1.8f, 0.8f, ATOM_H);
    addAtom(mol, -2.0f, -1.8f, -0.8f, ATOM_H);
    // H on CH3 (right)
    addAtom(mol, 1.3f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -1.8f, 0.8f, ATOM_H);
    addAtom(mol, 2.0f, -1.8f, -0.8f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 6, 13, 1);
    addBond(mol, 6, 14, 1);
    addBond(mol, 6, 15, 1);
    addBond(mol, 7, 16, 1);
    addBond(mol, 7, 17, 1);
    addBond(mol, 7, 18, 1);

    centerMolecule(mol);
}

// Build Leucine (C6H13NO2)
void buildLeucine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Leucine (C6H13NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // CH
    addAtom(mol, -1.3f, -3.7f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 1.3f, -3.7f, 0.0f, ATOM_C);     // CH3
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);      // H on alpha C
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);     // H on NH2
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);     // H on NH2
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);      // H on OH
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);    // H on CH2
    addAtom(mol, 0.9f, -1.8f, 0.0f, ATOM_H);     // H on CH2
    addAtom(mol, 0.0f, -3.6f, 0.9f, ATOM_H);     // H on CH
    // H on CH3 (left)
    addAtom(mol, -1.3f, -4.8f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -3.3f, 0.8f, ATOM_H);
    addAtom(mol, -2.0f, -3.3f, -0.8f, ATOM_H);
    // H on CH3 (right)
    addAtom(mol, 1.3f, -4.8f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -3.3f, 0.8f, ATOM_H);
    addAtom(mol, 2.0f, -3.3f, -0.8f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 6, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 5, 13, 1);
    addBond(mol, 5, 14, 1);
    addBond(mol, 6, 15, 1);
    addBond(mol, 7, 16, 1);
    addBond(mol, 7, 17, 1);
    addBond(mol, 7, 18, 1);
    addBond(mol, 8, 19, 1);
    addBond(mol, 8, 20, 1);
    addBond(mol, 8, 21, 1);

    centerMolecule(mol);
}

// Build tert-Butanol (C4H10O)
void buildTertButanol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "tert-Butanol (C4H10O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C
    addAtom(mol, 0.0f, 1.5f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, 2.3f, 0.0f, ATOM_H);
    addAtom(mol, 1.4f, -0.5f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -0.7f, -0.5f, 1.2f, ATOM_C);    // CH3
    addAtom(mol, -0.7f, -0.5f, -1.2f, ATOM_C);   // CH3
    // H on CH3 groups
    addAtom(mol, 2.0f, 0.4f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, 1.9f, -1.0f, -0.87f, ATOM_H);
    addAtom(mol, -0.1f, 0.0f, 2.0f, ATOM_H);
    addAtom(mol, -0.2f, -1.5f, 1.2f, ATOM_H);
    addAtom(mol, -1.7f, -0.3f, 1.4f, ATOM_H);
    addAtom(mol, -0.1f, 0.0f, -2.0f, ATOM_H);
    addAtom(mol, -0.2f, -1.5f, -1.2f, ATOM_H);
    addAtom(mol, -1.7f, -0.3f, -1.4f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 3, 7, 1);
    addBond(mol, 3, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 5, 13, 1);
    addBond(mol, 5, 14, 1);

    centerMolecule(mol);
}

// Build 1-Butanol (C4H10O)
void buildButanol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "1-Butanol (C4H10O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 3.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 4.5f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 5.9f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 6.5f, 0.8f, 0.0f, ATOM_H);
    // H atoms
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 3.0f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 3.0f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 4.5f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 4.5f, 0.6f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 2, 11, 1);
    addBond(mol, 2, 12, 1);
    addBond(mol, 3, 13, 1);
    addBond(mol, 3, 14, 1);

    centerMolecule(mol);
}

// Build Diethyl Ether (C4H10O)
void buildDiethylEther(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Diethyl Ether (C4H10O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, -1.4f, 0.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, -2.9f, 0.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 2.9f, 0.0f, 0.0f, ATOM_C);      // CH3
    // H on CH2 groups
    addAtom(mol, -1.4f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 1.4f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 1.4f, 0.6f, -0.9f, ATOM_H);
    // H on CH3 groups
    addAtom(mol, -3.4f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -3.4f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -3.4f, -0.5f, -0.87f, ATOM_H);
    addAtom(mol, 3.4f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.4f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 3.4f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 1, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 1, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 3, 9, 1);
    addBond(mol, 3, 10, 1);
    addBond(mol, 3, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 4, 14, 1);

    centerMolecule(mol);
}

// Build MTBE - Methyl tert-butyl ether (C5H12O)
void buildMTBE(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "MTBE (C5H12O)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C
    addAtom(mol, 0.0f, 1.4f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, 2.8f, 0.0f, ATOM_C);      // OCH3
    addAtom(mol, 1.4f, -0.5f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -0.7f, -0.5f, 1.2f, ATOM_C);    // CH3
    addAtom(mol, -0.7f, -0.5f, -1.2f, ATOM_C);   // CH3
    // H on OCH3
    addAtom(mol, 0.5f, 3.3f, 0.87f, ATOM_H);
    addAtom(mol, 0.5f, 3.3f, -0.87f, ATOM_H);
    addAtom(mol, -1.0f, 3.1f, 0.0f, ATOM_H);
    // H on tert-butyl CH3 groups
    addAtom(mol, 2.0f, 0.4f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, 1.9f, -1.0f, -0.87f, ATOM_H);
    addAtom(mol, -0.1f, 0.0f, 2.0f, ATOM_H);
    addAtom(mol, -0.2f, -1.5f, 1.2f, ATOM_H);
    addAtom(mol, -1.7f, -0.3f, 1.4f, ATOM_H);
    addAtom(mol, -0.1f, 0.0f, -2.0f, ATOM_H);
    addAtom(mol, -0.2f, -1.5f, -1.2f, ATOM_H);
    addAtom(mol, -1.7f, -0.3f, -1.4f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 3, 9, 1);
    addBond(mol, 3, 10, 1);
    addBond(mol, 3, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 4, 14, 1);
    addBond(mol, 5, 15, 1);
    addBond(mol, 5, 16, 1);
    addBond(mol, 5, 17, 1);

    centerMolecule(mol);
}

// Build THF - Tetrahydrofuran (C4H8O)
void buildTHF(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "THF (C4H8O)");

    float r = 1.2f;
    // 5-membered ring
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_O);
    addAtom(mol, r * cosf(2*PI/5), r * sinf(2*PI/5), 0.2f, ATOM_C);
    addAtom(mol, r * cosf(4*PI/5), r * sinf(4*PI/5), -0.2f, ATOM_C);
    addAtom(mol, r * cosf(6*PI/5), r * sinf(6*PI/5), 0.2f, ATOM_C);
    addAtom(mol, r * cosf(8*PI/5), r * sinf(8*PI/5), -0.2f, ATOM_C);
    // H atoms
    addAtom(mol, r * cosf(2*PI/5) + 0.5f, r * sinf(2*PI/5) + 0.8f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/5) + 0.5f, r * sinf(2*PI/5) + 0.8f, -0.5f, ATOM_H);
    addAtom(mol, r * cosf(4*PI/5) - 0.8f, r * sinf(4*PI/5) + 0.5f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(4*PI/5) - 0.8f, r * sinf(4*PI/5) + 0.5f, -0.5f, ATOM_H);
    addAtom(mol, r * cosf(6*PI/5) - 0.8f, r * sinf(6*PI/5) - 0.5f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(6*PI/5) - 0.8f, r * sinf(6*PI/5) - 0.5f, -0.5f, ATOM_H);
    addAtom(mol, r * cosf(8*PI/5) + 0.5f, r * sinf(8*PI/5) - 0.8f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(8*PI/5) + 0.5f, r * sinf(8*PI/5) - 0.8f, -0.5f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 0, 1);
    addBond(mol, 1, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 3, 9, 1);
    addBond(mol, 3, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 4, 12, 1);

    centerMolecule(mol);
}

// Build 1,4-Dioxane (C4H8O2)
void buildDioxane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "1,4-Dioxane (C4H8O2)");

    float r = 1.4f;
    // 6-membered ring with O at 1 and 4 positions
    addAtom(mol, r * cosf(0), r * sinf(0), 0.3f, ATOM_O);
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), -0.3f, ATOM_C);
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.3f, ATOM_C);
    addAtom(mol, r * cosf(PI), r * sinf(PI), -0.3f, ATOM_O);
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.3f, ATOM_C);
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), -0.3f, ATOM_C);
    // H atoms
    addAtom(mol, r * cosf(PI/3) * 1.6f, r * sinf(PI/3) * 1.6f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(PI/3) * 1.6f, r * sinf(PI/3) * 1.6f, -1.1f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/3) * 1.6f, r * sinf(2*PI/3) * 1.6f, 1.1f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/3) * 1.6f, r * sinf(2*PI/3) * 1.6f, -0.5f, ATOM_H);
    addAtom(mol, r * cosf(4*PI/3) * 1.6f, r * sinf(4*PI/3) * 1.6f, 1.1f, ATOM_H);
    addAtom(mol, r * cosf(4*PI/3) * 1.6f, r * sinf(4*PI/3) * 1.6f, -0.5f, ATOM_H);
    addAtom(mol, r * cosf(5*PI/3) * 1.6f, r * sinf(5*PI/3) * 1.6f, 0.5f, ATOM_H);
    addAtom(mol, r * cosf(5*PI/3) * 1.6f, r * sinf(5*PI/3) * 1.6f, -1.1f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 5, 13, 1);

    centerMolecule(mol);
}

// Build DMF - Dimethylformamide (C3H7NO)
void buildDMF(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "DMF (C3H7NO)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 1.2f, -0.6f, 0.0f, ATOM_N);     // N
    addAtom(mol, -1.2f, -0.6f, 0.0f, ATOM_H);    // H on C
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 2.4f, 0.1f, 0.0f, ATOM_C);      // CH3
    // H on CH3 groups
    addAtom(mol, 0.7f, -2.6f, 0.87f, ATOM_H);
    addAtom(mol, 0.7f, -2.6f, -0.87f, ATOM_H);
    addAtom(mol, 2.2f, -2.4f, 0.0f, ATOM_H);
    addAtom(mol, 2.9f, -0.4f, 0.87f, ATOM_H);
    addAtom(mol, 2.9f, -0.4f, -0.87f, ATOM_H);
    addAtom(mol, 2.9f, 1.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 4, 6, 1);
    addBond(mol, 4, 7, 1);
    addBond(mol, 4, 8, 1);
    addBond(mol, 5, 9, 1);
    addBond(mol, 5, 10, 1);
    addBond(mol, 5, 11, 1);

    centerMolecule(mol);
}

// Build Carbon Tetrachloride (CCl4)
void buildCarbonTetrachloride(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Carbon Tetrachloride (CCl4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.85f, 0.0f, ATOM_CL);
    addAtom(mol, -1.2f, 0.85f, 0.0f, ATOM_CL);
    addAtom(mol, 0.0f, -0.6f, 1.2f, ATOM_CL);
    addAtom(mol, 0.0f, -0.6f, -1.2f, ATOM_CL);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 0, 4, 1);

    centerMolecule(mol);
}

// Build Methyl Acetate (C3H6O2)
void buildMethylAcetate(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Methyl Acetate (C3H6O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // CH3 (acetyl)
    addAtom(mol, 0.6f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 0.8f, -1.2f, 0.0f, ATOM_O);     // O-
    addAtom(mol, 2.2f, -1.2f, 0.0f, ATOM_C);     // CH3 (methyl)
    // H on acetyl CH3
    addAtom(mol, -2.0f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -2.0f, -0.5f, -0.87f, ATOM_H);
    // H on methyl CH3
    addAtom(mol, 2.7f, -0.2f, 0.0f, ATOM_H);
    addAtom(mol, 2.7f, -1.7f, 0.87f, ATOM_H);
    addAtom(mol, 2.7f, -1.7f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 2);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 1, 5, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 4, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 4, 10, 1);

    centerMolecule(mol);
}

// Build Acetic Anhydride (C4H6O3)
void buildAceticAnhydride(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetic Anhydride (C4H6O3)");

    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // C=O (left)
    addAtom(mol, -2.1f, 1.1f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -3.0f, -0.5f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_O);      // Central O
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // C=O (right)
    addAtom(mol, 2.1f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 3.0f, -0.5f, 0.0f, ATOM_C);     // CH3
    // H atoms
    addAtom(mol, -3.5f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -3.5f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, -3.5f, -1.0f, -0.87f, ATOM_H);
    addAtom(mol, 3.5f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 3.5f, -1.0f, 0.87f, ATOM_H);
    addAtom(mol, 3.5f, -1.0f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 4, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 6, 11, 1);
    addBond(mol, 6, 12, 1);

    centerMolecule(mol);
}

// Build Propionic Acid (C3H6O2)
void buildPropionicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Propionic Acid (C3H6O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 3.0f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 3.6f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 3.6f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 4.5f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 1, 10, 1);

    centerMolecule(mol);
}

// Build Butyric Acid (C4H8O2)
void buildButyricAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Butyric Acid (C4H8O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 3.0f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 4.5f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 5.1f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 5.1f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 6.0f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 3.0f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 3.0f, 0.6f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 3, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 2, 12, 1);
    addBond(mol, 2, 13, 1);

    centerMolecule(mol);
}

// Build Succinic Acid (C4H6O4)
void buildSuccinicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Succinic Acid (C4H6O4)");

    addAtom(mol, -1.5f, 0.0f, 0.0f, ATOM_C);     // COOH
    addAtom(mol, -2.1f, 1.1f, 0.0f, ATOM_O);
    addAtom(mol, -2.1f, -1.1f, 0.0f, ATOM_O);
    addAtom(mol, -3.0f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // CH2
    addAtom(mol, 3.0f, 0.0f, 0.0f, ATOM_C);      // COOH
    addAtom(mol, 3.6f, 1.1f, 0.0f, ATOM_O);
    addAtom(mol, 3.6f, -1.1f, 0.0f, ATOM_O);
    addAtom(mol, 4.5f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 0.0f, 0.6f, -0.9f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 1.5f, 0.6f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 0, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 5, 13, 1);

    centerMolecule(mol);
}

// Build Benzaldehyde (C7H6O)
void buildBenzaldehyde(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Benzaldehyde (C7H6O)");

    float r = 1.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    // CHO group
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 3.2f, 1.1f, 0.0f, ATOM_O);
    addAtom(mol, 3.2f, -1.0f, 0.0f, ATOM_H);
    // H on benzene
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 8 + i, 1);
    }

    centerMolecule(mol);
}

// Build Bromobenzene (C6H5Br)
void buildBromobenzene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Bromobenzene (C6H5Br)");

    float r = 1.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    addAtom(mol, 2.8f, 0.0f, 0.0f, ATOM_BR);
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 6 + i, 1);
    }

    centerMolecule(mol);
}

// Build p-Xylene (C8H10)
void buildPXylene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "p-Xylene (C8H10)");

    float r = 1.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    // CH3 at positions 1 and 4
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);
    // H on CH3 groups
    addAtom(mol, 3.2f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, -0.45f, 0.8f, ATOM_H);
    addAtom(mol, 3.2f, -0.45f, -0.8f, ATOM_H);
    addAtom(mol, -3.2f, 0.9f, 0.0f, ATOM_H);
    addAtom(mol, -3.2f, -0.45f, 0.8f, ATOM_H);
    addAtom(mol, -3.2f, -0.45f, -0.8f, ATOM_H);
    // H on ring
    float rH = 2.4f;
    addAtom(mol, rH * cosf(PI/3), rH * sinf(PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(2*PI/3), rH * sinf(2*PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(4*PI/3), rH * sinf(4*PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(5*PI/3), rH * sinf(5*PI/3), 0.0f, ATOM_H);

    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 3, 7, 1);
    addBond(mol, 6, 8, 1);
    addBond(mol, 6, 9, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 7, 12, 1);
    addBond(mol, 7, 13, 1);
    addBond(mol, 1, 14, 1);
    addBond(mol, 2, 15, 1);
    addBond(mol, 4, 16, 1);
    addBond(mol, 5, 17, 1);

    centerMolecule(mol);
}

// Build Anisole (C7H8O)
void buildAnisole(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Anisole (C7H8O)");

    float r = 1.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 3.6f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 4.1f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, 4.1f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, 4.1f, -0.5f, -0.87f, ATOM_H);
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 7, 9, 1);
    addBond(mol, 7, 10, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 10 + i, 1);
    }

    centerMolecule(mol);
}

// Build Phenylacetylene (C8H6)
void buildPhenylacetylene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Phenylacetylene (C8H6)");

    float r = 1.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, r * cosf(angle), r * sinf(angle), 0.0f, ATOM_C);
    }
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C triple bond
    addAtom(mol, 3.8f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 4.85f, 0.0f, 0.0f, ATOM_H);
    float rH = 2.4f;
    for (int i = 1; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, rH * cosf(angle), rH * sinf(angle), 0.0f, ATOM_H);
    }

    for (int i = 0; i < 6; i++) {
        addBond(mol, i, (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 3);
    addBond(mol, 7, 8, 1);
    for (int i = 1; i < 6; i++) {
        addBond(mol, i, 8 + i, 1);
    }

    centerMolecule(mol);
}

// Build Fructose (C6H12O6)
void buildFructose(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Fructose (C6H12O6)");

    // Furanose ring form
    float r = 1.3f;
    addAtom(mol, r, 0.0f, 0.3f, ATOM_C);          // C2
    addAtom(mol, r * 0.31f, r * 0.95f, -0.2f, ATOM_C);   // C3
    addAtom(mol, -r * 0.81f, r * 0.59f, 0.3f, ATOM_C);   // C4
    addAtom(mol, -r * 0.81f, -r * 0.59f, -0.2f, ATOM_C); // C5
    addAtom(mol, r * 0.31f, -r * 0.95f, 0.1f, ATOM_O);   // Ring O
    // C1 and C6
    addAtom(mol, r + 1.3f, 0.3f, 0.0f, ATOM_C);   // C1 (CH2OH)
    addAtom(mol, -r * 0.81f - 1.2f, -r * 0.59f, 0.0f, ATOM_C);  // C6 (CH2OH)
    // OH groups
    addAtom(mol, r * 1.5f, 0.0f, 1.0f, ATOM_O);
    addAtom(mol, r * 0.5f, r * 1.5f, -0.8f, ATOM_O);
    addAtom(mol, -r * 1.3f, r * 1.0f, 1.0f, ATOM_O);
    addAtom(mol, r + 1.8f, 1.2f, 0.0f, ATOM_O);
    addAtom(mol, -r * 0.81f - 1.8f, -r * 0.59f - 1.0f, 0.0f, ATOM_O);
    // Simplified H atoms
    addAtom(mol, r * 1.5f + 0.8f, 0.0f, 1.0f, ATOM_H);
    addAtom(mol, r * 0.5f + 0.8f, r * 1.5f, -0.8f, ATOM_H);
    addAtom(mol, -r * 1.3f - 0.8f, r * 1.0f, 1.0f, ATOM_H);
    addAtom(mol, r + 1.8f + 0.8f, 1.2f, 0.0f, ATOM_H);
    addAtom(mol, -r * 0.81f - 1.8f - 0.8f, -r * 0.59f - 1.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 0, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 5, 10, 1);
    addBond(mol, 6, 11, 1);
    addBond(mol, 7, 12, 1);
    addBond(mol, 8, 13, 1);
    addBond(mol, 9, 14, 1);
    addBond(mol, 10, 15, 1);
    addBond(mol, 11, 16, 1);

    centerMolecule(mol);
}

// Build Ribose (C5H10O5)
void buildRibose(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ribose (C5H10O5)");

    float r = 1.2f;
    // Furanose ring
    addAtom(mol, r, 0.0f, 0.2f, ATOM_C);          // C1
    addAtom(mol, r * 0.31f, r * 0.95f, -0.2f, ATOM_C);   // C2
    addAtom(mol, -r * 0.81f, r * 0.59f, 0.2f, ATOM_C);   // C3
    addAtom(mol, -r * 0.81f, -r * 0.59f, -0.2f, ATOM_C); // C4
    addAtom(mol, r * 0.31f, -r * 0.95f, 0.1f, ATOM_O);   // Ring O
    addAtom(mol, -r * 0.81f - 1.2f, -r * 0.59f, 0.0f, ATOM_C);  // C5
    // OH groups
    addAtom(mol, r + 0.9f, 0.7f, 0.0f, ATOM_O);
    addAtom(mol, r * 0.5f, r * 1.5f, -0.5f, ATOM_O);
    addAtom(mol, -r * 1.3f, r * 1.0f, 0.5f, ATOM_O);
    addAtom(mol, -r * 0.81f - 1.8f, -r * 0.59f - 0.8f, 0.0f, ATOM_O);
    // H on OH
    addAtom(mol, r + 1.7f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, r * 0.5f + 0.8f, r * 1.5f, -0.5f, ATOM_H);
    addAtom(mol, -r * 1.3f - 0.8f, r * 1.0f, 0.5f, ATOM_H);
    addAtom(mol, -r * 0.81f - 2.6f, -r * 0.59f - 0.6f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 0, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 2, 8, 1);
    addBond(mol, 5, 9, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 8, 12, 1);
    addBond(mol, 9, 13, 1);

    centerMolecule(mol);
}

// Build Deoxyribose (C5H10O4)
void buildDeoxyribose(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Deoxyribose (C5H10O4)");

    float r = 1.2f;
    addAtom(mol, r, 0.0f, 0.2f, ATOM_C);          // C1
    addAtom(mol, r * 0.31f, r * 0.95f, -0.2f, ATOM_C);   // C2
    addAtom(mol, -r * 0.81f, r * 0.59f, 0.2f, ATOM_C);   // C3
    addAtom(mol, -r * 0.81f, -r * 0.59f, -0.2f, ATOM_C); // C4
    addAtom(mol, r * 0.31f, -r * 0.95f, 0.1f, ATOM_O);   // Ring O
    addAtom(mol, -r * 0.81f - 1.2f, -r * 0.59f, 0.0f, ATOM_C);  // C5
    // OH groups (one less than ribose)
    addAtom(mol, r + 0.9f, 0.7f, 0.0f, ATOM_O);
    addAtom(mol, -r * 1.3f, r * 1.0f, 0.5f, ATOM_O);
    addAtom(mol, -r * 0.81f - 1.8f, -r * 0.59f - 0.8f, 0.0f, ATOM_O);
    // H atoms
    addAtom(mol, r + 1.7f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -r * 1.3f - 0.8f, r * 1.0f, 0.5f, ATOM_H);
    addAtom(mol, -r * 0.81f - 2.6f, -r * 0.59f - 0.6f, 0.0f, ATOM_H);
    addAtom(mol, r * 0.5f, r * 1.5f, 0.5f, ATOM_H);
    addAtom(mol, r * 0.5f, r * 1.5f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 0, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 2, 7, 1);
    addBond(mol, 5, 8, 1);
    addBond(mol, 6, 9, 1);
    addBond(mol, 7, 10, 1);
    addBond(mol, 8, 11, 1);
    addBond(mol, 1, 12, 1);
    addBond(mol, 1, 13, 1);

    centerMolecule(mol);
}

// Build Isoleucine (C6H13NO2)
void buildIsoleucine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Isoleucine (C6H13NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // Beta C (CH)
    addAtom(mol, -1.3f, -2.2f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, 1.3f, -2.2f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 1.3f, -3.7f, 0.0f, ATOM_C);     // CH3
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, -2.1f, 0.9f, ATOM_H);
    addAtom(mol, -1.3f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, -2.0f, -1.8f, 0.8f, ATOM_H);
    addAtom(mol, -2.0f, -1.8f, -0.8f, ATOM_H);
    addAtom(mol, 2.0f, -1.8f, 0.8f, ATOM_H);
    addAtom(mol, 2.0f, -1.8f, -0.8f, ATOM_H);
    addAtom(mol, 1.3f, -4.3f, 0.9f, ATOM_H);
    addAtom(mol, 2.0f, -4.0f, -0.5f, ATOM_H);
    addAtom(mol, 0.4f, -4.0f, -0.5f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 5, 13, 1);
    addBond(mol, 6, 14, 1);
    addBond(mol, 6, 15, 1);
    addBond(mol, 6, 16, 1);
    addBond(mol, 7, 17, 1);
    addBond(mol, 7, 18, 1);
    addBond(mol, 8, 19, 1);
    addBond(mol, 8, 20, 1);
    addBond(mol, 8, 21, 1);

    centerMolecule(mol);
}

// Build Serine (C3H7NO3)
void buildSerine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Serine (C3H7NO3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -2.9f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.8f, -3.3f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 6, 13, 1);

    centerMolecule(mol);
}

// Build Threonine (C4H9NO3)
void buildThreonine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Threonine (C4H9NO3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH (with OH)
    addAtom(mol, 0.0f, -2.9f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 1.3f, -1.8f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -1.4f, 0.0f, ATOM_H);
    addAtom(mol, 0.0f, -3.5f, 0.9f, ATOM_H);
    addAtom(mol, 0.9f, -3.2f, -0.5f, ATOM_H);
    addAtom(mol, -0.9f, -3.2f, -0.5f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 4, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 7, 13, 1);
    addBond(mol, 6, 14, 1);
    addBond(mol, 6, 15, 1);
    addBond(mol, 6, 16, 1);

    centerMolecule(mol);
}

// Build Aspartic Acid (C4H7NO4)
void buildAsparticAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Aspartic Acid (C4H7NO4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // Side COOH
    addAtom(mol, 0.0f, -4.2f, 0.0f, ATOM_O);
    addAtom(mol, 1.1f, -3.3f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -2.9f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 5, 13, 1);
    addBond(mol, 5, 14, 1);
    addBond(mol, 8, 15, 1);

    centerMolecule(mol);
}

// Build Glutamic Acid (C5H9NO4)
void buildGlutamicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Glutamic Acid (C5H9NO4)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -4.5f, 0.0f, ATOM_C);     // Side COOH
    addAtom(mol, 0.0f, -5.7f, 0.0f, ATOM_O);
    addAtom(mol, 1.1f, -4.8f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, 0.9f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -4.4f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 7, 9, 1);
    addBond(mol, 0, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 1, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 5, 14, 1);
    addBond(mol, 5, 15, 1);
    addBond(mol, 6, 16, 1);
    addBond(mol, 6, 17, 1);
    addBond(mol, 9, 18, 1);

    centerMolecule(mol);
}

// Build Lysine (C6H14N2O2)
void buildLysine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Lysine (C6H14N2O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -4.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -6.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -7.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.8f, -8.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.8f, -8.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 0, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 1, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 9, 14, 1);
    addBond(mol, 9, 15, 1);

    centerMolecule(mol);
}

// Build Histidine (C6H9N3O2)
void buildHistidine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Histidine (C6H9N3O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    // Imidazole ring
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // C
    addAtom(mol, -1.0f, -3.7f, 0.0f, ATOM_N);    // N
    addAtom(mol, -0.6f, -5.0f, 0.0f, ATOM_C);    // C
    addAtom(mol, 0.8f, -5.0f, 0.0f, ATOM_N);     // NH
    addAtom(mol, 1.1f, -3.6f, 0.0f, ATOM_C);     // C
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -1.1f, -5.9f, 0.0f, ATOM_H);
    addAtom(mol, 1.3f, -5.8f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -3.2f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 6, 2);
    addBond(mol, 0, 11, 1);
    addBond(mol, 1, 12, 1);
    addBond(mol, 1, 13, 1);
    addBond(mol, 4, 14, 1);
    addBond(mol, 8, 15, 1);
    addBond(mol, 9, 16, 1);
    addBond(mol, 10, 17, 1);

    centerMolecule(mol);
}

// Build Phenylalanine (C9H11NO2)
void buildPhenylalanine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Phenylalanine (C9H11NO2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    // Benzene ring
    float r = 1.4f;
    float cx = 0.0f, cy = -3.5f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, cx + r * cosf(angle), cy + r * sinf(angle), 0.0f, ATOM_C);
    }
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    float rH = 2.4f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, cx + rH * cosf(angle), cy + rH * sinf(angle), 0.0f, ATOM_H);
    }

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    // Ring bonds
    for (int i = 0; i < 6; i++) {
        addBond(mol, 6 + i, 6 + (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 0, 12, 1);
    addBond(mol, 1, 13, 1);
    addBond(mol, 1, 14, 1);
    addBond(mol, 4, 15, 1);
    for (int i = 0; i < 6; i++) {
        addBond(mol, 6 + i, 16 + i, 1);
    }

    centerMolecule(mol);
}

// Build Tyrosine (C9H11NO3)
void buildTyrosine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Tyrosine (C9H11NO3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    // Benzene ring
    float r = 1.4f;
    float cx = 0.0f, cy = -3.5f;
    for (int i = 0; i < 6; i++) {
        float angle = i * PI / 3.0f;
        addAtom(mol, cx + r * cosf(angle), cy + r * sinf(angle), 0.0f, ATOM_C);
    }
    // OH on para position
    addAtom(mol, cx - 2.6f, cy, 0.0f, ATOM_O);
    addAtom(mol, cx - 3.4f, cy, 0.0f, ATOM_H);
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    float rH = 2.4f;
    for (int i = 0; i < 6; i++) {
        if (i == 3) continue;  // Skip para position (has OH)
        float angle = i * PI / 3.0f;
        addAtom(mol, cx + rH * cosf(angle), cy + rH * sinf(angle), 0.0f, ATOM_H);
    }

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    for (int i = 0; i < 6; i++) {
        addBond(mol, 6 + i, 6 + (i + 1) % 6, (i % 2 == 0) ? 2 : 1);
    }
    addBond(mol, 9, 12, 1);  // C-OH
    addBond(mol, 12, 13, 1);
    addBond(mol, 0, 14, 1);
    addBond(mol, 1, 15, 1);
    addBond(mol, 1, 16, 1);
    addBond(mol, 4, 17, 1);
    int hIdx = 18;
    for (int i = 0; i < 6; i++) {
        if (i == 3) continue;
        addBond(mol, 6 + i, hIdx++, 1);
    }

    centerMolecule(mol);
}

// Build Tryptophan (C11H12N2O2)
void buildTryptophan(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Tryptophan (C11H12N2O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    // Indole ring system
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // C3 pyrrole
    addAtom(mol, -1.0f, -3.7f, 0.0f, ATOM_C);    // C3a
    addAtom(mol, 1.1f, -3.7f, 0.0f, ATOM_C);     // C2
    addAtom(mol, 1.0f, -5.0f, 0.0f, ATOM_N);     // N1
    // Benzene fused ring
    addAtom(mol, -1.0f, -5.1f, 0.0f, ATOM_C);    // C7a
    addAtom(mol, -2.2f, -3.2f, 0.0f, ATOM_C);    // C4
    addAtom(mol, -3.3f, -4.0f, 0.0f, ATOM_C);    // C5
    addAtom(mol, -3.2f, -5.4f, 0.0f, ATOM_C);    // C6
    addAtom(mol, -2.0f, -6.0f, 0.0f, ATOM_C);    // C7
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 2.0f, -3.3f, 0.0f, ATOM_H);
    addAtom(mol, 1.7f, -5.6f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 6, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 7, 1);
    addBond(mol, 7, 11, 2);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 10, 2);
    addBond(mol, 0, 15, 1);
    addBond(mol, 1, 16, 1);
    addBond(mol, 1, 17, 1);
    addBond(mol, 4, 18, 1);
    addBond(mol, 8, 19, 1);
    addBond(mol, 9, 20, 1);

    centerMolecule(mol);
}

// Build Proline (C5H9NO2)
void buildProline(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Proline (C5H9NO2)");

    // 5-membered ring with N
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.2f, 0.0f, 0.0f, ATOM_N);     // N (in ring)
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // Ring C
    addAtom(mol, -1.4f, -1.5f, 0.0f, ATOM_C);    // Ring C
    addAtom(mol, -1.8f, -0.05f, 0.0f, ATOM_H);   // H on N
    // H atoms
    addAtom(mol, 0.0f, 0.6f, 0.9f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.5f, -2.0f, 0.9f, ATOM_H);
    addAtom(mol, 0.5f, -2.0f, -0.9f, ATOM_H);
    addAtom(mol, -1.9f, -2.0f, 0.9f, ATOM_H);
    addAtom(mol, -1.9f, -2.0f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 1, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 4, 9, 1);
    addBond(mol, 5, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 6, 12, 1);
    addBond(mol, 6, 13, 1);

    centerMolecule(mol);
}

// Build Cysteine (C3H7NO2S)
void buildCysteine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cysteine (C3H7NO2S)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.2f, 0.0f, ATOM_S);     // SH
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 1.0f, -3.8f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 6, 13, 1);

    centerMolecule(mol);
}

// Build Methionine (C5H11NO2S)
void buildMethionine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Methionine (C5H11NO2S)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -4.7f, 0.0f, ATOM_S);     // S
    addAtom(mol, 0.0f, -6.3f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.5f, -6.8f, 0.87f, ATOM_H);
    addAtom(mol, 0.5f, -6.8f, -0.87f, ATOM_H);
    addAtom(mol, -1.0f, -6.6f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 8, 13, 1);
    addBond(mol, 8, 14, 1);
    addBond(mol, 8, 15, 1);

    centerMolecule(mol);
}

// Build Pyruvate / Pyruvic Acid (C3H4O3)
void buildPyruvate(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Pyruvate (C3H4O3)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 1.5f, 0.0f, 0.0f, ATOM_C);      // C=O (ketone)
    addAtom(mol, 1.5f, 1.2f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 3.0f, 0.0f, 0.0f, ATOM_C);      // COOH
    addAtom(mol, 3.6f, 1.1f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 3.6f, -1.1f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 4.5f, -1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, 0.87f, ATOM_H);
    addAtom(mol, -0.5f, -0.5f, -0.87f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 1, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 3, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 0, 9, 1);

    centerMolecule(mol);
}

// Build Arginine (C6H14N4O2)
void buildArginine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Arginine (C6H14N4O2)");

    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Alpha C
    addAtom(mol, -1.3f, 0.5f, 0.0f, ATOM_N);     // NH2
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_O);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -1.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -3.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -4.5f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 0.0f, -6.0f, 0.0f, ATOM_N);     // NH
    addAtom(mol, 0.0f, -7.3f, 0.0f, ATOM_C);     // Guanidinium C
    addAtom(mol, -1.1f, -8.0f, 0.0f, ATOM_N);    // NH2
    addAtom(mol, 1.1f, -8.0f, 0.0f, ATOM_N);     // NH2
    // H atoms
    addAtom(mol, 0.0f, 0.5f, 0.9f, ATOM_H);
    addAtom(mol, -1.4f, 1.5f, 0.0f, ATOM_H);
    addAtom(mol, -2.1f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, 3.2f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 0.8f, -6.3f, 0.0f, ATOM_H);
    addAtom(mol, -1.1f, -9.0f, 0.0f, ATOM_H);
    addAtom(mol, -1.9f, -7.5f, 0.0f, ATOM_H);
    addAtom(mol, 1.1f, -9.0f, 0.0f, ATOM_H);
    addAtom(mol, 1.9f, -7.5f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 0, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 2, 4, 1);
    addBond(mol, 0, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 9, 11, 2);
    addBond(mol, 0, 12, 1);
    addBond(mol, 1, 13, 1);
    addBond(mol, 1, 14, 1);
    addBond(mol, 4, 15, 1);
    addBond(mol, 8, 16, 1);
    addBond(mol, 10, 17, 1);
    addBond(mol, 10, 18, 1);
    addBond(mol, 11, 19, 1);
    addBond(mol, 11, 20, 1);

    centerMolecule(mol);
}

// ============== VITAMINS ==============

// Build Vitamin C - L-Ascorbic Acid (C6H8O6)
void buildAscorbicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin C (C6H8O6)");

    // Lactone ring (5-membered)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.0f, -0.3f, 0.0f, ATOM_C);     // C3
    addAtom(mol, 1.3f, -1.4f, 0.0f, ATOM_O);     // O (ring)
    addAtom(mol, 0.0f, -1.2f, 0.0f, ATOM_C);     // C4 (lactone carbonyl)
    addAtom(mol, -1.0f, -1.8f, 0.0f, ATOM_O);    // =O (lactone)
    // Side chain
    addAtom(mol, 3.4f, -0.3f, 0.0f, ATOM_C);     // C5
    addAtom(mol, 4.2f, 0.9f, 0.0f, ATOM_C);      // C6
    // Hydroxyl groups
    addAtom(mol, -0.8f, 0.8f, 0.0f, ATOM_O);     // OH on C1
    addAtom(mol, 1.2f, 2.0f, 0.0f, ATOM_O);      // OH on C2
    addAtom(mol, 3.8f, -1.5f, 0.0f, ATOM_O);     // OH on C5
    addAtom(mol, 5.5f, 0.7f, 0.0f, ATOM_O);      // OH on C6
    // H atoms on OH
    addAtom(mol, -1.6f, 0.4f, 0.0f, ATOM_H);
    addAtom(mol, 1.2f, 2.7f, 0.0f, ATOM_H);
    addAtom(mol, 4.7f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, 5.8f, 1.5f, 0.0f, ATOM_H);
    // Other H atoms
    addAtom(mol, 3.6f, 0.2f, 0.9f, ATOM_H);
    addAtom(mol, 3.8f, 1.8f, 0.0f, ATOM_H);
    addAtom(mol, 4.0f, 0.9f, -0.9f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 0, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 2, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 8, 12, 1);
    addBond(mol, 9, 13, 1);
    addBond(mol, 10, 14, 1);
    addBond(mol, 11, 15, 1);
    addBond(mol, 6, 16, 1);
    addBond(mol, 7, 17, 1);
    addBond(mol, 7, 18, 1);

    centerMolecule(mol);
}

// Build Thiamine - Vitamin B1 (C12H17N4OS)
void buildThiamine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B1 (C12H17N4OS)");

    // Pyrimidine ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_N);      // N3
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6
    // Amino group on C4
    addAtom(mol, 3.6f, -2.0f, 0.0f, ATOM_N);     // NH2
    // Methyl on C2
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_C);      // CH3
    // Methylene bridge
    addAtom(mol, 1.2f, -3.6f, 0.0f, ATOM_C);     // CH2
    // Thiazole ring
    addAtom(mol, 1.2f, -5.0f, 0.0f, ATOM_C);     // C
    addAtom(mol, 0.0f, -5.7f, 0.0f, ATOM_N);     // N (positive)
    addAtom(mol, 0.0f, -7.1f, 0.0f, ATOM_C);     // C
    addAtom(mol, 1.2f, -7.8f, 0.0f, ATOM_C);     // C
    addAtom(mol, 2.2f, -6.8f, 0.0f, ATOM_S);     // S
    // Hydroxyethyl side chain
    addAtom(mol, 1.2f, -9.3f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 1.2f, -10.7f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, 1.2f, -12.0f, 0.0f, ATOM_O);    // OH
    // Methyl on thiazole
    addAtom(mol, 2.4f, -4.3f, 0.0f, ATOM_C);     // CH3
    // H atoms
    addAtom(mol, -0.9f, -1.8f, 0.0f, ATOM_H);
    addAtom(mol, 3.6f, -3.0f, 0.0f, ATOM_H);
    addAtom(mol, 4.4f, -1.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.9f, -7.5f, 0.0f, ATOM_H);
    addAtom(mol, 1.2f, -12.8f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 4, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 2);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 2);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 9, 1);
    addBond(mol, 12, 14, 1);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 9, 17, 1);
    addBond(mol, 5, 18, 1);
    addBond(mol, 6, 19, 1);
    addBond(mol, 6, 20, 1);
    addBond(mol, 11, 21, 1);
    addBond(mol, 16, 22, 1);

    centerMolecule(mol);
}

// Build Riboflavin - Vitamin B2 (C17H20N4O6)
void buildRiboflavin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B2 (C17H20N4O6)");

    // Isoalloxazine ring system (flavin)
    // Benzene ring (fused)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C5a
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C6
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);      // C7
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);     // C8
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);     // C9
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C9a
    // Pyrazine ring
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_N);     // N5
    addAtom(mol, -1.2f, -2.1f, 0.0f, ATOM_N);    // N10
    // Pyrimidine ring
    addAtom(mol, -2.4f, 0.0f, 0.0f, ATOM_C);     // C4a
    addAtom(mol, -2.4f, -1.4f, 0.0f, ATOM_C);    // C10a
    addAtom(mol, -3.6f, 0.7f, 0.0f, ATOM_C);     // C4 (=O)
    addAtom(mol, -3.6f, 2.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -4.8f, 0.0f, 0.0f, ATOM_N);     // N3
    addAtom(mol, -4.8f, -1.4f, 0.0f, ATOM_C);    // C2 (=O)
    addAtom(mol, -4.8f, -2.6f, 0.0f, ATOM_O);    // =O
    addAtom(mol, -3.6f, -2.1f, 0.0f, ATOM_N);    // N1
    // Methyl groups
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_C);      // CH3 on C6
    addAtom(mol, 2.4f, -2.8f, 0.0f, ATOM_C);     // CH3 on C8
    // Ribityl chain start
    addAtom(mol, -1.2f, -3.5f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.2f, -4.9f, 0.0f, ATOM_C);    // CHOH
    addAtom(mol, -1.2f, -6.3f, 0.0f, ATOM_C);    // CHOH
    addAtom(mol, -1.2f, -7.7f, 0.0f, ATOM_C);    // CHOH
    addAtom(mol, -1.2f, -9.1f, 0.0f, ATOM_C);    // CH2OH
    // Ribityl OH groups
    addAtom(mol, 0.0f, -5.2f, 0.0f, ATOM_O);
    addAtom(mol, -2.4f, -6.6f, 0.0f, ATOM_O);
    addAtom(mol, 0.0f, -8.0f, 0.0f, ATOM_O);
    addAtom(mol, -1.2f, -10.4f, 0.0f, ATOM_O);
    // Some H atoms
    addAtom(mol, 3.3f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -5.6f, 0.5f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 6, 8, 2);
    addBond(mol, 7, 9, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 8, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 10, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 14, 2);
    addBond(mol, 13, 15, 1);
    addBond(mol, 15, 9, 2);
    addBond(mol, 1, 16, 1);
    addBond(mol, 3, 17, 1);
    addBond(mol, 7, 18, 1);
    addBond(mol, 18, 19, 1);
    addBond(mol, 19, 20, 1);
    addBond(mol, 20, 21, 1);
    addBond(mol, 21, 22, 1);
    addBond(mol, 19, 23, 1);
    addBond(mol, 20, 24, 1);
    addBond(mol, 21, 25, 1);
    addBond(mol, 22, 26, 1);
    addBond(mol, 2, 27, 1);
    addBond(mol, 12, 28, 1);

    centerMolecule(mol);
}

// Build Niacin - Vitamin B3 / Nicotinic Acid (C6H5NO2)
void buildNiacin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B3 (C6H5NO2)");

    // Pyridine ring
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_N);        // N1
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);  // C2
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C); // C3
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);      // C4
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C); // C5
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C); // C6
    // Carboxylic acid on C3
    addAtom(mol, r * cosf(2*PI/3) * 1.8f, r * sinf(2*PI/3) * 1.8f + 0.6f, 0.0f, ATOM_C); // COOH C
    addAtom(mol, r * cosf(2*PI/3) * 2.2f, r * sinf(2*PI/3) * 2.2f + 1.5f, 0.0f, ATOM_O); // =O
    addAtom(mol, r * cosf(2*PI/3) * 2.4f, r * sinf(2*PI/3) * 1.2f, 0.0f, ATOM_O);        // OH
    addAtom(mol, r * cosf(2*PI/3) * 3.2f, r * sinf(2*PI/3) * 1.0f, 0.0f, ATOM_H);        // H on OH
    // H atoms on ring
    float rH = 2.4f;
    addAtom(mol, rH * cosf(PI/3), rH * sinf(PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(PI), rH * sinf(PI), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(4*PI/3), rH * sinf(4*PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(5*PI/3), rH * sinf(5*PI/3), 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 3, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 5, 13, 1);

    centerMolecule(mol);
}

// Build Pantothenic Acid - Vitamin B5 (C9H17NO5)
void buildPanthothenicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B5 (C9H17NO5)");

    // Beta-alanine part
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // COOH C
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 1.1f, -0.6f, 0.0f, ATOM_O);     // OH
    addAtom(mol, -1.3f, -0.7f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, -3.9f, -0.7f, 0.0f, ATOM_N);    // NH
    // Amide linkage
    addAtom(mol, -5.2f, 0.0f, 0.0f, ATOM_C);     // C=O
    addAtom(mol, -5.2f, 1.2f, 0.0f, ATOM_O);     // =O
    // Pantoic acid part
    addAtom(mol, -6.5f, -0.7f, 0.0f, ATOM_C);    // C (with OH)
    addAtom(mol, -6.5f, -2.1f, 0.0f, ATOM_O);    // OH
    addAtom(mol, -7.8f, 0.0f, 0.0f, ATOM_C);     // C (quaternary)
    addAtom(mol, -7.8f, 1.4f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -7.8f, -1.4f, 0.0f, ATOM_C);    // CH3
    addAtom(mol, -9.1f, 0.0f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, -10.4f, 0.0f, 0.0f, ATOM_O);    // OH
    // H atoms
    addAtom(mol, 1.8f, -0.1f, 0.0f, ATOM_H);
    addAtom(mol, -3.9f, -1.6f, 0.0f, ATOM_H);
    addAtom(mol, -6.5f, -2.8f, 0.0f, ATOM_H);
    addAtom(mol, -11.1f, 0.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 8, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 10, 12, 1);
    addBond(mol, 10, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 2, 15, 1);
    addBond(mol, 5, 16, 1);
    addBond(mol, 9, 17, 1);
    addBond(mol, 14, 18, 1);

    centerMolecule(mol);
}

// Build Pyridoxine - Vitamin B6 (C8H11NO3)
void buildPyridoxine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B6 (C8H11NO3)");

    // Pyridine ring
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_N);        // N1
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);  // C2
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C); // C3
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);      // C4
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C); // C5
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C); // C6
    // CH3 on C2
    addAtom(mol, r * cosf(PI/3) * 1.8f, r * sinf(PI/3) * 1.8f, 0.0f, ATOM_C);
    // CH2OH on C4
    addAtom(mol, r * cosf(PI) * 1.8f, r * sinf(PI) * 1.8f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI) * 2.8f, r * sinf(PI) * 2.0f, 0.0f, ATOM_O);
    // OH on C3
    addAtom(mol, r * cosf(2*PI/3) * 1.8f, r * sinf(2*PI/3) * 1.8f, 0.0f, ATOM_O);
    // CH2OH on C5
    addAtom(mol, r * cosf(4*PI/3) * 1.8f, r * sinf(4*PI/3) * 1.8f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(4*PI/3) * 2.8f, r * sinf(4*PI/3) * 2.0f, 0.0f, ATOM_O);
    // H atoms
    addAtom(mol, r * cosf(5*PI/3) * 1.7f, r * sinf(5*PI/3) * 1.7f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/3) * 2.5f, r * sinf(2*PI/3) * 2.5f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(PI) * 3.5f, r * sinf(PI) * 2.3f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(4*PI/3) * 3.5f, r * sinf(4*PI/3) * 2.3f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 1, 6, 1);
    addBond(mol, 3, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 4, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 5, 12, 1);
    addBond(mol, 9, 13, 1);
    addBond(mol, 8, 14, 1);
    addBond(mol, 11, 15, 1);

    centerMolecule(mol);
}

// Build Biotin - Vitamin B7 (C10H16N2O3S)
void buildBiotin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B7 (C10H16N2O3S)");

    // Ureido ring (imidazolidinone)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C=O (urea)
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_O);      // =O
    addAtom(mol, -1.1f, -0.7f, 0.0f, ATOM_N);    // NH
    addAtom(mol, 1.1f, -0.7f, 0.0f, ATOM_N);     // NH
    addAtom(mol, -1.1f, -2.1f, 0.0f, ATOM_C);    // CH
    addAtom(mol, 1.1f, -2.1f, 0.0f, ATOM_C);     // CH
    // Tetrahydrothiophene ring
    addAtom(mol, 0.0f, -2.8f, 0.0f, ATOM_S);     // S
    addAtom(mol, -1.1f, -3.5f, 0.0f, ATOM_C);    // CH
    addAtom(mol, 1.1f, -3.5f, 0.0f, ATOM_C);     // CH2
    // Valeric acid side chain
    addAtom(mol, -1.1f, -5.0f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.1f, -6.5f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.1f, -8.0f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.1f, -9.5f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.1f, -11.0f, 0.0f, ATOM_C);   // COOH
    addAtom(mol, -1.1f, -12.2f, 0.0f, ATOM_O);   // =O
    addAtom(mol, 0.0f, -11.5f, 0.0f, ATOM_O);    // OH
    // H atoms
    addAtom(mol, -1.8f, -0.2f, 0.0f, ATOM_H);
    addAtom(mol, 1.8f, -0.2f, 0.0f, ATOM_H);
    addAtom(mol, 0.7f, -11.9f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 2, 4, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 4, 6, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 6, 8, 1);
    addBond(mol, 4, 7, 1);
    addBond(mol, 5, 8, 1);
    addBond(mol, 7, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 14, 2);
    addBond(mol, 13, 15, 1);
    addBond(mol, 2, 16, 1);
    addBond(mol, 3, 17, 1);
    addBond(mol, 15, 18, 1);

    centerMolecule(mol);
}

// Build Folic Acid - Vitamin B9 (C19H19N7O6) - Simplified representation
void buildFolicAcid(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin B9 (C19H19N7O6)");

    // Pteridine ring system
    // Pyrimidine
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_N);      // N3
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);     // C4a
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C8a
    // Pyrazine fused
    addAtom(mol, 1.2f, -3.5f, 0.0f, ATOM_N);     // N5
    addAtom(mol, 2.4f, -4.2f, 0.0f, ATOM_C);     // C6
    addAtom(mol, 3.6f, -3.5f, 0.0f, ATOM_C);     // C7
    addAtom(mol, 3.6f, -2.1f, 0.0f, ATOM_N);     // N8
    // Amino on C2
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_N);      // NH2
    // =O on C4
    addAtom(mol, 3.6f, -0.7f, 0.0f, ATOM_O);     // =O
    // p-Aminobenzoic acid part (simplified)
    addAtom(mol, 2.4f, -5.6f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 2.4f, -7.0f, 0.0f, ATOM_N);     // NH
    // Benzene ring
    addAtom(mol, 2.4f, -8.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -9.1f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -10.5f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -11.2f, 0.0f, ATOM_C);
    addAtom(mol, 3.6f, -10.5f, 0.0f, ATOM_C);
    addAtom(mol, 3.6f, -9.1f, 0.0f, ATOM_C);
    // COOH on benzene
    addAtom(mol, 2.4f, -12.6f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -13.3f, 0.0f, ATOM_O);
    addAtom(mol, 3.6f, -13.3f, 0.0f, ATOM_N);    // NH to glutamate
    // Glutamate (simplified)
    addAtom(mol, 4.8f, -12.6f, 0.0f, ATOM_C);    // CH
    addAtom(mol, 6.0f, -13.3f, 0.0f, ATOM_C);    // COOH
    addAtom(mol, 6.0f, -14.5f, 0.0f, ATOM_O);
    addAtom(mol, 7.1f, -12.6f, 0.0f, ATOM_O);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 4, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 3, 1);
    addBond(mol, 1, 10, 1);
    addBond(mol, 3, 11, 1);
    addBond(mol, 7, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 19, 2);
    addBond(mol, 19, 14, 1);
    addBond(mol, 17, 20, 1);
    addBond(mol, 20, 21, 2);
    addBond(mol, 20, 22, 1);
    addBond(mol, 22, 23, 1);
    addBond(mol, 23, 24, 1);
    addBond(mol, 24, 25, 2);
    addBond(mol, 24, 26, 1);

    centerMolecule(mol);
}

// Build Retinol - Vitamin A (C20H30O)
void buildRetinol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin A (C20H30O)");

    // Cyclohexene ring (beta-ionone)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C3
    addAtom(mol, 2.6f, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 1.3f, -2.1f, 0.0f, ATOM_C);     // C5 (C=C)
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6 (C=C)
    // Methyls on C1
    addAtom(mol, -1.1f, 0.7f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, 0.0f, 1.0f, 1.0f, ATOM_C);      // CH3
    // Methyl on C5
    addAtom(mol, 1.3f, -3.5f, 0.0f, ATOM_C);     // CH3
    // Polyene chain
    addAtom(mol, -1.3f, -2.1f, 0.0f, ATOM_C);    // C7
    addAtom(mol, -2.6f, -1.4f, 0.0f, ATOM_C);    // C8
    addAtom(mol, -3.9f, -2.1f, 0.0f, ATOM_C);    // C9
    addAtom(mol, -3.9f, -3.5f, 0.0f, ATOM_C);    // CH3 on C9
    addAtom(mol, -5.2f, -1.4f, 0.0f, ATOM_C);    // C10
    addAtom(mol, -6.5f, -2.1f, 0.0f, ATOM_C);    // C11
    addAtom(mol, -7.8f, -1.4f, 0.0f, ATOM_C);    // C12
    addAtom(mol, -9.1f, -2.1f, 0.0f, ATOM_C);    // C13
    addAtom(mol, -9.1f, -3.5f, 0.0f, ATOM_C);    // CH3 on C13
    addAtom(mol, -10.4f, -1.4f, 0.0f, ATOM_C);   // C14
    addAtom(mol, -11.7f, -2.1f, 0.0f, ATOM_C);   // C15 (CH2OH)
    addAtom(mol, -13.0f, -1.4f, 0.0f, ATOM_O);   // OH
    addAtom(mol, -13.7f, -2.0f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 0, 7, 1);
    addBond(mol, 4, 8, 1);
    addBond(mol, 5, 9, 1);
    addBond(mol, 9, 10, 2);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 11, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 16, 18, 1);
    addBond(mol, 17, 19, 1);
    addBond(mol, 19, 20, 1);
    addBond(mol, 20, 21, 1);

    centerMolecule(mol);
}

// Build Beta-Carotene (C40H56) - Provitamin A - Simplified
void buildBetaCarotene(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Beta-Carotene (C40H56)");

    // Simplified: Two beta-ionone rings connected by polyene chain
    // Left ring
    addAtom(mol, -8.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -7.0f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, -5.6f, 1.0f, 0.0f, ATOM_C);
    addAtom(mol, -5.0f, -0.3f, 0.0f, ATOM_C);
    addAtom(mol, -5.8f, -1.5f, 0.0f, ATOM_C);
    addAtom(mol, -7.2f, -1.3f, 0.0f, ATOM_C);
    // Central polyene (simplified)
    addAtom(mol, -3.6f, -0.5f, 0.0f, ATOM_C);
    addAtom(mol, -2.4f, 0.2f, 0.0f, ATOM_C);
    addAtom(mol, -1.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 0.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 1.6f, 0.5f, 0.0f, ATOM_C);
    addAtom(mol, 2.8f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, 4.2f, 1.0f, 0.0f, ATOM_C);
    // Right ring
    addAtom(mol, 5.4f, 0.3f, 0.0f, ATOM_C);
    addAtom(mol, 5.8f, -1.1f, 0.0f, ATOM_C);
    addAtom(mol, 7.2f, -1.3f, 0.0f, ATOM_C);
    addAtom(mol, 8.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 7.0f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, 5.6f, 1.4f, 0.0f, ATOM_C);
    // Methyls
    addAtom(mol, -8.8f, 0.8f, 0.0f, ATOM_C);
    addAtom(mol, 8.8f, -0.8f, 0.0f, ATOM_C);
    addAtom(mol, -1.0f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.6f, -0.9f, 0.0f, ATOM_C);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 13, 1);
    addBond(mol, 0, 19, 1);
    addBond(mol, 16, 20, 1);
    addBond(mol, 8, 21, 1);
    addBond(mol, 10, 22, 1);

    centerMolecule(mol);
}

// Build Cholecalciferol - Vitamin D3 (C27H44O)
void buildCholecalciferol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin D3 (C27H44O)");

    // Secosteroid structure - simplified
    // Ring A (opened)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C3 (with OH)
    addAtom(mol, 2.6f, -1.4f, 0.0f, ATOM_O);     // OH
    addAtom(mol, 2.6f, -2.2f, 0.0f, ATOM_H);
    addAtom(mol, 3.9f, 0.7f, 0.0f, ATOM_C);      // C4
    addAtom(mol, 3.9f, 2.1f, 0.0f, ATOM_C);      // C5
    addAtom(mol, 2.6f, 2.8f, 0.0f, ATOM_C);      // C6
    addAtom(mol, 1.3f, 2.1f, 0.0f, ATOM_C);      // C7
    addAtom(mol, 0.0f, 2.8f, 0.0f, ATOM_C);      // C8
    // Ring C
    addAtom(mol, -1.3f, 2.1f, 0.0f, ATOM_C);     // C9
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_C);     // C10
    addAtom(mol, -2.6f, 2.8f, 0.0f, ATOM_C);     // C11
    addAtom(mol, -3.9f, 2.1f, 0.0f, ATOM_C);     // C12
    addAtom(mol, -3.9f, 0.7f, 0.0f, ATOM_C);     // C13
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);     // C14
    // Ring D
    addAtom(mol, -5.2f, 0.0f, 0.0f, ATOM_C);     // C15
    addAtom(mol, -6.5f, 0.7f, 0.0f, ATOM_C);     // C16
    addAtom(mol, -6.5f, 2.1f, 0.0f, ATOM_C);     // C17
    addAtom(mol, -5.2f, 2.8f, 0.0f, ATOM_C);     // C18 (CH3)
    // Side chain
    addAtom(mol, -7.8f, 2.8f, 0.0f, ATOM_C);     // C20
    addAtom(mol, -7.8f, 4.2f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -9.1f, 2.1f, 0.0f, ATOM_C);     // C22
    addAtom(mol, -10.4f, 2.8f, 0.0f, ATOM_C);    // C23
    addAtom(mol, -11.7f, 2.1f, 0.0f, ATOM_C);    // C24
    addAtom(mol, -13.0f, 2.8f, 0.0f, ATOM_C);    // C25
    addAtom(mol, -14.3f, 2.1f, 0.0f, ATOM_C);    // C26 (CH3)
    addAtom(mol, -13.0f, 4.2f, 0.0f, ATOM_C);    // C27 (CH3)

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 5, 6, 2);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 0, 1);
    addBond(mol, 9, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 10, 1);
    addBond(mol, 14, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 13, 1);
    addBond(mol, 13, 19, 1);
    addBond(mol, 18, 20, 1);
    addBond(mol, 20, 21, 1);
    addBond(mol, 20, 22, 1);
    addBond(mol, 22, 23, 1);
    addBond(mol, 23, 24, 1);
    addBond(mol, 24, 25, 1);
    addBond(mol, 25, 26, 1);
    addBond(mol, 25, 27, 1);

    centerMolecule(mol);
}

// Build Alpha-Tocopherol - Vitamin E (C29H50O2) - Simplified
void buildAlphaTocopherol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin E (C29H50O2)");

    // Chromanol ring (benzene + pyran)
    float r = 1.4f;
    // Benzene part
    addAtom(mol, r, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 0.5f, r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.5f, r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, -r, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.5f, -r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, r * 0.5f, -r * 0.866f, 0.0f, ATOM_C);
    // Pyran oxygen
    addAtom(mol, r * 1.8f, 0.0f, 0.0f, ATOM_O);
    // Pyran carbons
    addAtom(mol, r * 2.2f, r * 0.7f, 0.0f, ATOM_C);
    addAtom(mol, r * 1.5f, r * 1.5f, 0.0f, ATOM_C);
    // OH on benzene
    addAtom(mol, -r * 1.8f, 0.0f, 0.0f, ATOM_O);
    addAtom(mol, -r * 2.5f, 0.0f, 0.0f, ATOM_H);
    // Methyls on benzene
    addAtom(mol, r * 0.7f, r * 1.5f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.7f, r * 1.5f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.7f, -r * 1.5f, 0.0f, ATOM_C);
    // Phytyl tail (simplified)
    addAtom(mol, r * 3.5f, r * 0.7f, 0.0f, ATOM_C);
    addAtom(mol, r * 4.8f, r * 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 6.1f, r * 0.7f, 0.0f, ATOM_C);
    addAtom(mol, r * 7.4f, r * 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 8.7f, r * 0.7f, 0.0f, ATOM_C);
    addAtom(mol, r * 10.0f, r * 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 3.5f, r * 2.0f, 0.0f, ATOM_C);  // CH3 branches
    addAtom(mol, r * 6.1f, r * 2.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 8.7f, r * 2.0f, 0.0f, ATOM_C);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 1, 1);
    addBond(mol, 3, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 2, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 7, 14, 1);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 19, 1);
    addBond(mol, 14, 20, 1);
    addBond(mol, 16, 21, 1);
    addBond(mol, 18, 22, 1);

    centerMolecule(mol);
}

// Build Phylloquinone - Vitamin K1 (C31H46O2) - Simplified
void buildPhylloquinone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Vitamin K1 (C31H46O2)");

    // Naphthoquinone core
    // First benzene ring
    float r = 1.4f;
    addAtom(mol, r, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 0.5f, r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.5f, r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, -r, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -r * 0.5f, -r * 0.866f, 0.0f, ATOM_C);
    addAtom(mol, r * 0.5f, -r * 0.866f, 0.0f, ATOM_C);
    // Quinone ring (fused)
    addAtom(mol, r * 1.5f, r * 0.866f, 0.0f, ATOM_C);   // C=O
    addAtom(mol, r * 2.0f, r * 1.5f, 0.0f, ATOM_O);
    addAtom(mol, r * 2.0f, r * 0.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 2.0f, -r * 1.0f, 0.0f, ATOM_C);
    addAtom(mol, r * 1.5f, -r * 0.866f, 0.0f, ATOM_C);  // C=O
    addAtom(mol, r * 2.0f, -r * 1.8f, 0.0f, ATOM_O);
    // Methyl on quinone
    addAtom(mol, r * 3.3f, r * 0.0f, 0.0f, ATOM_C);
    // Phytyl tail (simplified)
    addAtom(mol, r * 3.3f, -r * 1.2f, 0.0f, ATOM_C);
    addAtom(mol, r * 4.6f, -r * 1.2f, 0.0f, ATOM_C);
    addAtom(mol, r * 5.9f, -r * 1.2f, 0.0f, ATOM_C);
    addAtom(mol, r * 7.2f, -r * 1.2f, 0.0f, ATOM_C);
    addAtom(mol, r * 8.5f, -r * 1.2f, 0.0f, ATOM_C);
    // H atoms
    addAtom(mol, -r * 1.7f, 0.0f, 0.0f, ATOM_H);
    addAtom(mol, -r * 0.8f, r * 1.6f, 0.0f, ATOM_H);
    addAtom(mol, -r * 0.8f, -r * 1.6f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 10, 5, 1);
    addBond(mol, 8, 12, 1);
    addBond(mol, 9, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 3, 18, 1);
    addBond(mol, 2, 19, 1);
    addBond(mol, 4, 20, 1);

    centerMolecule(mol);
}

// Build Nicotinamide (Niacinamide) - B3 form (C6H6N2O)
void buildNicotinamide(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Nicotinamide (C6H6N2O)");

    // Pyridine ring
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_N);
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C);
    // Amide on C3
    addAtom(mol, r * cosf(2*PI/3) * 1.8f, r * sinf(2*PI/3) * 1.8f + 0.6f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(2*PI/3) * 2.2f, r * sinf(2*PI/3) * 2.2f + 1.5f, 0.0f, ATOM_O);
    addAtom(mol, r * cosf(2*PI/3) * 2.4f, r * sinf(2*PI/3) * 1.2f, 0.0f, ATOM_N);
    // H atoms
    addAtom(mol, r * cosf(2*PI/3) * 3.0f, r * sinf(2*PI/3) * 0.8f, 0.0f, ATOM_H);
    addAtom(mol, r * cosf(2*PI/3) * 2.8f, r * sinf(2*PI/3) * 1.8f, 0.0f, ATOM_H);
    float rH = 2.4f;
    addAtom(mol, rH * cosf(PI/3), rH * sinf(PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(PI), rH * sinf(PI), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(4*PI/3), rH * sinf(4*PI/3), 0.0f, ATOM_H);
    addAtom(mol, rH * cosf(5*PI/3), rH * sinf(5*PI/3), 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 8, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 3, 12, 1);
    addBond(mol, 4, 13, 1);
    addBond(mol, 5, 14, 1);

    centerMolecule(mol);
}

// ============== ADDITIONAL COMPOUNDS ==============

// Build Cocaine (C17H21NO4) - Tropane alkaloid
void buildCocaine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cocaine (C17H21NO4)");

    // Tropane ring system (bicyclic)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N (bridgehead)
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C3 (with ester)
    addAtom(mol, 2.6f, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 1.3f, -2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6
    addAtom(mol, -1.3f, -0.7f, 0.0f, ATOM_C);    // C7
    // Bridge carbon
    addAtom(mol, 1.3f, -0.7f, 1.2f, ATOM_C);     // C1 (bridge)
    // N-methyl
    addAtom(mol, -0.5f, 1.0f, 0.0f, ATOM_C);     // CH3 on N
    // Benzoyl ester at C3
    addAtom(mol, 3.9f, 0.7f, 0.0f, ATOM_C);      // C=O ester
    addAtom(mol, 3.9f, 1.9f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 5.2f, 0.0f, 0.0f, ATOM_O);      // O-benzene
    // Benzene ring
    addAtom(mol, 6.5f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 7.7f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 8.9f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 8.9f, 2.1f, 0.0f, ATOM_C);
    addAtom(mol, 7.7f, 2.8f, 0.0f, ATOM_C);
    addAtom(mol, 6.5f, 2.1f, 0.0f, ATOM_C);
    // Methyl ester at C2
    addAtom(mol, 1.3f, 2.1f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 0.2f, 2.7f, 0.0f, ATOM_O);      // =O
    addAtom(mol, 2.4f, 2.7f, 0.0f, ATOM_O);      // O-CH3
    addAtom(mol, 2.4f, 4.0f, 0.0f, ATOM_C);      // CH3

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 0, 1);
    addBond(mol, 1, 7, 1);
    addBond(mol, 4, 7, 1);
    addBond(mol, 0, 8, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 9, 10, 2);
    addBond(mol, 9, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 17, 12, 1);
    addBond(mol, 1, 18, 1);
    addBond(mol, 18, 19, 2);
    addBond(mol, 18, 20, 1);
    addBond(mol, 20, 21, 1);

    centerMolecule(mol);
}

// Build Heroin / Diacetylmorphine (C21H23NO5)
void buildHeroin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Heroin (C21H23NO5)");

    // Morphine core - pentacyclic structure (simplified)
    // Ring A (benzene)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);
    // Ring B (cyclohexene fused)
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, -1.3f, 2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 2.8f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 2.1f, 0.0f, ATOM_C);
    // Ring C with oxygen bridge
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -2.6f, -1.4f, 0.0f, ATOM_O);    // O bridge
    addAtom(mol, -1.3f, -2.1f, 0.0f, ATOM_C);
    // Ring D (piperidine with N)
    addAtom(mol, -2.6f, 2.8f, 0.0f, ATOM_C);
    addAtom(mol, -3.9f, 2.1f, 0.0f, ATOM_C);
    addAtom(mol, -3.9f, 0.7f, 0.0f, ATOM_N);     // N
    addAtom(mol, -4.5f, 0.0f, 0.0f, ATOM_C);     // N-CH3
    // Acetyl groups (diacetyl = heroin)
    addAtom(mol, 3.6f, 0.7f, 0.0f, ATOM_O);      // O-acetyl 1
    addAtom(mol, 4.8f, 0.0f, 0.0f, ATOM_C);      // C=O
    addAtom(mol, 4.8f, -1.2f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 6.0f, 0.7f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, 3.6f, -2.1f, 0.0f, ATOM_O);     // O-acetyl 2
    addAtom(mol, 4.8f, -2.8f, 0.0f, ATOM_C);     // C=O
    addAtom(mol, 4.8f, -4.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 6.0f, -2.1f, 0.0f, ATOM_C);     // CH3

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 1, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 5, 1);
    addBond(mol, 7, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 10, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 2, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 19, 2);
    addBond(mol, 18, 20, 1);
    addBond(mol, 3, 21, 1);
    addBond(mol, 21, 22, 1);
    addBond(mol, 22, 23, 2);
    addBond(mol, 22, 24, 1);

    centerMolecule(mol);
}

// Build Fentanyl (C22H28N2O) - Synthetic opioid
void buildFentanyl(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Fentanyl/Sublimaze (C22H28N2O)");

    // Piperidine ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C3
    addAtom(mol, 2.6f, -1.4f, 0.0f, ATOM_C);     // C4 (with phenethyl)
    addAtom(mol, 1.3f, -2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6
    // Phenethyl group on C4
    addAtom(mol, 3.9f, -2.1f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 5.2f, -1.4f, 0.0f, ATOM_C);     // CH2
    // Benzene ring
    addAtom(mol, 6.5f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 7.7f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 8.9f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 8.9f, -3.5f, 0.0f, ATOM_C);
    addAtom(mol, 7.7f, -4.2f, 0.0f, ATOM_C);
    addAtom(mol, 6.5f, -3.5f, 0.0f, ATOM_C);
    // N-phenyl-propanamide on N1
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_C);     // C=O
    addAtom(mol, -1.3f, 2.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);     // CH2CH3
    addAtom(mol, -3.9f, 0.7f, 0.0f, ATOM_C);     // CH3
    // Aniline (N-phenyl)
    addAtom(mol, -1.3f, -2.1f, 0.0f, ATOM_N);    // N-aniline
    addAtom(mol, -2.6f, -2.8f, 0.0f, ATOM_C);    // benzene
    addAtom(mol, -2.6f, -4.2f, 0.0f, ATOM_C);
    addAtom(mol, -3.8f, -4.9f, 0.0f, ATOM_C);
    addAtom(mol, -5.0f, -4.2f, 0.0f, ATOM_C);
    addAtom(mol, -5.0f, -2.8f, 0.0f, ATOM_C);
    addAtom(mol, -3.8f, -2.1f, 0.0f, ATOM_C);

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 3, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 8, 1);
    addBond(mol, 0, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 14, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 5, 18, 1);
    addBond(mol, 18, 19, 1);
    addBond(mol, 19, 20, 2);
    addBond(mol, 20, 21, 1);
    addBond(mol, 21, 22, 2);
    addBond(mol, 22, 23, 1);
    addBond(mol, 23, 24, 2);
    addBond(mol, 24, 19, 1);

    centerMolecule(mol);
}

// Build Propofol (C12H18O) - General anesthetic
void buildPropofol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Propofol/Diprivan (C12H18O)");

    // Benzene ring with OH and two isopropyl groups
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_C);           // C1 (OH)
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);     // C2 (iPr)
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C); // C3
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);         // C4
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C); // C5
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C); // C6 (iPr)
    // OH on C1
    addAtom(mol, r * 1.8f * cosf(0), r * 1.8f * sinf(0), 0.0f, ATOM_O);
    addAtom(mol, r * 2.5f * cosf(0), r * 2.5f * sinf(0), 0.0f, ATOM_H);
    // Isopropyl on C2
    addAtom(mol, r * 1.8f * cosf(PI/3), r * 1.8f * sinf(PI/3), 0.0f, ATOM_C);  // CH
    addAtom(mol, r * 2.2f * cosf(PI/4), r * 2.2f * sinf(PI/4), 0.8f, ATOM_C);  // CH3
    addAtom(mol, r * 2.2f * cosf(PI/4), r * 2.2f * sinf(PI/4), -0.8f, ATOM_C); // CH3
    // Isopropyl on C6
    addAtom(mol, r * 1.8f * cosf(5*PI/3), r * 1.8f * sinf(5*PI/3), 0.0f, ATOM_C);  // CH
    addAtom(mol, r * 2.2f * cosf(11*PI/6), r * 2.2f * sinf(11*PI/6), 0.8f, ATOM_C);  // CH3
    addAtom(mol, r * 2.2f * cosf(11*PI/6), r * 2.2f * sinf(11*PI/6), -0.8f, ATOM_C); // CH3
    // H atoms on ring
    addAtom(mol, r * 1.7f * cosf(2*PI/3), r * 1.7f * sinf(2*PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(PI), r * 1.7f * sinf(PI), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(4*PI/3), r * 1.7f * sinf(4*PI/3), 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 1, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 8, 10, 1);
    addBond(mol, 5, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 11, 13, 1);
    addBond(mol, 2, 14, 1);
    addBond(mol, 3, 15, 1);
    addBond(mol, 4, 16, 1);

    centerMolecule(mol);
}

// Build THC - Tetrahydrocannabinol (C21H30O2)
void buildTHC(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "THC (C21H30O2)");

    // Dibenzopyran core
    // Benzene ring A
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);
    // Pyran ring (with O)
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, -2.4f, 0.0f, 0.0f, ATOM_O);     // O in pyran
    addAtom(mol, -2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, -1.2f, -2.1f, 0.0f, ATOM_C);
    // Cyclohexene ring C
    addAtom(mol, -1.2f, -3.5f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -4.2f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -3.5f, 0.0f, ATOM_C);
    // OH on benzene
    addAtom(mol, 3.6f, 0.7f, 0.0f, ATOM_O);
    addAtom(mol, 4.3f, 0.2f, 0.0f, ATOM_H);
    // Pentyl chain
    addAtom(mol, 3.6f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 4.9f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 6.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 7.5f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 8.8f, -2.1f, 0.0f, ATOM_C);
    // Methyls on cyclohexene
    addAtom(mol, -2.4f, -4.2f, 0.0f, ATOM_C);    // gem-dimethyl
    addAtom(mol, -2.4f, -3.0f, 1.0f, ATOM_C);
    addAtom(mol, -1.2f, 2.0f, 0.0f, ATOM_C);     // CH3 on C

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 5, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 4, 1);
    addBond(mol, 2, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 3, 15, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 19, 1);
    addBond(mol, 10, 20, 1);
    addBond(mol, 10, 21, 1);
    addBond(mol, 6, 22, 1);

    centerMolecule(mol);
}

// Build Creatine (C4H9N3O2) - Sports supplement
void buildCreatine(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Creatine (C4H9N3O2)");

    // Guanidino group
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // Central C of guanidino
    addAtom(mol, 0.0f, 1.2f, 0.0f, ATOM_N);      // =NH
    addAtom(mol, -1.1f, -0.7f, 0.0f, ATOM_N);    // NH2
    addAtom(mol, 1.1f, -0.7f, 0.0f, ATOM_N);     // N-CH3
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);      // CH3 on N
    // Acetic acid part
    addAtom(mol, 1.1f, -2.1f, 0.0f, ATOM_C);     // CH2
    addAtom(mol, 2.4f, -2.8f, 0.0f, ATOM_C);     // COOH
    addAtom(mol, 2.4f, -4.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 3.5f, -2.1f, 0.0f, ATOM_O);     // OH
    // H atoms
    addAtom(mol, 0.0f, 2.0f, 0.0f, ATOM_H);      // H on =NH
    addAtom(mol, -1.8f, -0.2f, 0.0f, ATOM_H);    // H on NH2
    addAtom(mol, -1.4f, -1.5f, 0.0f, ATOM_H);    // H on NH2
    addAtom(mol, 4.3f, -2.5f, 0.0f, ATOM_H);     // H on OH

    addBond(mol, 0, 1, 2);
    addBond(mol, 0, 2, 1);
    addBond(mol, 0, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 3, 5, 1);
    addBond(mol, 5, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 6, 8, 1);
    addBond(mol, 1, 9, 1);
    addBond(mol, 2, 10, 1);
    addBond(mol, 2, 11, 1);
    addBond(mol, 8, 12, 1);

    centerMolecule(mol);
}

// Build Octane (C8H18) - Representative gasoline component
void buildOctane(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Octane/Gasoline (C8H18)");

    // Linear 8-carbon chain
    for (int i = 0; i < 8; i++) {
        float x = i * 1.5f;
        float y = (i % 2 == 0) ? 0.0f : 0.3f;  // Slight zigzag
        addAtom(mol, x, y, 0.0f, ATOM_C);
    }
    // H atoms on each carbon (simplified - 3H on terminals, 2H on others)
    // Terminal CH3 groups
    addAtom(mol, -0.8f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.8f, -0.5f, 0.0f, ATOM_H);
    addAtom(mol, -0.5f, 0.0f, 0.8f, ATOM_H);
    addAtom(mol, 11.3f, 0.8f, 0.0f, ATOM_H);
    addAtom(mol, 11.3f, -0.2f, 0.0f, ATOM_H);
    addAtom(mol, 11.0f, 0.3f, 0.8f, ATOM_H);
    // CH2 groups (2H each for carbons 1-6)
    for (int i = 1; i < 7; i++) {
        float x = i * 1.5f;
        float y = (i % 2 == 0) ? 0.0f : 0.3f;
        addAtom(mol, x, y + 0.8f, 0.0f, ATOM_H);
        addAtom(mol, x, y - 0.8f, 0.0f, ATOM_H);
    }

    // C-C backbone bonds
    for (int i = 0; i < 7; i++) {
        addBond(mol, i, i + 1, 1);
    }
    // C-H bonds for terminal methyls
    addBond(mol, 0, 8, 1);
    addBond(mol, 0, 9, 1);
    addBond(mol, 0, 10, 1);
    addBond(mol, 7, 11, 1);
    addBond(mol, 7, 12, 1);
    addBond(mol, 7, 13, 1);
    // C-H bonds for CH2 groups
    for (int i = 0; i < 6; i++) {
        addBond(mol, i + 1, 14 + i * 2, 1);
        addBond(mol, i + 1, 15 + i * 2, 1);
    }

    centerMolecule(mol);
}

// ============== STATINS & NSAIDs ==============

// Build Simvastatin (C25H38O5) - Statin cholesterol drug
void buildSimvastatin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Simvastatin/Zocor (C25H38O5)");

    // Decalin-like fused ring system (simplified)
    // Ring A (cyclohexene)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C1
    addAtom(mol, 1.3f, 0.7f, 0.0f, ATOM_C);      // C2
    addAtom(mol, 2.6f, 0.0f, 0.0f, ATOM_C);      // C3
    addAtom(mol, 2.6f, -1.4f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 1.3f, -2.1f, 0.0f, ATOM_C);     // C5
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);     // C6
    // Ring B (fused cyclohexane)
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_C);     // C7
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);     // C8
    addAtom(mol, -2.6f, -1.4f, 0.0f, ATOM_C);    // C9
    addAtom(mol, -1.3f, -2.1f, 0.0f, ATOM_C);    // C10
    // Lactone ring
    addAtom(mol, 3.9f, 0.7f, 0.0f, ATOM_C);      // C11
    addAtom(mol, 5.2f, 0.0f, 0.0f, ATOM_O);      // O (lactone)
    addAtom(mol, 5.2f, -1.4f, 0.0f, ATOM_C);     // C12 (C=O)
    addAtom(mol, 5.2f, -2.6f, 0.0f, ATOM_O);     // =O
    addAtom(mol, 3.9f, -0.7f, 0.0f, ATOM_C);     // C13
    // Hydroxy groups
    addAtom(mol, 3.9f, 2.0f, 0.0f, ATOM_O);      // OH
    addAtom(mol, 3.9f, -2.0f, 0.0f, ATOM_O);     // OH
    // Side chain ester
    addAtom(mol, -3.9f, 0.7f, 0.0f, ATOM_O);     // O-ester
    addAtom(mol, -5.2f, 0.0f, 0.0f, ATOM_C);     // C=O
    addAtom(mol, -5.2f, -1.2f, 0.0f, ATOM_O);    // =O
    addAtom(mol, -6.5f, 0.7f, 0.0f, ATOM_C);     // CH(CH3)2
    addAtom(mol, -7.8f, 0.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -6.5f, 2.0f, 0.0f, ATOM_C);     // CH3
    // Methyls
    addAtom(mol, 1.3f, 2.0f, 0.0f, ATOM_C);      // CH3
    addAtom(mol, -1.3f, -3.5f, 0.0f, ATOM_C);    // CH3

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 1);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 5, 1);
    addBond(mol, 2, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 12, 14, 1);
    addBond(mol, 14, 3, 1);
    addBond(mol, 10, 15, 1);
    addBond(mol, 14, 16, 1);
    addBond(mol, 7, 17, 1);
    addBond(mol, 17, 18, 1);
    addBond(mol, 18, 19, 2);
    addBond(mol, 18, 20, 1);
    addBond(mol, 20, 21, 1);
    addBond(mol, 20, 22, 1);
    addBond(mol, 1, 23, 1);
    addBond(mol, 9, 24, 1);

    centerMolecule(mol);
}

// Build Ibuprofen (C13H18O2) - NSAID
void buildIbuprofen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Ibuprofen/Advil (C13H18O2)");

    // Benzene ring
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_C);           // C1
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);     // C2
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C); // C3
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);         // C4 (isobutyl)
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C); // C5
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C); // C6
    // Propionic acid on C1
    addAtom(mol, r * 1.8f, 0.0f, 0.0f, ATOM_C);   // CH(CH3)
    addAtom(mol, r * 2.4f, 1.0f, 0.0f, ATOM_C);   // CH3
    addAtom(mol, r * 3.0f, -0.7f, 0.0f, ATOM_C);  // COOH
    addAtom(mol, r * 3.0f, -1.9f, 0.0f, ATOM_O);  // =O
    addAtom(mol, r * 4.1f, 0.0f, 0.0f, ATOM_O);   // OH
    addAtom(mol, r * 4.8f, -0.5f, 0.0f, ATOM_H);
    // Isobutyl on C4
    addAtom(mol, r * cosf(PI) * 1.8f, r * sinf(PI) * 1.8f, 0.0f, ATOM_C);  // CH2
    addAtom(mol, r * cosf(PI) * 2.8f, r * sinf(PI) * 1.2f, 0.0f, ATOM_C);  // CH
    addAtom(mol, r * cosf(PI) * 3.5f, r * sinf(PI) * 0.5f, 0.0f, ATOM_C);  // CH3
    addAtom(mol, r * cosf(PI) * 3.5f, r * sinf(PI) * 1.9f, 0.0f, ATOM_C);  // CH3
    // H atoms on ring
    addAtom(mol, r * 1.7f * cosf(PI/3), r * 1.7f * sinf(PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(2*PI/3), r * 1.7f * sinf(2*PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(4*PI/3), r * 1.7f * sinf(4*PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(5*PI/3), r * 1.7f * sinf(5*PI/3), 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 8, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 3, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 13, 14, 1);
    addBond(mol, 13, 15, 1);
    addBond(mol, 1, 16, 1);
    addBond(mol, 2, 17, 1);
    addBond(mol, 4, 18, 1);
    addBond(mol, 5, 19, 1);

    centerMolecule(mol);
}

// Build Naproxen (C14H14O3) - NSAID
void buildNaproxen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Naproxen/Aleve (C14H14O3)");

    // Naphthalene ring system
    // Ring 1
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);
    // Ring 2 (fused)
    addAtom(mol, 3.6f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 4.8f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 4.8f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 3.6f, -2.1f, 0.0f, ATOM_C);
    // Methoxy on C6 position
    addAtom(mol, 3.6f, 2.0f, 0.0f, ATOM_O);      // O
    addAtom(mol, 3.6f, 3.3f, 0.0f, ATOM_C);      // CH3
    // Propionic acid on C2 position
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_C);     // CH(CH3)
    addAtom(mol, -1.3f, 2.0f, 0.0f, ATOM_C);     // CH3
    addAtom(mol, -2.6f, 0.0f, 0.0f, ATOM_C);     // COOH
    addAtom(mol, -2.6f, -1.2f, 0.0f, ATOM_O);    // =O
    addAtom(mol, -3.7f, 0.7f, 0.0f, ATOM_O);     // OH
    addAtom(mol, -4.4f, 0.2f, 0.0f, ATOM_H);
    // H atoms
    addAtom(mol, 1.2f, 1.7f, 0.0f, ATOM_H);
    addAtom(mol, 1.2f, -3.1f, 0.0f, ATOM_H);
    addAtom(mol, 5.7f, 0.5f, 0.0f, ATOM_H);
    addAtom(mol, 5.7f, -1.9f, 0.0f, ATOM_H);
    addAtom(mol, 3.6f, -3.1f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 2, 6, 1);
    addBond(mol, 6, 7, 2);
    addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 2);
    addBond(mol, 9, 3, 1);
    addBond(mol, 6, 10, 1);
    addBond(mol, 10, 11, 1);
    addBond(mol, 0, 12, 1);
    addBond(mol, 12, 13, 1);
    addBond(mol, 12, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 14, 16, 1);
    addBond(mol, 16, 17, 1);
    addBond(mol, 1, 18, 1);
    addBond(mol, 4, 19, 1);
    addBond(mol, 7, 20, 1);
    addBond(mol, 8, 21, 1);
    addBond(mol, 9, 22, 1);

    centerMolecule(mol);
}

// Build Diclofenac (C14H11Cl2NO2) - NSAID
void buildDiclofenac(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Diclofenac/Voltaren (C14H11Cl2NO2)");

    // Phenylacetic acid ring
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C);
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C);
    // Acetic acid on C1
    addAtom(mol, r * 1.8f, 0.0f, 0.0f, ATOM_C);   // CH2
    addAtom(mol, r * 3.0f, 0.0f, 0.0f, ATOM_C);   // COOH
    addAtom(mol, r * 3.5f, 1.0f, 0.0f, ATOM_O);   // =O
    addAtom(mol, r * 3.5f, -1.0f, 0.0f, ATOM_O);  // OH
    addAtom(mol, r * 4.3f, -1.3f, 0.0f, ATOM_H);
    // NH bridge on C2
    addAtom(mol, r * cosf(PI/3) * 1.8f, r * sinf(PI/3) * 1.8f, 0.0f, ATOM_N);
    // Dichlorophenyl ring
    addAtom(mol, r * cosf(PI/3) * 2.8f, r * sinf(PI/3) * 2.8f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI/3) * 3.3f + 0.6f, r * sinf(PI/3) * 3.3f + 1.0f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI/3) * 3.8f + 1.2f, r * sinf(PI/3) * 3.3f + 0.3f, 0.0f, ATOM_C);  // Cl
    addAtom(mol, r * cosf(PI/3) * 3.8f + 1.2f, r * sinf(PI/3) * 2.3f - 0.4f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI/3) * 3.3f + 0.6f, r * sinf(PI/3) * 1.8f - 0.4f, 0.0f, ATOM_C);
    addAtom(mol, r * cosf(PI/3) * 2.8f, r * sinf(PI/3) * 2.3f, 0.0f, ATOM_C);   // Cl
    // Chlorines
    addAtom(mol, r * cosf(PI/3) * 4.5f + 1.8f, r * sinf(PI/3) * 3.5f + 0.5f, 0.0f, ATOM_CL);
    addAtom(mol, r * cosf(PI/3) * 2.3f - 0.6f, r * sinf(PI/3) * 2.3f - 0.6f, 0.0f, ATOM_CL);
    // H on NH
    addAtom(mol, r * cosf(PI/3) * 1.8f + 0.5f, r * sinf(PI/3) * 1.8f + 0.7f, 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 7, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 1, 11, 1);
    addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 17, 12, 1);
    addBond(mol, 14, 18, 1);
    addBond(mol, 17, 19, 1);
    addBond(mol, 11, 20, 1);

    centerMolecule(mol);
}

// Build Indomethacin (C19H16ClNO4) - NSAID
void buildIndomethacin(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Indomethacin/Indocin (C19H16ClNO4)");

    // Indole ring system
    // Benzene
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);
    // Pyrrole fused
    addAtom(mol, -1.0f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, -1.0f, -0.7f, 0.0f, ATOM_C);
    addAtom(mol, -2.2f, 0.0f, 0.0f, ATOM_N);
    // Methoxy on benzene
    addAtom(mol, 3.6f, 0.7f, 0.0f, ATOM_O);
    addAtom(mol, 4.8f, 0.0f, 0.0f, ATOM_C);
    // N-p-chlorobenzoyl
    addAtom(mol, -3.4f, 0.7f, 0.0f, ATOM_C);     // C=O
    addAtom(mol, -3.4f, 2.0f, 0.0f, ATOM_O);     // =O
    // p-Chlorobenzene
    addAtom(mol, -4.6f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -5.8f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, -7.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -7.0f, -1.4f, 0.0f, ATOM_C);    // para-Cl
    addAtom(mol, -5.8f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, -4.6f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, -8.2f, -2.1f, 0.0f, ATOM_CL);   // Cl
    // Acetic acid on pyrrole C3
    addAtom(mol, -1.0f, -2.1f, 0.0f, ATOM_C);    // CH2
    addAtom(mol, -1.0f, -3.5f, 0.0f, ATOM_C);    // COOH
    addAtom(mol, 0.1f, -4.1f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -2.1f, -4.1f, 0.0f, ATOM_O);    // OH
    // Methyl on pyrrole C2
    addAtom(mol, -1.0f, 2.0f, 0.0f, ATOM_C);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 5, 7, 1);
    addBond(mol, 6, 8, 1);
    addBond(mol, 8, 7, 1);
    addBond(mol, 2, 9, 1);
    addBond(mol, 9, 10, 1);
    addBond(mol, 8, 11, 1);
    addBond(mol, 11, 12, 2);
    addBond(mol, 11, 13, 1);
    addBond(mol, 13, 14, 2);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 16, 2);
    addBond(mol, 16, 17, 1);
    addBond(mol, 17, 18, 2);
    addBond(mol, 18, 13, 1);
    addBond(mol, 16, 19, 1);
    addBond(mol, 7, 20, 1);
    addBond(mol, 20, 21, 1);
    addBond(mol, 21, 22, 2);
    addBond(mol, 21, 23, 1);
    addBond(mol, 6, 24, 1);

    centerMolecule(mol);
}

// Build Celecoxib (C17H14F3N3O2S) - COX-2 selective NSAID
void buildCelecoxib(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Celecoxib/Celebrex (C17H14F3N3O2S)");

    // Pyrazole ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_N);      // N1
    addAtom(mol, 1.0f, 0.8f, 0.0f, ATOM_N);      // N2
    addAtom(mol, 2.2f, 0.2f, 0.0f, ATOM_C);      // C3
    addAtom(mol, 2.0f, -1.2f, 0.0f, ATOM_C);     // C4
    addAtom(mol, 0.6f, -1.2f, 0.0f, ATOM_C);     // C5
    // Tolyl (4-methylphenyl) on C3
    addAtom(mol, 3.5f, 0.9f, 0.0f, ATOM_C);      // benzene
    addAtom(mol, 4.7f, 0.2f, 0.0f, ATOM_C);
    addAtom(mol, 5.9f, 0.9f, 0.0f, ATOM_C);
    addAtom(mol, 5.9f, 2.3f, 0.0f, ATOM_C);      // para-CH3
    addAtom(mol, 4.7f, 3.0f, 0.0f, ATOM_C);
    addAtom(mol, 3.5f, 2.3f, 0.0f, ATOM_C);
    addAtom(mol, 7.1f, 3.0f, 0.0f, ATOM_C);      // CH3
    // Trifluoromethylphenyl on C5
    addAtom(mol, 0.0f, -2.5f, 0.0f, ATOM_C);     // benzene
    addAtom(mol, -1.2f, -3.2f, 0.0f, ATOM_C);
    addAtom(mol, -1.2f, -4.6f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -5.3f, 0.0f, ATOM_C);     // para-CF3
    addAtom(mol, 1.2f, -4.6f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -3.2f, 0.0f, ATOM_C);
    // CF3 group
    addAtom(mol, 0.0f, -6.7f, 0.0f, ATOM_C);
    addAtom(mol, -1.1f, -7.3f, 0.0f, ATOM_F);
    addAtom(mol, 0.0f, -7.9f, 0.0f, ATOM_F);
    addAtom(mol, 1.1f, -7.3f, 0.0f, ATOM_F);
    // Sulfonamide on N1
    addAtom(mol, -1.3f, 0.7f, 0.0f, ATOM_S);     // S
    addAtom(mol, -1.3f, 2.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -2.5f, 0.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -1.3f, 0.7f, 1.5f, ATOM_N);     // NH2

    addBond(mol, 0, 1, 1);
    addBond(mol, 1, 2, 2);
    addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 2);
    addBond(mol, 4, 0, 1);
    addBond(mol, 2, 5, 1);
    addBond(mol, 5, 6, 2);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 10, 2);
    addBond(mol, 10, 5, 1);
    addBond(mol, 8, 11, 1);
    addBond(mol, 4, 12, 1);
    addBond(mol, 12, 13, 2);
    addBond(mol, 13, 14, 1);
    addBond(mol, 14, 15, 2);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 17, 12, 1);
    addBond(mol, 15, 18, 1);
    addBond(mol, 18, 19, 1);
    addBond(mol, 18, 20, 1);
    addBond(mol, 18, 21, 1);
    addBond(mol, 0, 22, 1);
    addBond(mol, 22, 23, 2);
    addBond(mol, 22, 24, 2);
    addBond(mol, 22, 25, 1);

    centerMolecule(mol);
}

// Build Meloxicam (C14H13N3O4S2) - NSAID
void buildMeloxicam(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Meloxicam/Mobic (C14H13N3O4S2)");

    // Benzothiazine core
    // Benzene ring
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, 0.7f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.4f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, 1.2f, -2.1f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, -1.4f, 0.0f, ATOM_C);
    // Thiazine fused
    addAtom(mol, -1.2f, 0.7f, 0.0f, ATOM_S);     // S
    addAtom(mol, -2.4f, 0.0f, 0.0f, ATOM_C);     // C
    addAtom(mol, -2.4f, -1.4f, 0.0f, ATOM_C);    // C (enol)
    addAtom(mol, -1.2f, -2.1f, 0.0f, ATOM_N);    // N
    // Sulfonamide SO2 on S
    addAtom(mol, -1.2f, 2.0f, 0.0f, ATOM_O);     // =O
    addAtom(mol, -0.5f, 0.7f, 1.2f, ATOM_O);     // =O
    // Amide C=O
    addAtom(mol, -3.6f, 0.7f, 0.0f, ATOM_O);     // =O on C8
    // Thiazole ring on enol C
    addAtom(mol, -3.6f, -2.1f, 0.0f, ATOM_N);
    addAtom(mol, -4.8f, -1.4f, 0.0f, ATOM_C);
    addAtom(mol, -6.0f, -2.1f, 0.0f, ATOM_S);
    addAtom(mol, -5.5f, -3.5f, 0.0f, ATOM_C);
    addAtom(mol, -4.0f, -3.5f, 0.0f, ATOM_C);
    // Methyl on thiazole
    addAtom(mol, -4.8f, 0.0f, 0.0f, ATOM_C);     // CH3
    // OH (enol)
    addAtom(mol, -2.4f, -2.6f, 0.0f, ATOM_O);
    // N-methyl
    addAtom(mol, -1.2f, -3.5f, 0.0f, ATOM_C);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 7, 8, 2);
    addBond(mol, 8, 9, 1);
    addBond(mol, 9, 5, 1);
    addBond(mol, 6, 10, 2);
    addBond(mol, 6, 11, 2);
    addBond(mol, 7, 12, 2);
    addBond(mol, 8, 13, 1);
    addBond(mol, 13, 14, 2);
    addBond(mol, 14, 15, 1);
    addBond(mol, 15, 16, 1);
    addBond(mol, 16, 17, 2);
    addBond(mol, 17, 13, 1);
    addBond(mol, 14, 18, 1);
    addBond(mol, 8, 19, 1);
    addBond(mol, 9, 20, 1);

    centerMolecule(mol);
}

// Build Acetaminophen/Paracetamol (C8H9NO2) - Tylenol
void buildAcetaminophen(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Acetaminophen/Tylenol (C8H9NO2)");

    // Benzene ring (para-substituted)
    float r = 1.4f;
    addAtom(mol, r * cosf(0), r * sinf(0), 0.0f, ATOM_C);           // C1 (OH)
    addAtom(mol, r * cosf(PI/3), r * sinf(PI/3), 0.0f, ATOM_C);     // C2
    addAtom(mol, r * cosf(2*PI/3), r * sinf(2*PI/3), 0.0f, ATOM_C); // C3
    addAtom(mol, r * cosf(PI), r * sinf(PI), 0.0f, ATOM_C);         // C4 (NHCOCH3)
    addAtom(mol, r * cosf(4*PI/3), r * sinf(4*PI/3), 0.0f, ATOM_C); // C5
    addAtom(mol, r * cosf(5*PI/3), r * sinf(5*PI/3), 0.0f, ATOM_C); // C6
    // Hydroxyl on C1 (para to amide)
    addAtom(mol, r * 1.8f * cosf(0), r * 1.8f * sinf(0), 0.0f, ATOM_O);  // OH
    addAtom(mol, r * 2.5f * cosf(0), r * 2.5f * sinf(0), 0.0f, ATOM_H);  // H
    // Acetamide on C4 (para position)
    addAtom(mol, r * 1.8f * cosf(PI), r * 1.8f * sinf(PI), 0.0f, ATOM_N);   // NH
    addAtom(mol, r * 1.8f * cosf(PI) - 0.5f, r * 1.8f * sinf(PI) + 0.8f, 0.0f, ATOM_H);  // H on N
    addAtom(mol, r * 2.8f * cosf(PI), r * 2.8f * sinf(PI), 0.0f, ATOM_C);   // C=O
    addAtom(mol, r * 2.8f * cosf(PI), r * 2.8f * sinf(PI) + 1.2f, 0.0f, ATOM_O);  // =O
    addAtom(mol, r * 4.0f * cosf(PI), r * 4.0f * sinf(PI), 0.0f, ATOM_C);   // CH3
    // H atoms on ring
    addAtom(mol, r * 1.7f * cosf(PI/3), r * 1.7f * sinf(PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(2*PI/3), r * 1.7f * sinf(2*PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(4*PI/3), r * 1.7f * sinf(4*PI/3), 0.0f, ATOM_H);
    addAtom(mol, r * 1.7f * cosf(5*PI/3), r * 1.7f * sinf(5*PI/3), 0.0f, ATOM_H);

    addBond(mol, 0, 1, 2);
    addBond(mol, 1, 2, 1);
    addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1);
    addBond(mol, 4, 5, 2);
    addBond(mol, 5, 0, 1);
    addBond(mol, 0, 6, 1);
    addBond(mol, 6, 7, 1);
    addBond(mol, 3, 8, 1);
    addBond(mol, 8, 9, 1);
    addBond(mol, 8, 10, 1);
    addBond(mol, 10, 11, 2);
    addBond(mol, 10, 12, 1);
    addBond(mol, 1, 13, 1);
    addBond(mol, 2, 14, 1);
    addBond(mol, 4, 15, 1);
    addBond(mol, 5, 16, 1);

    centerMolecule(mol);
}

// ============== STEROID HORMONES ==============

// Helper: Build the steroid core (gonane skeleton) - 4 fused rings (A, B, C, D)
// Returns indices: 0-5 = ring A, 6-9 shared with A = ring B, 10-12 shared = ring C, 13-16 = ring D
void buildSteroidCore(Molecule* mol) {
    // Ring A (cyclohexane) - carbons 0-5
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);      // C0 (C1)
    addAtom(mol, 1.4f, 0.0f, 0.5f, ATOM_C);      // C1 (C2)
    addAtom(mol, 2.5f, 0.0f, -0.4f, ATOM_C);     // C2 (C3)
    addAtom(mol, 2.5f, 1.3f, -1.0f, ATOM_C);     // C3 (C4)
    addAtom(mol, 1.4f, 2.0f, -0.5f, ATOM_C);     // C4 (C5)
    addAtom(mol, 0.0f, 1.5f, 0.0f, ATOM_C);      // C5 (C10)

    // Ring B (cyclohexane) shares C4, C5 - carbons 6-9
    addAtom(mol, -1.2f, 2.2f, 0.5f, ATOM_C);     // C6 (C6)
    addAtom(mol, -2.4f, 1.5f, 0.0f, ATOM_C);     // C7 (C7)
    addAtom(mol, -2.4f, 0.0f, 0.5f, ATOM_C);     // C8 (C8)
    addAtom(mol, -1.2f, -0.5f, 0.0f, ATOM_C);    // C9 (C9)

    // Ring C (cyclohexane) shares C7, C8 - carbons 10-13
    addAtom(mol, -3.6f, -0.5f, 0.0f, ATOM_C);    // C10 (C11)
    addAtom(mol, -4.8f, 0.2f, 0.5f, ATOM_C);     // C11 (C12)
    addAtom(mol, -4.8f, 1.5f, 0.0f, ATOM_C);     // C12 (C13)
    addAtom(mol, -3.6f, 2.2f, 0.5f, ATOM_C);     // C13 (C14)

    // Ring D (cyclopentane) shares C12, C13 - carbons 14-16
    addAtom(mol, -5.2f, 2.8f, -0.5f, ATOM_C);    // C14 (C15)
    addAtom(mol, -6.0f, 2.0f, -1.2f, ATOM_C);    // C15 (C16)
    addAtom(mol, -5.8f, 0.6f, -0.7f, ATOM_C);    // C16 (C17)

    // Angular methyl at C10 (C18)
    addAtom(mol, -1.2f, 1.5f, 1.5f, ATOM_C);     // C17 (C19 - angular methyl)

    // Angular methyl at C13 (C19)
    addAtom(mol, -3.6f, 2.0f, 2.0f, ATOM_C);     // C18 (C18 - angular methyl)

    // Ring A bonds
    addBond(mol, 0, 1, 1); addBond(mol, 1, 2, 1); addBond(mol, 2, 3, 1);
    addBond(mol, 3, 4, 1); addBond(mol, 4, 5, 1); addBond(mol, 5, 0, 1);

    // Ring B bonds (shares 5-9 edge conceptually, connects to ring A)
    addBond(mol, 5, 6, 1); addBond(mol, 6, 7, 1); addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1); addBond(mol, 9, 0, 1);

    // Ring C bonds (shares 8-13 with ring B)
    addBond(mol, 8, 10, 1); addBond(mol, 10, 11, 1); addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 1); addBond(mol, 13, 7, 1);

    // Ring D bonds (5-membered, shares 12-13 with ring C)
    addBond(mol, 12, 14, 1); addBond(mol, 14, 15, 1); addBond(mol, 15, 16, 1);
    addBond(mol, 16, 11, 1);

    // Angular methyl bonds
    addBond(mol, 9, 17, 1);  // C19 methyl at junction
    addBond(mol, 13, 18, 1); // C18 methyl at junction
}

void buildTestosterone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Testosterone (C19H28O2)");

    buildSteroidCore(mol);  // 19 carbons (0-18)

    // C3 ketone (=O on C2, index 2)
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);   // O at C3 (index 19)
    addBond(mol, 2, 19, 2);  // C=O double bond

    // C4-C5 double bond (indices 3-4)
    // Change bond 3-4 to double
    mol->bonds[3].order = 2;

    // C17 hydroxyl (on C16, index 16)
    addAtom(mol, -6.8f, 0.0f, -1.2f, ATOM_O);   // OH at C17 (index 20)
    addBond(mol, 16, 20, 1);

    // Add hydrogens to reach C19H28O2 (need 28 H)
    // Simplified - add key hydrogens
    int hStart = mol->numAtoms;
    addAtom(mol, -7.5f, 0.3f, -0.8f, ATOM_H);   // H on OH
    addBond(mol, 20, hStart, 1);

    // Add more H atoms at various positions
    float hPositions[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-0.8f, 2.2f, 1.3f},
        {-3.0f, -0.5f, -0.8f}, {-4.0f, -1.5f, 0.3f}, {-5.5f, -0.3f, 1.2f},
        {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f}, {-6.8f, 2.5f, -1.8f},
        {-1.0f, 0.8f, 2.2f}, {-1.8f, 2.2f, 1.8f}, {-0.5f, 1.8f, 1.8f},
        {-3.2f, 1.3f, 2.7f}, {-4.3f, 2.7f, 2.3f}, {-3.0f, 2.8f, 2.3f},
        {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f}, {-2.8f, 1.8f, -0.8f},
        {-5.0f, 3.3f, 0.2f}, {-4.2f, -0.8f, 0.8f}, {-6.5f, 0.3f, 0.0f},
        {3.2f, 1.8f, -1.5f}, {2.0f, 1.0f, 1.0f}, {-0.7f, -0.3f, -0.7f}
    };
    for (int i = 0; i < 27 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPositions[i][0], hPositions[i][1], hPositions[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildDHT(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "DHT (C19H30O2)");

    buildSteroidCore(mol);  // 19 carbons

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C17 hydroxyl (no C4-C5 double bond in DHT, unlike testosterone)
    addAtom(mol, -6.8f, 0.0f, -1.2f, ATOM_O);
    addBond(mol, 16, 20, 1);

    // Add hydroxyl H
    addAtom(mol, -7.5f, 0.3f, -0.8f, ATOM_H);
    addBond(mol, 20, 21, 1);

    // Add remaining hydrogens (30 total - 1 on OH = 29 more needed, simplified)
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {3.2f, 1.8f, -1.5f}, {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f},
        {-3.0f, -0.5f, -0.8f}, {-4.0f, -1.5f, 0.3f}, {-5.5f, 2.0f, 0.7f},
        {-4.8f, 3.5f, -1.0f}, {-6.8f, 2.5f, -1.8f}, {-1.0f, 0.8f, 2.2f},
        {-3.2f, 1.3f, 2.7f}, {-4.3f, 2.7f, 2.3f}, {-1.8f, -1.3f, -0.5f},
        {0.3f, 2.0f, -0.8f}, {-2.8f, 1.8f, -0.8f}, {-5.0f, 3.3f, 0.2f},
        {1.8f, 1.5f, 0.2f}, {2.8f, 0.8f, -1.2f}, {-0.7f, 0.8f, -0.7f},
        {-1.8f, 2.8f, 0.0f}, {-2.0f, 0.5f, 0.8f}, {-4.2f, 0.5f, -0.7f},
        {-5.5f, 1.0f, 1.2f}, {-6.2f, 1.5f, -0.2f}, {-5.2f, 2.2f, -1.2f},
        {0.8f, 0.5f, 0.5f}, {-0.5f, -0.5f, 0.5f}
    };
    for (int i = 0; i < 29 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildAndrostenedione(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Androstenedione (C19H26O2)");

    buildSteroidCore(mol);

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C4-C5 double bond
    mol->bonds[3].order = 2;

    // C17 ketone (instead of hydroxyl)
    addAtom(mol, -6.8f, 0.0f, -1.2f, ATOM_O);
    addBond(mol, 16, 20, 2);  // Double bond for ketone

    // Add hydrogens (26 total)
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-3.0f, -0.5f, -0.8f},
        {-4.0f, -1.5f, 0.3f}, {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f},
        {-6.8f, 2.5f, -1.8f}, {-1.0f, 0.8f, 2.2f}, {-3.2f, 1.3f, 2.7f},
        {-4.3f, 2.7f, 2.3f}, {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f},
        {-2.8f, 1.8f, -0.8f}, {-5.0f, 3.3f, 0.2f}, {-0.7f, 0.8f, -0.7f},
        {-1.8f, 2.8f, 0.0f}, {-2.0f, 0.5f, 0.8f}, {-4.2f, 0.5f, -0.7f},
        {-5.5f, 1.0f, 1.2f}, {-5.2f, 2.2f, -1.2f}, {0.8f, 0.5f, 0.5f},
        {3.2f, 1.8f, -1.5f}, {-0.5f, -0.5f, 0.5f}
    };
    for (int i = 0; i < 26 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildEstradiol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Estradiol/E2 (C18H24O2)");

    // Estrogens have aromatic A ring and no C19 methyl
    // Ring A (aromatic benzene-like) - carbons 0-5
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.1f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, -0.7f, 1.2f, 0.0f, ATOM_C);

    // Ring B - carbons 6-9
    addAtom(mol, -2.1f, 1.2f, 0.5f, ATOM_C);
    addAtom(mol, -2.8f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -2.1f, -1.2f, 0.5f, ATOM_C);
    addAtom(mol, -0.7f, -1.2f, 0.0f, ATOM_C);

    // Ring C - carbons 10-13
    addAtom(mol, -2.8f, -2.4f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, -2.4f, 0.5f, ATOM_C);
    addAtom(mol, -4.9f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, 0.0f, 0.5f, ATOM_C);

    // Ring D (5-membered) - carbons 14-16
    addAtom(mol, -5.6f, 0.0f, -0.5f, ATOM_C);
    addAtom(mol, -6.3f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -5.6f, -2.4f, -0.5f, ATOM_C);

    // C18 angular methyl (only one in estrogens)
    addAtom(mol, -4.2f, 0.0f, 2.0f, ATOM_C);

    // Ring A bonds (aromatic)
    addBond(mol, 0, 1, 2); addBond(mol, 1, 2, 1); addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1); addBond(mol, 4, 5, 2); addBond(mol, 5, 0, 1);

    // Ring B bonds
    addBond(mol, 5, 6, 1); addBond(mol, 6, 7, 1); addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1); addBond(mol, 9, 0, 1);

    // Ring C bonds
    addBond(mol, 8, 10, 1); addBond(mol, 10, 11, 1); addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 1); addBond(mol, 13, 7, 1);

    // Ring D bonds
    addBond(mol, 12, 14, 1); addBond(mol, 14, 15, 1); addBond(mol, 15, 16, 1);
    addBond(mol, 16, 11, 1);

    // Angular methyl bond
    addBond(mol, 13, 17, 1);

    // C3 hydroxyl (phenolic OH on aromatic ring)
    addAtom(mol, 1.4f, 3.6f, 0.0f, ATOM_O);
    addBond(mol, 3, 18, 1);

    // C17 hydroxyl
    addAtom(mol, -5.6f, 1.2f, -1.0f, ATOM_O);
    addBond(mol, 14, 19, 1);

    // Hydroxyl hydrogens
    addAtom(mol, 2.2f, 3.9f, 0.0f, ATOM_H);
    addBond(mol, 18, 20, 1);
    addAtom(mol, -6.3f, 1.5f, -0.5f, ATOM_H);
    addBond(mol, 19, 21, 1);

    // Add remaining hydrogens
    float hPos[][3] = {
        {2.0f, -0.9f, 0.0f}, {3.2f, 1.2f, 0.0f}, {-0.5f, 3.3f, 0.0f},
        {-2.1f, 1.2f, 1.6f}, {-2.6f, 2.1f, 0.2f}, {-2.1f, -1.2f, 1.6f},
        {-0.7f, -1.2f, -1.1f}, {-0.2f, -2.1f, 0.3f}, {-2.3f, -3.3f, 0.3f},
        {-4.2f, -2.4f, 1.6f}, {-4.7f, -3.3f, 0.2f}, {-6.0f, -0.5f, -1.3f},
        {-7.0f, -0.9f, 0.7f}, {-7.0f, -1.5f, -0.7f}, {-5.2f, -3.0f, 0.2f},
        {-6.2f, -2.9f, -1.0f}, {-3.5f, -0.5f, 2.5f}, {-5.0f, 0.5f, 2.3f},
        {-4.0f, 0.9f, 2.3f}, {-3.5f, -0.9f, -0.5f}, {-2.8f, -0.5f, -1.0f}
    };
    for (int i = 0; i < 21 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 18, idx, 1);
    }

    centerMolecule(mol);
}

void buildEstrone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Estrone/E1 (C18H22O2)");

    // Similar to estradiol but C17 is ketone
    // Ring A (aromatic)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.1f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, -0.7f, 1.2f, 0.0f, ATOM_C);

    // Rings B, C, D
    addAtom(mol, -2.1f, 1.2f, 0.5f, ATOM_C);
    addAtom(mol, -2.8f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -2.1f, -1.2f, 0.5f, ATOM_C);
    addAtom(mol, -0.7f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -2.8f, -2.4f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, -2.4f, 0.5f, ATOM_C);
    addAtom(mol, -4.9f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, 0.0f, 0.5f, ATOM_C);
    addAtom(mol, -5.6f, 0.0f, -0.5f, ATOM_C);
    addAtom(mol, -6.3f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -5.6f, -2.4f, -0.5f, ATOM_C);
    addAtom(mol, -4.2f, 0.0f, 2.0f, ATOM_C);  // Angular methyl

    // Aromatic ring A bonds
    addBond(mol, 0, 1, 2); addBond(mol, 1, 2, 1); addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1); addBond(mol, 4, 5, 2); addBond(mol, 5, 0, 1);

    // Other ring bonds
    addBond(mol, 5, 6, 1); addBond(mol, 6, 7, 1); addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1); addBond(mol, 9, 0, 1);
    addBond(mol, 8, 10, 1); addBond(mol, 10, 11, 1); addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 1); addBond(mol, 13, 7, 1);
    addBond(mol, 12, 14, 1); addBond(mol, 14, 15, 1); addBond(mol, 15, 16, 1);
    addBond(mol, 16, 11, 1);
    addBond(mol, 13, 17, 1);

    // C3 hydroxyl
    addAtom(mol, 1.4f, 3.6f, 0.0f, ATOM_O);
    addBond(mol, 3, 18, 1);

    // C17 ketone (double bond O)
    addAtom(mol, -5.6f, 1.2f, -1.0f, ATOM_O);
    addBond(mol, 14, 19, 2);

    // Hydroxyl hydrogen
    addAtom(mol, 2.2f, 3.9f, 0.0f, ATOM_H);
    addBond(mol, 18, 20, 1);

    // Add remaining hydrogens (22 - 1 on phenol = 21)
    float hPos[][3] = {
        {2.0f, -0.9f, 0.0f}, {3.2f, 1.2f, 0.0f}, {-0.5f, 3.3f, 0.0f},
        {-2.1f, 1.2f, 1.6f}, {-2.6f, 2.1f, 0.2f}, {-2.1f, -1.2f, 1.6f},
        {-0.7f, -1.2f, -1.1f}, {-0.2f, -2.1f, 0.3f}, {-2.3f, -3.3f, 0.3f},
        {-4.2f, -2.4f, 1.6f}, {-4.7f, -3.3f, 0.2f}, {-7.0f, -0.9f, 0.7f},
        {-7.0f, -1.5f, -0.7f}, {-5.2f, -3.0f, 0.2f}, {-6.2f, -2.9f, -1.0f},
        {-3.5f, -0.5f, 2.5f}, {-5.0f, 0.5f, 2.3f}, {-4.0f, 0.9f, 2.3f},
        {-3.5f, -0.9f, -0.5f}, {-2.8f, -0.5f, -1.0f}, {-6.0f, -0.5f, -1.3f}
    };
    for (int i = 0; i < 21 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 18, idx, 1);
    }

    centerMolecule(mol);
}

void buildEstriol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Estriol/E3 (C18H24O3)");

    // Same base as estradiol but with extra OH at C16
    // Ring A (aromatic)
    addAtom(mol, 0.0f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, 2.1f, 1.2f, 0.0f, ATOM_C);
    addAtom(mol, 1.4f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, 0.0f, 2.4f, 0.0f, ATOM_C);
    addAtom(mol, -0.7f, 1.2f, 0.0f, ATOM_C);

    // Rings B, C, D
    addAtom(mol, -2.1f, 1.2f, 0.5f, ATOM_C);
    addAtom(mol, -2.8f, 0.0f, 0.0f, ATOM_C);
    addAtom(mol, -2.1f, -1.2f, 0.5f, ATOM_C);
    addAtom(mol, -0.7f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -2.8f, -2.4f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, -2.4f, 0.5f, ATOM_C);
    addAtom(mol, -4.9f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -4.2f, 0.0f, 0.5f, ATOM_C);
    addAtom(mol, -5.6f, 0.0f, -0.5f, ATOM_C);
    addAtom(mol, -6.3f, -1.2f, 0.0f, ATOM_C);
    addAtom(mol, -5.6f, -2.4f, -0.5f, ATOM_C);
    addAtom(mol, -4.2f, 0.0f, 2.0f, ATOM_C);  // Angular methyl

    // Ring bonds
    addBond(mol, 0, 1, 2); addBond(mol, 1, 2, 1); addBond(mol, 2, 3, 2);
    addBond(mol, 3, 4, 1); addBond(mol, 4, 5, 2); addBond(mol, 5, 0, 1);
    addBond(mol, 5, 6, 1); addBond(mol, 6, 7, 1); addBond(mol, 7, 8, 1);
    addBond(mol, 8, 9, 1); addBond(mol, 9, 0, 1);
    addBond(mol, 8, 10, 1); addBond(mol, 10, 11, 1); addBond(mol, 11, 12, 1);
    addBond(mol, 12, 13, 1); addBond(mol, 13, 7, 1);
    addBond(mol, 12, 14, 1); addBond(mol, 14, 15, 1); addBond(mol, 15, 16, 1);
    addBond(mol, 16, 11, 1);
    addBond(mol, 13, 17, 1);

    // C3 hydroxyl (phenolic)
    addAtom(mol, 1.4f, 3.6f, 0.0f, ATOM_O);
    addBond(mol, 3, 18, 1);

    // C16 hydroxyl
    addAtom(mol, -7.0f, -1.2f, 1.0f, ATOM_O);
    addBond(mol, 15, 19, 1);

    // C17 hydroxyl
    addAtom(mol, -5.6f, 1.2f, -1.0f, ATOM_O);
    addBond(mol, 14, 20, 1);

    // Hydroxyl hydrogens
    addAtom(mol, 2.2f, 3.9f, 0.0f, ATOM_H);
    addBond(mol, 18, 21, 1);
    addAtom(mol, -7.7f, -0.8f, 0.6f, ATOM_H);
    addBond(mol, 19, 22, 1);
    addAtom(mol, -6.3f, 1.5f, -0.5f, ATOM_H);
    addBond(mol, 20, 23, 1);

    // Add remaining hydrogens
    float hPos[][3] = {
        {2.0f, -0.9f, 0.0f}, {3.2f, 1.2f, 0.0f}, {-0.5f, 3.3f, 0.0f},
        {-2.1f, 1.2f, 1.6f}, {-2.6f, 2.1f, 0.2f}, {-2.1f, -1.2f, 1.6f},
        {-0.7f, -1.2f, -1.1f}, {-0.2f, -2.1f, 0.3f}, {-2.3f, -3.3f, 0.3f},
        {-4.2f, -2.4f, 1.6f}, {-4.7f, -3.3f, 0.2f}, {-6.0f, -0.5f, -1.3f},
        {-5.2f, -3.0f, 0.2f}, {-6.2f, -2.9f, -1.0f}, {-3.5f, -0.5f, 2.5f},
        {-5.0f, 0.5f, 2.3f}, {-4.0f, 0.9f, 2.3f}, {-3.5f, -0.9f, -0.5f},
        {-2.8f, -0.5f, -1.0f}, {-6.5f, -1.8f, -0.7f}, {-5.8f, -2.8f, 0.3f}
    };
    for (int i = 0; i < 21 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 18, idx, 1);
    }

    centerMolecule(mol);
}

void buildProgesterone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Progesterone (C21H30O2)");

    buildSteroidCore(mol);  // 19 carbons (0-18)

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C4-C5 double bond
    mol->bonds[3].order = 2;

    // C17 acetyl group (-COCH3)
    addAtom(mol, -6.8f, 0.0f, -1.2f, ATOM_C);   // Carbonyl carbon (C20)
    addBond(mol, 16, 20, 1);
    addAtom(mol, -7.5f, 0.5f, -2.0f, ATOM_O);   // Carbonyl oxygen
    addBond(mol, 20, 21, 2);
    addAtom(mol, -7.5f, -1.0f, -0.5f, ATOM_C);  // Methyl (C21)
    addBond(mol, 20, 22, 1);

    // Methyl hydrogens on C21
    addAtom(mol, -8.3f, -0.5f, 0.0f, ATOM_H);
    addAtom(mol, -7.9f, -1.6f, -1.2f, ATOM_H);
    addAtom(mol, -6.9f, -1.6f, 0.1f, ATOM_H);
    addBond(mol, 22, 23, 1);
    addBond(mol, 22, 24, 1);
    addBond(mol, 22, 25, 1);

    // Add remaining hydrogens
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-3.0f, -0.5f, -0.8f},
        {-4.0f, -1.5f, 0.3f}, {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f},
        {-6.8f, 2.5f, -1.8f}, {-1.0f, 0.8f, 2.2f}, {-3.2f, 1.3f, 2.7f},
        {-4.3f, 2.7f, 2.3f}, {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f},
        {-2.8f, 1.8f, -0.8f}, {-5.0f, 3.3f, 0.2f}, {-0.7f, 0.8f, -0.7f},
        {-1.8f, 2.8f, 0.0f}, {-2.0f, 0.5f, 0.8f}, {-4.2f, 0.5f, -0.7f},
        {-5.5f, 1.0f, 1.2f}, {0.8f, 0.5f, 0.5f}, {3.2f, 1.8f, -1.5f}
    };
    for (int i = 0; i < 24 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildCortisol(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cortisol (C21H30O5)");

    buildSteroidCore(mol);  // 19 carbons (0-18)

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C4-C5 double bond
    mol->bonds[3].order = 2;

    // C11 hydroxyl
    addAtom(mol, -4.8f, -0.5f, 1.5f, ATOM_O);
    addBond(mol, 10, 20, 1);

    // C17 hydroxyl
    addAtom(mol, -5.8f, 1.0f, -1.5f, ATOM_O);
    addBond(mol, 16, 21, 1);

    // C17 side chain: -COCH2OH
    addAtom(mol, -6.8f, 0.0f, -0.5f, ATOM_C);   // Carbonyl carbon (C20)
    addBond(mol, 16, 22, 1);
    addAtom(mol, -7.5f, 0.5f, 0.3f, ATOM_O);    // Carbonyl oxygen
    addBond(mol, 22, 23, 2);
    addAtom(mol, -7.2f, -1.2f, -1.0f, ATOM_C);  // CH2 (C21)
    addBond(mol, 22, 24, 1);
    addAtom(mol, -8.3f, -1.5f, -0.3f, ATOM_O);  // Primary OH
    addBond(mol, 24, 25, 1);

    // Hydroxyl hydrogens
    addAtom(mol, -5.5f, 0.0f, 2.0f, ATOM_H);
    addBond(mol, 20, 26, 1);
    addAtom(mol, -5.3f, 1.5f, -2.0f, ATOM_H);
    addBond(mol, 21, 27, 1);
    addAtom(mol, -8.8f, -0.8f, 0.0f, ATOM_H);
    addBond(mol, 25, 28, 1);

    // CH2 hydrogens
    addAtom(mol, -6.5f, -1.8f, -1.5f, ATOM_H);
    addAtom(mol, -7.5f, -1.0f, -1.9f, ATOM_H);
    addBond(mol, 24, 29, 1);
    addBond(mol, 24, 30, 1);

    // Add remaining hydrogens
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-3.0f, -0.5f, -0.8f},
        {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f}, {-6.8f, 2.5f, -1.8f},
        {-1.0f, 0.8f, 2.2f}, {-3.2f, 1.3f, 2.7f}, {-4.3f, 2.7f, 2.3f},
        {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f}, {-2.8f, 1.8f, -0.8f},
        {-5.0f, 3.3f, 0.2f}, {0.8f, 0.5f, 0.5f}, {3.2f, 1.8f, -1.5f},
        {-4.0f, -1.5f, 0.3f}
    };
    for (int i = 0; i < 19 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildCortisone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Cortisone (C21H28O5)");

    buildSteroidCore(mol);

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C4-C5 double bond
    mol->bonds[3].order = 2;

    // C11 ketone (not hydroxyl like cortisol)
    addAtom(mol, -4.8f, -0.5f, 1.5f, ATOM_O);
    addBond(mol, 10, 20, 2);

    // C17 hydroxyl
    addAtom(mol, -5.8f, 1.0f, -1.5f, ATOM_O);
    addBond(mol, 16, 21, 1);

    // C17 side chain
    addAtom(mol, -6.8f, 0.0f, -0.5f, ATOM_C);
    addBond(mol, 16, 22, 1);
    addAtom(mol, -7.5f, 0.5f, 0.3f, ATOM_O);
    addBond(mol, 22, 23, 2);
    addAtom(mol, -7.2f, -1.2f, -1.0f, ATOM_C);
    addBond(mol, 22, 24, 1);
    addAtom(mol, -8.3f, -1.5f, -0.3f, ATOM_O);
    addBond(mol, 24, 25, 1);

    // Hydroxyl hydrogens
    addAtom(mol, -5.3f, 1.5f, -2.0f, ATOM_H);
    addBond(mol, 21, 26, 1);
    addAtom(mol, -8.8f, -0.8f, 0.0f, ATOM_H);
    addBond(mol, 25, 27, 1);

    // CH2 hydrogens
    addAtom(mol, -6.5f, -1.8f, -1.5f, ATOM_H);
    addAtom(mol, -7.5f, -1.0f, -1.9f, ATOM_H);
    addBond(mol, 24, 28, 1);
    addBond(mol, 24, 29, 1);

    // Add remaining hydrogens (28 - 2 OH H - 2 CH2 H = 24 more)
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-3.0f, -0.5f, -0.8f},
        {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f}, {-6.8f, 2.5f, -1.8f},
        {-1.0f, 0.8f, 2.2f}, {-3.2f, 1.3f, 2.7f}, {-4.3f, 2.7f, 2.3f},
        {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f}, {-2.8f, 1.8f, -0.8f},
        {-5.0f, 3.3f, 0.2f}, {0.8f, 0.5f, 0.5f}, {3.2f, 1.8f, -1.5f},
        {-4.0f, -1.5f, 0.3f}, {-2.5f, -0.8f, 0.5f}, {-0.5f, 0.5f, -0.8f},
        {-3.8f, 0.5f, 0.0f}, {-5.2f, -0.3f, -0.5f}, {-6.2f, 1.8f, 0.0f}
    };
    for (int i = 0; i < 24 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

void buildAldosterone(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Aldosterone (C21H28O5)");

    buildSteroidCore(mol);

    // C3 ketone
    addAtom(mol, 2.5f, -1.0f, -1.0f, ATOM_O);
    addBond(mol, 2, 19, 2);

    // C4-C5 double bond
    mol->bonds[3].order = 2;

    // C11 hydroxyl
    addAtom(mol, -4.8f, -0.5f, 1.5f, ATOM_O);
    addBond(mol, 10, 20, 1);

    // C18 aldehyde (CHO replaces angular methyl at C13)
    // Remove the existing methyl hydrogen assumption, add aldehyde
    addAtom(mol, -3.6f, 3.0f, 2.5f, ATOM_O);  // Aldehyde oxygen
    addBond(mol, 18, 21, 2);

    // C17 side chain
    addAtom(mol, -6.8f, 0.0f, -0.5f, ATOM_C);
    addBond(mol, 16, 22, 1);
    addAtom(mol, -7.5f, 0.5f, 0.3f, ATOM_O);
    addBond(mol, 22, 23, 2);
    addAtom(mol, -7.2f, -1.2f, -1.0f, ATOM_C);
    addBond(mol, 22, 24, 1);
    addAtom(mol, -8.3f, -1.5f, -0.3f, ATOM_O);
    addBond(mol, 24, 25, 1);

    // Hydroxyl hydrogens
    addAtom(mol, -5.5f, 0.0f, 2.0f, ATOM_H);
    addBond(mol, 20, 26, 1);
    addAtom(mol, -8.8f, -0.8f, 0.0f, ATOM_H);
    addBond(mol, 25, 27, 1);

    // Aldehyde H
    addAtom(mol, -3.0f, 3.5f, 1.8f, ATOM_H);
    addBond(mol, 18, 28, 1);

    // CH2 hydrogens
    addAtom(mol, -6.5f, -1.8f, -1.5f, ATOM_H);
    addAtom(mol, -7.5f, -1.0f, -1.9f, ATOM_H);
    addBond(mol, 24, 29, 1);
    addBond(mol, 24, 30, 1);

    // Add remaining hydrogens
    float hPos[][3] = {
        {0.0f, -0.8f, 0.7f}, {1.4f, -0.8f, 1.2f}, {3.3f, -0.5f, 0.0f},
        {1.4f, 3.0f, -0.8f}, {-1.2f, 3.2f, 0.8f}, {-3.0f, -0.5f, -0.8f},
        {-5.5f, 2.0f, 0.7f}, {-4.8f, 3.5f, -1.0f}, {-6.8f, 2.5f, -1.8f},
        {-1.0f, 0.8f, 2.2f}, {-1.8f, -1.3f, -0.5f}, {0.3f, 2.0f, -0.8f},
        {-2.8f, 1.8f, -0.8f}, {-5.0f, 3.3f, 0.2f}, {0.8f, 0.5f, 0.5f},
        {3.2f, 1.8f, -1.5f}, {-4.0f, -1.5f, 0.3f}, {-2.5f, -0.8f, 0.5f},
        {-0.5f, 0.5f, -0.8f}, {-3.8f, 0.5f, 0.0f}, {-5.2f, -0.3f, -0.5f}
    };
    for (int i = 0; i < 21 && mol->numAtoms < MAX_ATOMS; i++) {
        int idx = mol->numAtoms;
        addAtom(mol, hPos[i][0], hPos[i][1], hPos[i][2], ATOM_H);
        addBond(mol, i % 19, idx, 1);
    }

    centerMolecule(mol);
}

// Random molecule generator
float randf() { return (float)rand() / RAND_MAX; }

void buildRandomMolecule(Molecule* mol) {
    mol->numAtoms = 0;
    mol->numBonds = 0;
    strcpy(mol->name, "Random Molecule");

    // Start with a carbon backbone
    int backboneLength = 4 + rand() % 6;

    // Add backbone carbons
    for (int i = 0; i < backboneLength; i++) {
        float angle = randf() * 0.5f - 0.25f;
        float x = i * 1.5f + randf() * 0.3f;
        float y = sinf(i * 0.8f) * 0.8f + randf() * 0.3f;
        float z = cosf(i * 0.5f) * 0.5f + randf() * 0.3f;
        addAtom(mol, x, y, z, ATOM_C);
    }

    // Connect backbone
    for (int i = 0; i < backboneLength - 1; i++) {
        int order = (rand() % 4 == 0) ? 2 : 1;
        addBond(mol, i, i + 1, order);
    }

    // Maybe add a ring
    if (backboneLength >= 5 && rand() % 2 == 0) {
        addBond(mol, 0, backboneLength - 1, 1);
    }

    // Add functional groups
    for (int i = 0; i < backboneLength; i++) {
        int numH = 2 + rand() % 2;

        // Random chance of heteroatom
        if (rand() % 4 == 0) {
            int hetero = (rand() % 3 == 0) ? ATOM_N : ATOM_O;
            float angle = randf() * TWO_PI;
            float x = mol->atoms[i].x + cosf(angle) * 1.3f;
            float y = mol->atoms[i].y + sinf(angle) * 1.3f;
            float z = mol->atoms[i].z + (randf() - 0.5f) * 0.8f;
            int heteroIdx = mol->numAtoms;
            addAtom(mol, x, y, z, hetero);
            addBond(mol, i, heteroIdx, (rand() % 2 == 0) ? 2 : 1);

            // Add H to O or N
            if (rand() % 2 == 0) {
                float hx = x + (randf() - 0.5f) * 1.0f;
                float hy = y + 0.8f;
                float hz = z + (randf() - 0.5f) * 0.5f;
                int hIdx = mol->numAtoms;
                addAtom(mol, hx, hy, hz, ATOM_H);
                addBond(mol, heteroIdx, hIdx, 1);
            }
            numH--;
        }

        // Add hydrogens
        for (int h = 0; h < numH && mol->numAtoms < MAX_ATOMS - 1; h++) {
            float angle = h * PI + randf() * 0.5f;
            float hx = mol->atoms[i].x + cosf(angle) * 0.9f;
            float hy = mol->atoms[i].y + (randf() - 0.5f) * 0.8f;
            float hz = mol->atoms[i].z + sinf(angle) * 0.9f;
            int hIdx = mol->numAtoms;
            addAtom(mol, hx, hy, hz, ATOM_H);
            addBond(mol, i, hIdx, 1);
        }
    }

    centerMolecule(mol);
}

// ============== MAIN ==============

// Molecule builder function pointers
typedef void (*MoleculeBuilder)(Molecule*);

#define NUM_MOLECULES 140

MoleculeBuilder moleculeBuilders[NUM_MOLECULES] = {
    buildWater, buildMethane, buildBenzene, buildEthanol,
    buildCaffeine, buildAdenine, buildGlucose, buildAspirin,
    buildAmmonia, buildCarbonDioxide, buildFormaldehyde, buildAcetone,
    buildAceticAcid, buildPropane, buildButane, buildCyclohexane,
    buildNaphthalene, buildUrea, buildGlycine, buildAlanine,
    buildThymine, buildCytosine, buildGuanine, buildDopamine,
    buildSerotonin, buildNitricOxide, buildHydrogenPeroxide, buildSulfuricAcid,
    buildPhosphoricAcid, buildToluene, buildPhenol, buildAcetylene,
    // Batch 1 molecules
    buildOxygen, buildNitrogen, buildHydrogen, buildOzone,
    buildCarbonMonoxide, buildNitrousOxide, buildSulfurDioxide, buildHydrogenChloride,
    buildNitricAcid, buildMethanol, buildEthane, buildPropene,
    buildIsopropanol, buildEthyleneGlycol, buildGlycerol, buildAcetaldehyde,
    buildFormicAcid, buildLacticAcid, buildEthylAcetate, buildAcetonitrile,
    buildDMSO, buildDichloromethane, buildChlorobenzene, buildNitrobenzene,
    buildAniline, buildStyrene, buildBenzoicAcid, buildValine,
    buildLeucine, buildEthylene, buildHydrogenSulfide, buildChloroform,
    // Batch 2 molecules
    buildTertButanol, buildButanol, buildDiethylEther, buildMTBE,
    buildTHF, buildDioxane, buildDMF, buildCarbonTetrachloride,
    buildMethylAcetate, buildAceticAnhydride, buildPropionicAcid, buildButyricAcid,
    buildSuccinicAcid, buildBenzaldehyde, buildBromobenzene, buildPXylene,
    buildAnisole, buildPhenylacetylene, buildFructose, buildRibose,
    buildDeoxyribose, buildIsoleucine, buildSerine, buildThreonine,
    buildAsparticAcid, buildGlutamicAcid, buildLysine, buildHistidine,
    buildPhenylalanine, buildTyrosine, buildTryptophan, buildProline,
    buildCysteine, buildMethionine, buildPyruvate, buildArginine,
    // Vitamins
    buildAscorbicAcid, buildThiamine, buildRiboflavin, buildNiacin,
    buildPanthothenicAcid, buildPyridoxine, buildBiotin, buildFolicAcid,
    buildRetinol, buildBetaCarotene, buildCholecalciferol, buildAlphaTocopherol,
    buildPhylloquinone, buildNicotinamide,
    // Additional compounds
    buildCocaine, buildHeroin, buildFentanyl, buildPropofol,
    buildTHC, buildCreatine, buildOctane,
    // Statins & NSAIDs
    buildSimvastatin, buildIbuprofen, buildNaproxen, buildDiclofenac,
    buildIndomethacin, buildCelecoxib, buildMeloxicam, buildAcetaminophen,
    // Steroid hormones
    buildTestosterone, buildDHT, buildAndrostenedione, buildEstradiol,
    buildEstrone, buildEstriol, buildProgesterone, buildCortisol,
    buildCortisone, buildAldosterone,
    buildRandomMolecule
};

const char* moleculeNames[NUM_MOLECULES] = {
    "Water", "Methane", "Benzene", "Ethanol",
    "Caffeine", "Adenine", "Glucose", "Aspirin/Bayer",
    "Ammonia", "CO2", "Formaldehyde", "Acetone",
    "Acetic Acid", "Propane", "Butane", "Cyclohexane",
    "Naphthalene", "Urea", "Glycine", "Alanine",
    "Thymine", "Cytosine", "Guanine", "Dopamine",
    "Serotonin", "NO", "H2O2", "H2SO4",
    "H3PO4", "Toluene", "Phenol", "Acetylene",
    // Batch 1 molecules
    "O2", "N2", "H2", "O3",
    "CO", "N2O", "SO2", "HCl",
    "HNO3", "Methanol", "Ethane", "Propene",
    "Isopropanol", "Ethylene Glycol", "Glycerol", "Acetaldehyde",
    "Formic Acid", "Lactic Acid", "Ethyl Acetate", "Acetonitrile",
    "DMSO", "CH2Cl2", "Chlorobenzene", "Nitrobenzene",
    "Aniline", "Styrene", "Benzoic Acid", "Valine",
    "Leucine", "Ethylene", "H2S", "CHCl3",
    // Batch 2 molecules
    "tert-Butanol", "1-Butanol", "Diethyl Ether", "MTBE",
    "THF", "1,4-Dioxane", "DMF", "CCl4",
    "Methyl Acetate", "Acetic Anhydride", "Propionic Acid", "Butyric Acid",
    "Succinic Acid", "Benzaldehyde", "Bromobenzene", "p-Xylene",
    "Anisole", "Phenylacetylene", "Fructose", "Ribose",
    "Deoxyribose", "Isoleucine", "Serine", "Threonine",
    "Aspartic Acid", "Glutamic Acid", "Lysine", "Histidine",
    "Phenylalanine", "Tyrosine", "Tryptophan", "Proline",
    "Cysteine", "Methionine", "Pyruvate", "Arginine",
    // Vitamins
    "Vitamin C", "Vitamin B1", "Vitamin B2", "Vitamin B3",
    "Vitamin B5", "Vitamin B6", "Vitamin B7", "Vitamin B9",
    "Vitamin A", "Beta-Carotene", "Vitamin D3", "Vitamin E",
    "Vitamin K1", "Nicotinamide",
    // Additional compounds
    "Cocaine", "Heroin", "Fentanyl/Sublimaze", "Propofol/Diprivan",
    "THC", "Creatine", "Octane",
    // Statins & NSAIDs
    "Simvastatin/Zocor", "Ibuprofen/Advil", "Naproxen/Aleve", "Diclofenac/Voltaren",
    "Indomethacin/Indocin", "Celecoxib/Celebrex", "Meloxicam/Mobic", "Tylenol",
    // Steroid hormones
    "Testosterone", "DHT", "Androstenedione", "Estradiol/E2",
    "Estrone/E1", "Estriol/E3", "Progesterone", "Cortisol",
    "Cortisone", "Aldosterone",
    "Random"
};

int main() {
    printf("=== Windows CUDA Molecule Visualization ===\n\n");
    printf("Controls:\n");
    printf("  A/D     - Previous/Next molecule\n");
    printf("  R       - Random molecule\n");
    printf("  Arrows  - Rotate view\n");
    printf("  W/S     - Zoom in/out\n");
    printf("  Space   - Pause/resume rotation\n");
    printf("  Q/Esc   - Quit\n\n");
    printf("%d Molecules available - use A/D to cycle through\n\n", NUM_MOLECULES);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n\n", prop.name);

    Win32Display* display = win32_create_window("CUDA Molecule Visualization", WIDTH, HEIGHT);
    if (!display) {
        fprintf(stderr, "Cannot create window\n");
        return 1;
    }

    // Allocate molecule
    Molecule* h_mol = (Molecule*)malloc(sizeof(Molecule));
    Atom* d_atoms;
    Bond* d_bonds;
    cudaMalloc(&d_atoms, MAX_ATOMS * sizeof(Atom));
    cudaMalloc(&d_bonds, MAX_BONDS * sizeof(Bond));

    // Allocate display buffer
    unsigned char *h_pixels, *d_pixels;
    cudaMallocHost(&h_pixels, WIDTH * HEIGHT * 4);
    cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

    srand((unsigned int)time(NULL));

    // Current molecule index
    int currentMolecule = 0;

    // Start with water
    moleculeBuilders[currentMolecule](h_mol);
    cudaMemcpy(d_atoms, h_mol->atoms, h_mol->numAtoms * sizeof(Atom), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bonds, h_mol->bonds, h_mol->numBonds * sizeof(Bond), cudaMemcpyHostToDevice);

    printf("[%d/%d] %s (%d atoms, %d bonds)\n", currentMolecule + 1, NUM_MOLECULES, h_mol->name, h_mol->numAtoms, h_mol->numBonds);

    // View state
    float rotX = 0.3f, rotY = 0.0f;
    float zoom = 12.0f;
    int autoRotate = 1;
    int paused = 0;

    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    const double TARGET_FRAME_TIME = 1.0 / 60.0;
    double lastFpsTime = win32_get_time(display);
    int frameCount = 0;

    while (!win32_should_close(display)) {
        double frameStart = win32_get_time(display);

        win32_process_events(display);

        Win32Event event;
        while (win32_pop_event(display, &event)) {
            if (event.type == WIN32_EVENT_KEY_PRESS) {
                int key = event.key;

                if (key == XK_Escape || key == XK_q) goto cleanup;

                if (key == XK_Left) rotY -= 0.15f;
                if (key == XK_Right) rotY += 0.15f;
                if (key == XK_Up) rotX -= 0.15f;
                if (key == XK_Down) rotX += 0.15f;
                if (key == XK_w) zoom -= 1.0f;
                if (key == XK_s) zoom += 1.0f;
                zoom = fmaxf(5.0f, fminf(30.0f, zoom));

                if (key == XK_space) {
                    paused = !paused;
                }

                // Molecule cycling with A/D keys
                int newMol = 0;
                if (key == XK_a) {
                    currentMolecule = (currentMolecule - 1 + NUM_MOLECULES) % NUM_MOLECULES;
                    moleculeBuilders[currentMolecule](h_mol);
                    newMol = 1;
                }
                if (key == XK_d) {
                    currentMolecule = (currentMolecule + 1) % NUM_MOLECULES;
                    moleculeBuilders[currentMolecule](h_mol);
                    newMol = 1;
                }
                if (key == XK_r) {
                    currentMolecule = rand() % NUM_MOLECULES;
                    moleculeBuilders[currentMolecule](h_mol);
                    newMol = 1;
                }

                if (newMol) {
                    cudaMemcpy(d_atoms, h_mol->atoms, h_mol->numAtoms * sizeof(Atom), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_bonds, h_mol->bonds, h_mol->numBonds * sizeof(Bond), cudaMemcpyHostToDevice);
                    printf("[%d/%d] %s (%d atoms, %d bonds)\n", currentMolecule + 1, NUM_MOLECULES, h_mol->name, h_mol->numAtoms, h_mol->numBonds);
                }
            }

            if (event.type == WIN32_EVENT_CLOSE) goto cleanup;
        }

        // Auto-rotate
        if (autoRotate && !paused) {
            rotY += 0.01f;
        }

        // Render
        clearKernel<<<gridSize, blockSize>>>(d_pixels, WIDTH, HEIGHT);

        renderMoleculeKernel<<<gridSize, blockSize>>>(
            d_pixels, WIDTH, HEIGHT,
            d_atoms, h_mol->numAtoms,
            d_bonds, h_mol->numBonds,
            rotX, rotY, zoom);

        // Render molecule name as text overlay (72pt = 6x scale of 12px font)
        {
            static char* d_text = nullptr;
            static int d_textLen = 0;
            int textLen = (int)strlen(h_mol->name);
            if (d_text == nullptr || textLen > d_textLen) {
                if (d_text) cudaFree(d_text);
                cudaMalloc(&d_text, 128);
                d_textLen = 128;
            }
            cudaMemcpy(d_text, h_mol->name, textLen + 1, cudaMemcpyHostToDevice);
            renderTextKernel<<<gridSize, blockSize>>>(
                d_pixels, WIDTH, HEIGHT,
                d_text, textLen, 20, 20, 3);  // 3x scale = ~24pt
        }

        cudaDeviceSynchronize();

        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        win32_blit_pixels(display, h_pixels);

        frameCount++;
        double now = win32_get_time(display);
        if (now - lastFpsTime >= 1.0) {
            printf("FPS: %.1f | %s\n", frameCount / (now - lastFpsTime), h_mol->name);
            frameCount = 0;
            lastFpsTime = now;
        }

        // Frame limiting
        double frameEnd = win32_get_time(display);
        if (frameEnd - frameStart < TARGET_FRAME_TIME) {
            Sleep((DWORD)((TARGET_FRAME_TIME - (frameEnd - frameStart)) * 1000.0));
        }
    }

cleanup:
    printf("\nCleaning up...\n");

    win32_destroy_window(display);
    cudaFree(d_atoms);
    cudaFree(d_bonds);
    cudaFree(d_pixels);
    free(h_mol);
    cudaFreeHost(h_pixels);

    printf("Done!\n");
    return 0;
}
