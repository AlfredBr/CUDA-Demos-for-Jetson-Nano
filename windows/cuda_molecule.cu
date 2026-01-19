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
    strcpy(mol->name, "Aspirin (C9H8O4)");
    
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

int main() {
    printf("=== Windows CUDA Molecule Visualization ===\n\n");
    printf("Controls:\n");
    printf("  1-8     - Select molecule preset\n");
    printf("  R       - Random molecule\n");
    printf("  Arrows  - Rotate view\n");
    printf("  W/S     - Zoom in/out\n");
    printf("  A       - Toggle auto-rotate\n");
    printf("  Space   - Pause rotation\n");
    printf("  Q/Esc   - Quit\n\n");
    printf("Molecules:\n");
    printf("  1: Water    2: Methane   3: Benzene  4: Ethanol\n");
    printf("  5: Caffeine 6: Adenine   7: Glucose  8: Aspirin\n\n");
    
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
    
    // Start with caffeine
    buildCaffeine(h_mol);
    cudaMemcpy(d_atoms, h_mol->atoms, h_mol->numAtoms * sizeof(Atom), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bonds, h_mol->bonds, h_mol->numBonds * sizeof(Bond), cudaMemcpyHostToDevice);
    
    printf("Molecule: %s (%d atoms, %d bonds)\n", h_mol->name, h_mol->numAtoms, h_mol->numBonds);
    
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
                
                if (key == XK_a) {
                    autoRotate = !autoRotate;
                    printf("Auto-rotate: %s\n", autoRotate ? "ON" : "OFF");
                }
                if (key == XK_space) {
                    paused = !paused;
                }
                
                // Molecule presets
                int newMol = 0;
                if (key == XK_1) { buildWater(h_mol); newMol = 1; }
                if (key == XK_2) { buildMethane(h_mol); newMol = 1; }
                if (key == XK_3) { buildBenzene(h_mol); newMol = 1; }
                if (key == XK_4) { buildEthanol(h_mol); newMol = 1; }
                if (key == XK_5) { buildCaffeine(h_mol); newMol = 1; }
                if (key == XK_6) { buildAdenine(h_mol); newMol = 1; }
                if (key == XK_7) { buildGlucose(h_mol); newMol = 1; }
                if (key == XK_8) { buildAspirin(h_mol); newMol = 1; }
                if (key == XK_r) { buildRandomMolecule(h_mol); newMol = 1; }
                
                if (newMol) {
                    cudaMemcpy(d_atoms, h_mol->atoms, h_mol->numAtoms * sizeof(Atom), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_bonds, h_mol->bonds, h_mol->numBonds * sizeof(Bond), cudaMemcpyHostToDevice);
                    printf("Molecule: %s (%d atoms, %d bonds)\n", h_mol->name, h_mol->numAtoms, h_mol->numBonds);
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
