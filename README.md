# Jetson Nano Native CUDA Graphics Demos

A collection of **18 high-performance, hardware-accelerated graphics demonstrations** for the NVIDIA Jetson Nano. These demos bypass traditional OpenGL pipelines and use CUDA kernels directly for rendering, achieving smooth framerates on the Tegra X1 GPU.

![CUDA Graphics](https://img.shields.io/badge/CUDA-10.2-green) ![Platform](https://img.shields.io/badge/Platform-Jetson%20Nano-blue) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Why Native CUDA?

The Jetson Nano's Tegra X1 has excellent CUDA support but limited OpenGL ES capabilities when running X11 (hardware acceleration requires EGL with GBM/DRM). These demos solve that by:

- **Direct GPU Rendering**: CUDA kernels compute every pixel in parallel
- **X11 Display**: Results are copied to XImage for display (no OpenGL needed)
- **Full Hardware Acceleration**: All 128 CUDA cores working on graphics
- **No Driver Issues**: Works on any X11 display without GPU-accelerated GL

## 📋 Requirements

- NVIDIA Jetson Nano (or other Tegra device with CUDA)
- L4T (Linux for Tegra) with CUDA Toolkit (tested with CUDA 10.2)
- X11 display server
- Development packages: `libx11-dev`

## 🔨 Building

```bash
# Build all demos
make all

# Build individual demos
make cuda_life        # Conway's Game of Life
make cuda_perlin      # Perlin Noise Explorer
make cuda_cornell     # Cornell Box Path Tracer
# ... etc

# Or compile directly
nvcc -O3 -arch=sm_53 -o cuda_life cuda_life.cu -lX11

# Clean build artifacts
make clean
```

## 🎮 Demo Gallery

| Demo | Description | FPS | Complexity |
|------|-------------|-----|------------|
| [Game of Life](#1-conways-game-of-life) | Interactive cellular automaton | 60 | ⭐ |
| [Perlin Noise](#2-perlin-noise-explorer) | Procedural terrain generator | 60 | ⭐⭐ |
| [Plasma](#3-plasma-effect) | Classic demoscene effect | 60 | ⭐ |
| [Particles](#4-particle-system) | 50K particle fountain | 60 | ⭐⭐ |
| [Mandelbrot](#5-mandelbrot-explorer) | Interactive fractal zoom | Var | ⭐⭐ |
| [3D Cube](#6-3d-bouncing-ball) | Software 3D renderer | 45 | ⭐⭐⭐ |
| [Fluid Sim](#7-fluid-simulation) | Navier-Stokes solver | 35 | ⭐⭐⭐⭐ |
| [Ray Marcher](#8-ray-marcher) | SDF scene renderer | 9 | ⭐⭐⭐⭐ |
| [N-Body](#9-n-body-gravity) | 4096 gravitational bodies | 45 | ⭐⭐⭐ |
| [Primitives](#10-2d-primitives-renderer) | Anti-aliased shape library | 60 | ⭐⭐ |
| [Cornell Box](#11-cornell-box-path-tracer) | Monte Carlo path tracing | 2 | ⭐⭐⭐⭐⭐ |
| [Flame](#12-flame-effect) | Shadertoy volumetric fire | 30 | ⭐⭐⭐ |
| [Fractal](#13-kishimisu-fractal) | Animated fractal art | 60 | ⭐⭐ |
| [Tunnel](#14-volumetric-tunnel) | Low-step volumetrics | 45 | ⭐⭐⭐ |
| [Pyramid](#15-fractal-pyramid) | 3D Sierpinski pyramid | 30 | ⭐⭐⭐ |
| [Teapot](#16-utah-teapot) | Software rasterizer w/ Phong | 30 | ⭐⭐⭐⭐ |
| [Functions](#17-math-function-visualizer) | Animated function plotter | 60 | ⭐⭐ |

---

# 🎲 Demo Descriptions

---

## 1. Conway's Game of Life

**File**: `cuda_life.cu` | **Resolution**: 800×600 | **Grid**: 1024×1024 cells

### Overview

A fully interactive cellular automaton with zoom, pan, painting, and 10 rule presets. The classic B3/S23 rules and variants like HighLife, Seeds, and Day & Night are included.

### 🔑 Key Source Code Highlights

```cuda
// Ping-pong buffer pattern - essential for cellular automata
unsigned char *d_grid[2];  // Two buffers: read from one, write to other
currentGrid = 1 - currentGrid;  // Swap after each generation

// Rules stored in constant memory for fast access
__constant__ int d_birthRule[9];   // B3 = birth on exactly 3 neighbors
__constant__ int d_surviveRule[9]; // S23 = survive on 2 or 3 neighbors

// Neighbor counting with wrap-around (toroidal topology)
nx = (nx + GRID_W) % GRID_W;  // Wrap horizontally
ny = (ny + GRID_H) % GRID_H;  // Wrap vertically
```

### 📊 What to Look For

- **Gliders**: Small patterns that travel across the grid (press C to clear, then paint one)
- **Oscillators**: Patterns that return to their initial state (blinker, toad, beacon)
- **Still lifes**: Stable patterns (block, beehive, loaf)
- **Chaos vs Order**: Random starts often stabilize into recognizable structures
- **Rule differences**: Try HighLife (1) for replicators, Seeds (2) for explosive growth

### 🎮 Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `Space` | Pause/Play | `S` | Single step |
| `R` | Randomize | `C` | Clear grid |
| `+/-` | Zoom in/out | `Arrows` | Pan view |
| `1-5` | Brush size | `[/]` | Speed ±1 |
| `W` | Toggle wrap | `H` | Toggle UI |
| `0-9` | Rule presets | `Q` | Quit |
| `Left Click` | Paint alive | `Right Click` | Paint dead |

### 🧬 Rule Presets

| Key | Rule | Name | Behavior |
|-----|------|------|----------|
| 0 | B3/S23 | Conway | Classic Life |
| 1 | B36/S23 | HighLife | Has replicators! |
| 2 | B2/S | Seeds | Explosive, chaotic |
| 3 | B3/S012345678 | Life w/o Death | Only growth |
| 4 | B1357/S1357 | Replicator | Everything copies |
| 5 | B368/S245 | Morley | Move-like patterns |
| 6 | B3678/S34678 | Day & Night | Symmetric rules |
| 7 | B35678/S5678 | Diamoeba | Large blob-like |
| 8 | B4678/S35678 | Anneal | Blob formation |
| 9 | B3/S2345 | Long Life | Slow decay |

---

## 2. Perlin Noise Explorer

**File**: `cuda_perlin.cu` | **Resolution**: 640×480 | **Interactive UI**

### Overview

Real-time procedural noise generation with three noise modes (fBm, domain warp, ridged), four color palettes, and full parameter control. This is what terrain generators, clouds, and procedural textures are built from.

### 🔑 Key Source Code Highlights

```cuda
// Ken Perlin's improved noise - the "fade" function for smooth interpolation
__device__ float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);  // 6t⁵ - 15t⁴ + 10t³
}

// Permutation table in constant memory (fast random-access lookup)
__constant__ int perm[512];  // Doubled to avoid modulo operations

// fBm (fractal Brownian motion) - layered noise octaves
for (int i = 0; i < octaves; i++) {
    total += perlin3D(x * freq, y * freq, z) * amp;
    freq *= lacunarity;   // Each octave: higher frequency
    amp *= persistence;   // Each octave: lower amplitude
}

// Domain warping - use noise to distort the input coordinates
float warpX = fbm(x + 5.2f, y + 1.3f, z, ...);
float warpY = fbm(x + 1.7f, y + 9.2f, z, ...);
return fbm(x + warpX * warpStrength, y + warpY * warpStrength, z, ...);
```

### 📊 What to Look For

- **fBm**: Natural-looking terrain with self-similar detail at all scales
- **Domain Warp**: Swirling, organic distortions like marble or clouds
- **Ridged**: Sharp mountain ridges (uses `1 - |noise|` instead of noise)
- **Octaves effect**: More octaves = more fine detail (but slower)
- **Lacunarity**: Controls frequency jump between octaves (2.0 = standard)
- **Persistence**: Controls amplitude decay (0.5 = each octave half as strong)

### 🎮 Controls

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `F/G` | Frequency ±0.1 | `L/K` | Lacunarity ±0.1 |
| `P/O` | Persistence ±0.05 | `W/S` | Warp strength ±0.1 |
| `T/R` | Time speed ±0.1 | `+/-` | Octaves ±1 |
| `C` | Cycle color mode | `N` | Cycle noise mode |
| `Z` | Randomize seed | `Space` | Pause animation |
| `H` | Toggle help | `Q` | Quit |

### 🎨 Color Modes

- **Grayscale**: Raw noise visualization
- **Terrain**: Water → Sand → Grass → Rock → Snow (heightmap)
- **Marble**: Blue-white veined stone pattern
- **Plasma**: Vibrant rainbow cycling

---

## 3. Plasma Effect

**File**: `cuda_render.cu` | **Resolution**: 800×600 | **~60 FPS**

### Overview

A classic demoscene plasma effect - the simplest demo and perfect introduction to CUDA graphics. Each pixel's color comes from overlapping sine waves.

### 🔑 Key Source Code Highlights

```cuda
// The core plasma formula - pure math, perfectly parallel
float dx = (x - WIDTH/2) / 100.0f;
float dy = (y - HEIGHT/2) / 100.0f;
float dist = sqrtf(dx*dx + dy*dy);

float v = sinf(x/20.0f + t)
        + sinf(y/15.0f - t*0.7f)
        + sinf((x+y)/25.0f + t*0.5f)
        + sinf(dist + t);

// Map to RGB with phase offsets for color variation
r = (unsigned char)(128 + 127 * sinf(v * PI));
g = (unsigned char)(128 + 127 * sinf(v * PI + 2.094f));  // +120°
b = (unsigned char)(128 + 127 * sinf(v * PI + 4.188f));  // +240°
```

### 📊 What to Look For

- **Wave interference**: Where sine waves align, colors intensify
- **Radial component**: `dist` creates circular patterns from center
- **Time evolution**: Each term has different time multipliers for complex motion
- **Color cycling**: The phase offsets (2π/3 apart) ensure full spectrum coverage

---

## 4. Particle System

**File**: `cuda_particles.cu` | **Particles**: 50,000 | **~60 FPS**

### Overview

Real-time physics simulation with 50K particles featuring gravity, bouncing, and fading trails.

### 🔑 Key Source Code Highlights

```cuda
// Structure of Arrays (SoA) for coalesced memory access
float *d_posX, *d_posY;    // Positions
float *d_velX, *d_velY;    // Velocities
float *d_life;             // Remaining lifetime

// Physics update - each particle independent (embarrassingly parallel)
velY[i] += gravity * dt;           // Apply gravity
posX[i] += velX[i] * dt;           // Integrate position
posY[i] += velY[i] * dt;

// Bounce off walls with energy loss
if (posY[i] > floorY) {
    posY[i] = floorY;
    velY[i] *= -0.7f;  // Coefficient of restitution
}
```

### 📊 What to Look For

- **Fountain pattern**: Particles spawn upward, arc under gravity
- **Trail persistence**: Fading trails show motion history
- **Bounce behavior**: Watch for realistic energy loss at floor
- **Color variation**: Each particle has unique hue for visual richness

---

## 5. Mandelbrot Explorer

**File**: `cuda_mandelbrot.cu` | **Resolution**: 800×600 | **Interactive Zoom**

### Overview

Navigate the infinite complexity of the Mandelbrot set with real-time GPU rendering.

### 🔑 Key Source Code Highlights

```cuda
// The Mandelbrot iteration: z = z² + c
double zr = 0, zi = 0;
for (int i = 0; i < maxIter; i++) {
    double zr2 = zr * zr;
    double zi2 = zi * zi;
    if (zr2 + zi2 > 4.0) {  // Escape radius squared
        // Smooth coloring using continuous iteration count
        float smooth = i + 1 - log2f(log2f((float)(zr2 + zi2)));
        return smooth;
    }
    zi = 2 * zr * zi + ci;      // Imaginary part of z²
    zr = zr2 - zi2 + cr;        // Real part of z²
}
return maxIter;  // Point is in the set
```

### 📊 What to Look For

- **The main cardioid**: Heart-shaped central body
- **Period-2 bulb**: Large circle attached to the left
- **Seahorse Valley**: Zoom into the neck between cardioid and bulb
- **Mini-Mandelbrots**: Self-similar copies appear at all scales!
- **Iteration bands**: Color bands show "escape speed"

### 🎮 Controls

| Key | Action |
|-----|--------|
| `Left Click` | Zoom in at cursor |
| `Right Click` | Zoom out |
| `Arrows` | Pan |
| `+/-` | Adjust max iterations |
| `R` | Reset view |

---

## 6. 3D Bouncing Ball

**File**: `cuda_3d_cube.cu` | **Resolution**: 800×600 | **~45 FPS**

### Overview

A complete software 3D renderer with a physics-enabled ball bouncing inside a wireframe cube. No OpenGL!

### 🔑 Key Source Code Highlights

```cuda
// Perspective projection matrix application
float scale = fov / (fov + z);  // Perspective divide
screenX = x * scale + WIDTH/2;
screenY = y * scale + HEIGHT/2;

// Sphere rendering using signed distance field
float dist = length(pixelPos - sphereCenter) - radius;
if (dist < 0) {
    // Inside sphere - compute lighting
    vec3 normal = normalize(pixelPos - sphereCenter);
    float diffuse = max(0, dot(normal, lightDir));
}

// Physics: velocity reflection on collision
if (pos.x > boundary) {
    pos.x = boundary;
    vel.x *= -0.9f;  // Bounce with energy loss
}
```

### 📊 What to Look For

- **Perspective depth**: Objects shrink with distance
- **Shadow projection**: Ball casts shadow on cube floor
- **Physics accuracy**: Ball loses energy on each bounce
- **Wireframe rendering**: Lines are depth-sorted for proper overlap

---

## 7. Fluid Simulation

**File**: `cuda_fluid.cu` | **Grid**: 256×256 | **~35 FPS**

### Overview

Implementation of Jos Stam's "Stable Fluids" (SIGGRAPH 1999) - the algorithm behind movie fluid effects.

### 🔑 Key Source Code Highlights

```cuda
// Semi-Lagrangian advection - trace particle backwards in time
float srcX = x - dt * velX[idx];
float srcY = y - dt * velY[idx];
// Bilinear interpolation at source location
newDensity[idx] = bilerp(density, srcX, srcY);

// Pressure projection - make velocity divergence-free (incompressible)
// Jacobi iteration solving: ∇²p = ∇·v
for (int iter = 0; iter < 20; iter++) {
    p[idx] = (p[left] + p[right] + p[up] + p[down] - div[idx]) / 4.0f;
}
// Subtract pressure gradient from velocity
velX[idx] -= 0.5f * (p[right] - p[left]);
velY[idx] -= 0.5f * (p[up] - p[down]);
```

### 📊 What to Look For

- **Vortex formation**: Drag creates spinning eddies
- **Advection**: Dye follows the velocity field
- **Incompressibility**: Fluid doesn't compress, it flows around
- **Diffusion**: Colors slowly spread and mix
- **Viscosity effect**: Higher viscosity = thicker, slower fluid

### 🎮 Controls

| Key | Action |
|-----|--------|
| `Left Drag` | Add dye + velocity |
| `1-4` | Color schemes |
| `V` | Show velocity field |
| `C` | Clear |
| `+/-` | Viscosity |
| `[/]` | Diffusion |

---

## 8. Ray Marcher

**File**: `cuda_raymarcher.cu` | **Resolution**: 800×600 | **~9 FPS**

### Overview

Render 3D scenes using Signed Distance Fields (SDFs) - the technique behind many Shadertoy creations.

### 🔑 Key Source Code Highlights

```cuda
// SDF primitives - return distance to surface
__device__ float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

__device__ float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0)) + min(max(d.x, max(d.y, d.z)), 0);
}

// Ray marching loop
float t = 0;
for (int i = 0; i < MAX_STEPS; i++) {
    vec3 p = ro + rd * t;
    float d = sceneSDF(p);      // Distance to nearest surface
    if (d < EPSILON) break;     // Hit!
    t += d;                     // Safe to step this far
    if (t > MAX_DIST) break;    // Miss
}

// Soft shadows - march toward light, accumulate occlusion
float shadow = 1.0;
for (float t = 0.1; t < maxDist; ) {
    float d = sceneSDF(p + lightDir * t);
    shadow = min(shadow, k * d / t);  // k controls softness
    t += d;
}
```

### 📊 What to Look For

- **Smooth surfaces**: SDFs naturally produce smooth geometry
- **CSG operations**: Boolean combinations of shapes
- **Soft shadows**: Penumbra from partial occlusion
- **Ambient occlusion**: Corners appear darker

### 🎮 Controls

| Key | Action |
|-----|--------|
| `1-4` | Switch scenes |
| `Arrows` | Orbit camera |
| `W/S` | Dolly in/out |
| `P` | Toggle shadows |
| `O` | Toggle AO |
| `Space` | Pause animation |

---

## 9. N-Body Gravity

**File**: `cuda_nbody.cu` | **Bodies**: 4,096 | **~45 FPS**

### Overview

Gravitational simulation using tiled shared-memory optimization for the O(N²) force calculation.

### 🔑 Key Source Code Highlights

```cuda
// Tiled force computation - the key optimization
__shared__ float4 sharedPos[TILE_SIZE];  // 128 bodies per tile

for (int tile = 0; tile < numTiles; tile++) {
    // Cooperatively load tile into shared memory
    sharedPos[threadIdx.x] = positions[tile * TILE_SIZE + threadIdx.x];
    __syncthreads();

    // Each thread computes force from all bodies in tile
    for (int j = 0; j < TILE_SIZE; j++) {
        float3 r = sharedPos[j].xyz - myPos.xyz;
        float distSq = dot(r, r) + SOFTENING * SOFTENING;
        float invDist3 = rsqrtf(distSq * distSq * distSq);
        acc += r * sharedPos[j].w * invDist3;  // F = Gm/r²
    }
    __syncthreads();
}

// Leapfrog integration (symplectic - conserves energy better)
vel += acc * dt * 0.5f;
pos += vel * dt;
vel += acc * dt * 0.5f;
```

### 📊 What to Look For

- **Gravitational collapse**: Random sphere contracts to dense core
- **Spiral arms**: Disk galaxies form characteristic structure
- **Galaxy collision**: Two disks merge into elliptical galaxy
- **Orbital mechanics**: Bodies with enough tangential velocity orbit
- **Three-body chaos**: Even 3 bodies exhibit chaotic behavior

### 🎮 Controls

| Key | Action |
|-----|--------|
| `1-5` | Galaxy presets |
| `T` | Toggle trails |
| `Space` | Pause |
| `+/-` | Time step |

---

## 10. 2D Primitives Renderer

**File**: `cuda_primitives.cu` | **Resolution**: 800×600 | **~60 FPS**

### Overview

A GPU-accelerated 2D shape library with anti-aliased rendering for lines, circles, rectangles, triangles, and Bézier curves.

### 🔑 Key Source Code Highlights

```cuda
// Anti-aliased circle using signed distance
__device__ float sdCircle(float2 p, float2 center, float radius) {
    return length(p - center) - radius;
}

// Convert distance to alpha for smooth edges
float d = sdCircle(pixel, circleCenter, radius);
float alpha = 1.0f - smoothstep(-1.0f, 1.0f, d);

// Quadratic Bézier curve - parametric form
__device__ float2 bezier(float2 p0, float2 p1, float2 p2, float t) {
    float u = 1.0f - t;
    return u*u*p0 + 2*u*t*p1 + t*t*p2;
}

// Line with round caps and anti-aliasing
__device__ float sdLine(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0f, 1.0f);
    return length(pa - ba * h);
}
```

### 📊 What to Look For

- **Smooth edges**: Anti-aliasing prevents jagged lines
- **Filled vs stroked**: Same SDF, different thresholding
- **Curve smoothness**: Bézier curves are perfectly smooth at any scale
- **Alpha blending**: Semi-transparent overlapping shapes

---

## 11. Cornell Box Path Tracer

**File**: `cuda_cornell.cu` | **Resolution**: 512×512 | **~2 FPS** (progressive)

### Overview

Physically-based Monte Carlo path tracing producing photorealistic global illumination. This is how movie CGI lighting works!

### 🔑 Key Source Code Highlights

```cuda
// Russian Roulette for unbiased path termination
float p = max(throughput.r, max(throughput.g, throughput.b));
if (curand_uniform(&rng) > p) break;
throughput /= p;  // Compensate for terminated paths

// Cosine-weighted hemisphere sampling (importance sampling)
float r1 = curand_uniform(&rng);
float r2 = curand_uniform(&rng);
float phi = 2 * PI * r1;
float cosTheta = sqrtf(r2);  // Not uniform - weighted by cos
float sinTheta = sqrtf(1 - r2);
vec3 sample = vec3(cos(phi)*sinTheta, sin(phi)*sinTheta, cosTheta);

// Transform to world space using normal
vec3 newDir = tangentToWorld(sample, normal);

// Progressive accumulation
frameBuffer[idx] = (frameBuffer[idx] * frameCount + newSample) / (frameCount + 1);
```

### 📊 What to Look For

- **Color bleeding**: Red/green walls tint nearby surfaces
- **Soft shadows**: Area light produces penumbra
- **Caustics**: Light focused by glass sphere (if present)
- **Convergence**: Image starts noisy, gradually clears
- **Indirect illumination**: Ceiling lit only by bounced light

### 🎮 Controls

| Key | Action |
|-----|--------|
| `Space` | Pause/resume |
| `R` | Reset accumulation |
| `+/-` | Samples per pixel |

---

## 12. Flame Effect

**File**: `cuda_flame.cu` | **Resolution**: 640×480 | **~30 FPS**

### Overview

Port of XT95's Shadertoy flame shader - volumetric fire using ray marching with noise-based density.

### 🔑 Key Source Code Highlights

```cuda
// Las^Mercury noise function - the heart of the flame
__device__ float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);  // Smoothstep
    // 3D noise lookup with trilinear interpolation
    return mix(mix(mix(hash(i), hash(i+vec3(1,0,0)), f.x), ...));
}

// Flame density function
__device__ float flameDensity(vec3 p, float time) {
    float d = sdSphere(p, 0.5f);  // Base sphere shape
    d += noise(p * 4.0f + time) * 0.3f;  // Noise displacement
    d += noise(p * 8.0f - time * 2.0f) * 0.15f;  // Finer detail
    return d;
}

// Volumetric accumulation
for (int i = 0; i < 64; i++) {
    float density = flameDensity(pos, time);
    if (density < 0) {
        // Inside flame - accumulate glow
        glow += exp(-density * 10.0f) * stepSize;
        color += flameColor(density) * glow;
    }
    pos += rayDir * stepSize;
}
```

### 📊 What to Look For

- **Flame shape**: Sphere distorted by layered noise
- **Turbulence**: Multiple noise octaves at different speeds
- **Glow accumulation**: Volumetric scattering simulation
- **Color gradient**: Hot white core → orange → red → black

---

## 13. Kishimisu Fractal

**File**: `cuda_fractal.cu` | **Resolution**: 640×480 | **~60 FPS**

### Overview

Port of Kishimisu's mesmerizing fractal shader with IQ cosine color palettes.

### 🔑 Key Source Code Highlights

```cuda
// IQ's cosine palette - compact, beautiful color generation
__device__ vec3 palette(float t) {
    vec3 a = vec3(0.5f, 0.5f, 0.5f);      // Brightness
    vec3 b = vec3(0.5f, 0.5f, 0.5f);      // Contrast
    vec3 c = vec3(1.0f, 1.0f, 1.0f);      // Oscillation
    vec3 d = vec3(0.263f, 0.416f, 0.557f); // Phase
    return a + b * cos(6.28318f * (c * t + d));
}

// Fractal iteration with domain repetition
for (int i = 0; i < 4; i++) {
    uv = fract(uv * 1.5f) - 0.5f;  // Tile and center
    float d = length(uv) * exp(-length(uv0));  // Distance with falloff
    color += palette(length(uv0) + time * 0.4f + i * 0.4f) / d;
}
```

### 📊 What to Look For

- **Self-similarity**: Same pattern at multiple scales
- **Domain repetition**: `fract()` creates infinite tiling
- **Color evolution**: Palette shifts with distance and time
- **Exponential falloff**: Creates depth illusion

---

## 14. Volumetric Tunnel

**File**: `cuda_tunnel.cu` | **Resolution**: 640×480 | **~45 FPS**

### Overview

Port of Frostbyte's Shadertoy - impressive volumetrics in only 10 ray march steps!

### 🔑 Key Source Code Highlights

```cuda
// Xor's dot noise - cheap volumetric density
__device__ float dotNoise(vec3 p) {
    const mat3 GOLD = mat3(...);  // Golden ratio rotation
    return dot(cos(GOLD * p), sin(PHI * p * GOLD));
}

// The magic: only 10 steps needed!
for (int i = 0; i < 10; i++) {
    b = p;
    b.xy = rotate(sin(b.xy), t * 1.5f + b.z * 3.0f);

    s = 0.001f + fabsf(dotNoise(b * 12.0f) / 12.0f - dotNoise(b)) * 0.4f;
    s = fmaxf(s, 2.0f - length(p.xy));  // Clear tunnel for camera
    s += fabsf(p.y * 0.75f + sinf(p.z + t * 0.1f + p.x * 1.5f)) * 0.2f;

    p += d * s;  // Accumulate along ray
    color += (1.0f + sin(i + length(p.xy * 0.1f) + vec3(3,1.5,1))) / s;
}
```

### 📊 What to Look For

- **Low step count**: Only 10 iterations yet rich volumetrics
- **Turbulence**: Sine distortion creates swirling effect
- **Depth perception**: Movement along Z creates depth
- **ACES tonemapping**: High dynamic range compressed beautifully

---

## 15. Fractal Pyramid

**File**: `cuda_pyramid.cu` | **Resolution**: 640×480 | **~30 FPS**

### Overview

3D ray-marched Sierpinski tetrahedron (fractal pyramid) with rotation and glow effects.

### 🔑 Key Source Code Highlights

```cuda
// Sierpinski tetrahedron iteration - folding space
__device__ float sierpinski(vec3 p, int iterations) {
    float scale = 2.0f;
    vec3 offset = vec3(1, 1, 1);

    for (int i = 0; i < iterations; i++) {
        // Fold along planes to create self-similarity
        if (p.x + p.y < 0) p.xy = -p.yx;
        if (p.x + p.z < 0) p.xz = -p.zx;
        if (p.y + p.z < 0) p.yz = -p.zy;

        p = p * scale - offset * (scale - 1.0f);
    }
    return length(p) * powf(scale, -iterations);
}
```

### 📊 What to Look For

- **Self-similarity**: Zoom reveals identical structure at all scales
- **Folding symmetry**: Same shape from any angle
- **Infinite detail**: More iterations = finer triangular structure

---

## 16. Utah Teapot

**File**: `cuda_teapot.cu` | **Resolution**: 800×600 | **~30 FPS**

### Overview

100% CUDA software rasterizer rendering the iconic Utah Teapot with Phong shading. No OpenGL whatsoever - implements the entire 3D graphics pipeline in CUDA:
- OBJ mesh loading with automatic normal computation
- Model/View/Projection matrix transforms
- Triangle rasterization with edge functions
- Depth buffer (Z-buffer) with atomic operations
- Perspective-correct interpolation
- Blinn-Phong lighting with orbiting light source

### 🔑 Key Source Code Highlights

```cuda
// Edge function for triangle rasterization
__device__ float edgeFunction(float ax, float ay, float bx, float by, float cx, float cy) {
    return (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);
}

// Atomic depth test using CAS (Compare-And-Swap)
unsigned int assumed, old;
do {
    assumed = __float_as_uint(oldDepth);
    old = atomicCAS((unsigned int*)&depth[pixelIdx], assumed, __float_as_uint(z));
    oldDepth = __uint_as_float(old);
} while (oldDepth > z && old != assumed);

// Blinn-Phong specular highlights
vec3 H = normalize(L + V);  // Half-vector
float spec = powf(fmaxf(dot(normal, H), 0.0f), shininess);
```

### 📊 What to Look For

- **Smooth shading**: Vertex normals are averaged from face normals for Gouraud-style interpolation
- **Specular highlights**: Bright spots that move as the light orbits
- **Silhouette**: Back-face culling shows clean edges
- **Copper material**: Warm orange-brown diffuse with bright specular

### 🎮 Controls

| Key | Action |
|-----|--------|
| ←/→ | Rotate teapot |
| ↑/↓ | Adjust light height |
| W/S | Zoom in/out |
| Space | Toggle auto-rotate |
| Q/ESC | Quit |

---

## 17. Math Function Visualizer

**File**: `cuda_functions.cu` | **Resolution**: 800×600 | **~60 FPS**

### Overview

A port of Shadertoy #2244 - an animated mathematical function plotter displaying many functions simultaneously on an interactive coordinate grid. Uses IQ's hash functions for temporal anti-aliasing.

### 🔑 Key Source Code Highlights

```cuda
// IQ's integer hash for quality randomness
__device__ unsigned int hash(unsigned int n) { return n*(n^(n>>15)); }

// Animated sigmoid with time-varying steepness
__device__ float f3(float x, float t) {
    return 1.0f/(1.0f+exp2f(-x*16.0f*sinf(t*PI2/4.0f)));
}

// Power function with animated exponent
__device__ float fpow(float x, float t) {
    return powf(fabsf(x)+0.001f, exp2f(4.0f*sinf(t*PI2/5.0f)));
}

// Distance to line for anti-aliased function plotting
__device__ float dist_line(vec2 p, vec2 a, vec2 n) {
    p = p - a;
    return length(p - dot(p,n)*n);
}
```

### 📊 What to Look For

- **Multiple grids**: 3 levels of grid detail (10, 1, 0.1 units)
- **Color coding**: Different functions have different colors
- **Animation**: Sigmoid, power, and sine functions morph over time
- **Anti-aliasing**: Smooth lines using distance-based falloff

### 🎮 Controls

| Key | Action |
|-----|--------|
| ←/→/↑/↓ | Pan view |
| +/= | Zoom in |
| - | Zoom out |
| R | Reset view |
| Q/ESC | Quit |

---

## ⚡ Performance Summary

| Demo | FPS | GPU Load | Memory | Key Bottleneck |
|------|-----|----------|--------|----------------|
| Game of Life | 60 | Low | 2 MB | Memory bandwidth |
| Perlin Noise | 60 | Medium | 1.2 MB | Computation |
| Plasma | 60 | Low | 1.9 MB | Memory bandwidth |
| Particles | 60 | Medium | 2.4 MB | Physics update |
| Mandelbrot | Var | High | 1.9 MB | Iteration depth |
| 3D Cube | 45 | Medium | 1.9 MB | Distance field |
| Fluid | 35 | High | 3.2 MB | Multiple passes |
| Ray Marcher | 9 | Very High | 1.9 MB | Per-pixel raycast |
| N-Body | 45 | High | 0.5 MB | O(N²) forces |
| Primitives | 60 | Low | 1.9 MB | Simple SDFs |
| Cornell Box | 2 | Extreme | 1 MB | Path tracing |
| Flame | 30 | High | 1.2 MB | Volumetric |
| Fractal | 60 | Medium | 1.2 MB | Simple iteration |
| Tunnel | 45 | Medium | 1.2 MB | Low step count |
| Pyramid | 30 | High | 1.2 MB | Fractal SDF |
| Teapot | 30 | High | 2 MB | Rasterization |
| Functions | 60 | Low | 1.9 MB | Simple math |

---

## 🏗️ Architecture Pattern

All demos follow the same structure:

```cuda
// 1. GPU Memory Allocation
unsigned char* d_pixels;
cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4);

// 2. Main Loop
while (running) {
    // Handle X11 events
    while (XPending(display)) { ... }

    // Launch render kernel
    renderKernel<<<gridDim, blockDim>>>(d_pixels, time, params);

    // Copy to CPU and display
    cudaMemcpy(image->data, d_pixels, size, cudaMemcpyDeviceToHost);
    XPutImage(display, window, gc, image, ...);
}

// 3. Cleanup
cudaFree(d_pixels);
```

---

## 📚 Educational Value

Each demo teaches specific GPU programming concepts:

| Concept | Best Demo to Study |
|---------|-------------------|
| Basic CUDA kernel | Plasma |
| Particle systems | Particles |
| Stencil operations | Game of Life |
| Procedural generation | Perlin Noise |
| Fluid dynamics | Fluid Sim |
| Ray marching | Ray Marcher |
| Path tracing | Cornell Box |
| Shared memory tiling | N-Body |
| SDF rendering | Primitives |
| Ping-pong buffers | Game of Life, Fluid |
| Software rasterization | Teapot |
| Function plotting | Functions |

---

## 📜 License

MIT License - Feel free to use, modify, and distribute.

## 🙏 Credits

- Fluid simulation: Jos Stam's "Stable Fluids" (SIGGRAPH 1999)
- Ray marching techniques: Inigo Quilez (iq)
- N-body optimization: NVIDIA CUDA samples
- Flame shader: XT95 (anatole duprat) - Shadertoy
- Fractal shader: Kishimisu - Shadertoy
- Tunnel shader: Frostbyte - Shadertoy
- Perlin noise: Ken Perlin's improved noise (2002)
- Function visualizer: Shadertoy #2244
- Utah Teapot: Martin Newell (1975)

---

*Built for Jetson Nano - Demonstrating the power of CUDA for real-time graphics* 🚀

# CUDA Graphics Demos for Jetson Nano by Alfred Broderick and Claude Opus 4.5
