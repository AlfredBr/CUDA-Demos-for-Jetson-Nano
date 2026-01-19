/*
 * Jetson Nano CUDA Parallel Primitives Toybox
 * 
 * The foundational building blocks of GPU algorithms:
 *   - Reduce (sum/min/max)
 *   - Scan (exclusive/inclusive prefix sum)
 *   - Stream Compaction (filter with predicate)
 *   - Histogram (counting/binning)
 * 
 * Three practical demonstrations:
 *   Demo A: Particle speed statistics (reduce)
 *   Demo B: GPU prime sieve (compaction)
 *   Demo C: Radix sort skeleton (histogram + scan)
 * 
 * "If you can write a correct, fast scan, you've earned your GPU stripes."
 * 
 * Controls:
 *   1       - Demo A: Particle Statistics
 *   2       - Demo B: Prime Sieve
 *   3       - Demo C: Radix Sort Demo
 *   Space   - Run current demo
 *   +/-     - Adjust problem size
 *   V       - Toggle verbose output
 *   Q/Esc   - Quit
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>

#define WIDTH 1000
#define HEIGHT 700

// Block sizes for different kernels
#define REDUCE_BLOCK_SIZE 256
#define SCAN_BLOCK_SIZE 256
#define HISTOGRAM_BLOCK_SIZE 256

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Check CUDA errors
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// REDUCTION KERNELS
// ============================================================================

// Tree reduction in shared memory - sum
__global__ void reduceSumKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    // Load and do first reduction during load
    float sum = 0.0f;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    
    // Tree reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed within warp)
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Tree reduction - min
__global__ void reduceMinKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float minVal = FLT_MAX;
    if (i < n) minVal = input[i];
    if (i + blockDim.x < n) minVal = fminf(minVal, input[i + blockDim.x]);
    sdata[tid] = minVal;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] = fminf(smem[tid], smem[tid + 32]);
        smem[tid] = fminf(smem[tid], smem[tid + 16]);
        smem[tid] = fminf(smem[tid], smem[tid + 8]);
        smem[tid] = fminf(smem[tid], smem[tid + 4]);
        smem[tid] = fminf(smem[tid], smem[tid + 2]);
        smem[tid] = fminf(smem[tid], smem[tid + 1]);
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Tree reduction - max
__global__ void reduceMaxKernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float maxVal = -FLT_MAX;
    if (i < n) maxVal = input[i];
    if (i + blockDim.x < n) maxVal = fmaxf(maxVal, input[i + blockDim.x]);
    sdata[tid] = maxVal;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* smem = sdata;
        smem[tid] = fmaxf(smem[tid], smem[tid + 32]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 16]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 8]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 4]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 2]);
        smem[tid] = fmaxf(smem[tid], smem[tid + 1]);
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Host function to perform full reduction
float reduceSum(float* d_input, float* d_temp, int n) {
    int blocks = (n + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
    int smemSize = REDUCE_BLOCK_SIZE * sizeof(float);
    
    reduceSumKernel<<<blocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_input, d_temp, n);
    
    // Reduce partial results
    while (blocks > 1) {
        int newBlocks = (blocks + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
        reduceSumKernel<<<newBlocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_temp, d_temp, blocks);
        blocks = newBlocks;
    }
    
    float result;
    cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

float reduceMin(float* d_input, float* d_temp, int n) {
    int blocks = (n + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
    int smemSize = REDUCE_BLOCK_SIZE * sizeof(float);
    
    reduceMinKernel<<<blocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_input, d_temp, n);
    
    while (blocks > 1) {
        int newBlocks = (blocks + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
        reduceMinKernel<<<newBlocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_temp, d_temp, blocks);
        blocks = newBlocks;
    }
    
    float result;
    cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

float reduceMax(float* d_input, float* d_temp, int n) {
    int blocks = (n + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
    int smemSize = REDUCE_BLOCK_SIZE * sizeof(float);
    
    reduceMaxKernel<<<blocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_input, d_temp, n);
    
    while (blocks > 1) {
        int newBlocks = (blocks + REDUCE_BLOCK_SIZE * 2 - 1) / (REDUCE_BLOCK_SIZE * 2);
        reduceMaxKernel<<<newBlocks, REDUCE_BLOCK_SIZE, smemSize>>>(d_temp, d_temp, blocks);
        blocks = newBlocks;
    }
    
    float result;
    cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    return result;
}

// ============================================================================
// SCAN KERNELS (Blelloch algorithm)
// ============================================================================

// Single-block exclusive scan (Blelloch)
__global__ void scanBlockExclusiveKernel(int* output, const int* input, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int offset = 1;
    
    // Load input into shared memory (with padding to avoid bank conflicts)
    int ai = tid;
    int bi = tid + (n / 2);
    
    // Bank conflict avoidance: add offset based on index
    int bankOffsetA = ai >> 4;  // CONFLICT_FREE_OFFSET
    int bankOffsetB = bi >> 4;
    
    if (ai < n) temp[ai + bankOffsetA] = input[ai];
    else temp[ai + bankOffsetA] = 0;
    
    if (bi < n) temp[bi + bankOffsetB] = input[bi];
    else temp[bi + bankOffsetB] = 0;
    
    // Up-sweep (reduce) phase
    for (int d = n >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_local = offset * (2 * tid + 1) - 1;
            int bi_local = offset * (2 * tid + 2) - 1;
            ai_local += ai_local >> 4;
            bi_local += bi_local >> 4;
            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;
    }
    
    // Clear the last element (for exclusive scan)
    if (tid == 0) {
        int lastIdx = n - 1 + ((n - 1) >> 4);
        temp[lastIdx] = 0;
    }
    
    // Down-sweep phase
    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_local = offset * (2 * tid + 1) - 1;
            int bi_local = offset * (2 * tid + 2) - 1;
            ai_local += ai_local >> 4;
            bi_local += bi_local >> 4;
            
            int t = temp[ai_local];
            temp[ai_local] = temp[bi_local];
            temp[bi_local] += t;
        }
    }
    __syncthreads();
    
    // Write results to output
    if (ai < n) output[ai] = temp[ai + bankOffsetA];
    if (bi < n) output[bi] = temp[bi + bankOffsetB];
}

// Large array scan - per-block scan with block sums
__global__ void scanBlocksKernel(int* output, int* blockSums, const int* input, int n) {
    extern __shared__ int temp[];
    
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x * 2;
    
    int ai = tid;
    int bi = tid + blockDim.x;
    
    int bankOffsetA = ai >> 4;
    int bankOffsetB = bi >> 4;
    
    // Load into shared memory
    if (blockOffset + ai < n) temp[ai + bankOffsetA] = input[blockOffset + ai];
    else temp[ai + bankOffsetA] = 0;
    
    if (blockOffset + bi < n) temp[bi + bankOffsetB] = input[blockOffset + bi];
    else temp[bi + bankOffsetB] = 0;
    
    int blockSize = blockDim.x * 2;
    int offset = 1;
    
    // Up-sweep
    for (int d = blockSize >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai_local = offset * (2 * tid + 1) - 1;
            int bi_local = offset * (2 * tid + 2) - 1;
            ai_local += ai_local >> 4;
            bi_local += bi_local >> 4;
            temp[bi_local] += temp[ai_local];
        }
        offset *= 2;
    }
    
    // Store block sum and clear last element
    if (tid == 0) {
        int lastIdx = blockSize - 1 + ((blockSize - 1) >> 4);
        if (blockSums != NULL) blockSums[blockIdx.x] = temp[lastIdx];
        temp[lastIdx] = 0;
    }
    
    // Down-sweep
    for (int d = 1; d < blockSize; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai_local = offset * (2 * tid + 1) - 1;
            int bi_local = offset * (2 * tid + 2) - 1;
            ai_local += ai_local >> 4;
            bi_local += bi_local >> 4;
            
            int t = temp[ai_local];
            temp[ai_local] = temp[bi_local];
            temp[bi_local] += t;
        }
    }
    __syncthreads();
    
    // Write output
    if (blockOffset + ai < n) output[blockOffset + ai] = temp[ai + bankOffsetA];
    if (blockOffset + bi < n) output[blockOffset + bi] = temp[bi + bankOffsetB];
}

// Add block sums to scanned blocks
__global__ void addBlockSumsKernel(int* data, const int* blockSums, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    int blockSum = blockSums[blockIdx.x];
    
    if (idx < n) data[idx] += blockSum;
    if (idx + blockDim.x < n) data[idx + blockDim.x] += blockSum;
}

// Host function for large exclusive scan
void exclusiveScan(int* d_output, const int* d_input, int n,
                   int* d_temp, int* d_blockSums) {
    int elementsPerBlock = SCAN_BLOCK_SIZE * 2;
    int blocks = (n + elementsPerBlock - 1) / elementsPerBlock;
    int smemSize = (elementsPerBlock + elementsPerBlock / 16) * sizeof(int);
    
    if (blocks == 1) {
        // Single block - direct scan
        scanBlocksKernel<<<1, SCAN_BLOCK_SIZE, smemSize>>>(d_output, NULL, d_input, n);
    } else {
        // Multi-block scan
        scanBlocksKernel<<<blocks, SCAN_BLOCK_SIZE, smemSize>>>(d_output, d_blockSums, d_input, n);
        
        // Recursively scan block sums
        if (blocks <= elementsPerBlock) {
            scanBlocksKernel<<<1, SCAN_BLOCK_SIZE, smemSize>>>(d_blockSums, NULL, d_blockSums, blocks);
        } else {
            // Need another level (for very large arrays)
            int* d_blockSums2;
            int blocks2 = (blocks + elementsPerBlock - 1) / elementsPerBlock;
            cudaMalloc(&d_blockSums2, blocks2 * sizeof(int));
            exclusiveScan(d_blockSums, d_blockSums, blocks, d_temp, d_blockSums2);
            cudaFree(d_blockSums2);
        }
        
        // Add block sums back
        addBlockSumsKernel<<<blocks, SCAN_BLOCK_SIZE>>>(d_output, d_blockSums, n);
    }
}

// ============================================================================
// STREAM COMPACTION KERNELS
// ============================================================================

// Create predicate array (1 if element satisfies condition, 0 otherwise)
__global__ void createPredicateKernel(int* predicate, const int* input, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Example predicate: is the number prime? (for prime sieve demo)
        predicate[idx] = input[idx];  // input already contains 0/1 markers
    }
}

// Scatter elements based on scan results
__global__ void scatterKernel(int* output, const int* input, const int* predicate,
                               const int* scanResult, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && predicate[idx]) {
        output[scanResult[idx]] = input[idx];
    }
}

// Scatter with value transformation (for indices)
__global__ void scatterIndicesKernel(int* output, const int* predicate,
                                      const int* scanResult, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && predicate[idx]) {
        output[scanResult[idx]] = idx;
    }
}

// ============================================================================
// HISTOGRAM KERNELS
// ============================================================================

// Simple histogram with atomics
__global__ void histogramAtomicKernel(unsigned int* histogram, const unsigned int* data,
                                       int n, int numBins, int shift, int mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = (data[idx] >> shift) & mask;
        atomicAdd(&histogram[bin], 1);
    }
}

// Shared memory histogram (reduces atomic contention)
__global__ void histogramSharedKernel(unsigned int* histogram, const unsigned int* data,
                                       int n, int numBins, int shift, int mask) {
    extern __shared__ unsigned int sharedHist[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared histogram
    for (int i = tid; i < numBins; i += blockDim.x) {
        sharedHist[i] = 0;
    }
    __syncthreads();
    
    // Accumulate in shared memory
    if (idx < n) {
        int bin = (data[idx] >> shift) & mask;
        atomicAdd(&sharedHist[bin], 1);
    }
    __syncthreads();
    
    // Write to global memory
    for (int i = tid; i < numBins; i += blockDim.x) {
        atomicAdd(&histogram[i], sharedHist[i]);
    }
}

// ============================================================================
// DEMO A: PARTICLE SPEED STATISTICS
// ============================================================================

struct Particle {
    float x, y;
    float vx, vy;
};

__global__ void initParticlesKernel(Particle* particles, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple LCG random
        unsigned int state = seed + idx * 1099087573;
        state = state * 1103515245 + 12345;
        float rx = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        state = state * 1103515245 + 12345;
        float ry = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        state = state * 1103515245 + 12345;
        float rvx = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        state = state * 1103515245 + 12345;
        float rvy = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        
        particles[idx].x = rx * WIDTH;
        particles[idx].y = ry * HEIGHT;
        particles[idx].vx = (rvx - 0.5f) * 20.0f;
        particles[idx].vy = (rvy - 0.5f) * 20.0f;
    }
}

__global__ void updateParticlesKernel(Particle* particles, int n, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Particle p = particles[idx];
        
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        // Bounce off walls
        if (p.x < 0) { p.x = -p.x; p.vx = -p.vx * 0.9f; }
        if (p.x > WIDTH) { p.x = 2*WIDTH - p.x; p.vx = -p.vx * 0.9f; }
        if (p.y < 0) { p.y = -p.y; p.vy = -p.vy * 0.9f; }
        if (p.y > HEIGHT) { p.y = 2*HEIGHT - p.y; p.vy = -p.vy * 0.9f; }
        
        // Random acceleration
        unsigned int state = (unsigned int)(p.x * 1000 + p.y) + idx;
        state = state * 1103515245 + 12345;
        float ax = ((state & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * 2.0f;
        state = state * 1103515245 + 12345;
        float ay = ((state & 0x7FFFFFFF) / (float)0x7FFFFFFF - 0.5f) * 2.0f;
        
        p.vx += ax * dt;
        p.vy += ay * dt;
        
        particles[idx] = p;
    }
}

__global__ void computeSpeedsKernel(float* speeds, const Particle* particles, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float vx = particles[idx].vx;
        float vy = particles[idx].vy;
        speeds[idx] = sqrtf(vx * vx + vy * vy);
    }
}

__global__ void renderParticlesKernel(unsigned char* pixels, const Particle* particles,
                                       const float* speeds, int n,
                                       float minSpeed, float maxSpeed) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    
    int idx = (py * WIDTH + px) * 4;
    
    // Fade background
    pixels[idx + 0] = pixels[idx + 0] * 0.95f;
    pixels[idx + 1] = pixels[idx + 1] * 0.95f;
    pixels[idx + 2] = pixels[idx + 2] * 0.95f;
    pixels[idx + 3] = 255;
}

__global__ void drawParticlesKernel(unsigned char* pixels, const Particle* particles,
                                     const float* speeds, int n,
                                     float minSpeed, float maxSpeed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int px = (int)particles[idx].x;
    int py = (int)particles[idx].y;
    
    if (px < 0 || px >= WIDTH || py < 0 || py >= HEIGHT) return;
    
    // Color based on speed
    float t = (speeds[idx] - minSpeed) / (maxSpeed - minSpeed + 0.001f);
    t = fminf(1.0f, fmaxf(0.0f, t));
    
    unsigned char r = (unsigned char)(t * 255);
    unsigned char g = (unsigned char)((1 - t) * 128 + 127);
    unsigned char b = (unsigned char)((1 - t) * 255);
    
    int pidx = (py * WIDTH + px) * 4;
    pixels[pidx + 0] = b;
    pixels[pidx + 1] = g;
    pixels[pidx + 2] = r;
}

// ============================================================================
// DEMO B: PRIME SIEVE WITH STREAM COMPACTION
// ============================================================================

__global__ void initSieveKernel(int* sieve, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // 0 and 1 are not prime
        sieve[idx] = (idx >= 2) ? 1 : 0;
    }
}

__global__ void markCompositesKernel(int* sieve, int n, int prime) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int multiple = prime * prime + idx * prime;
    
    if (multiple < n) {
        sieve[multiple] = 0;
    }
}

// Count primes using reduction
__global__ void countOnesKernel(const int* input, int* output, int n) {
    extern __shared__ int sdata_int[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    int sum = 0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata_int[tid] = sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) sdata_int[tid] += sdata_int[tid + s];
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile int* smem = sdata_int;
        smem[tid] += smem[tid + 32];
        smem[tid] += smem[tid + 16];
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata_int[0];
}

// ============================================================================
// DEMO C: RADIX SORT SKELETON
// ============================================================================

// Single pass of radix sort (sort by one digit)
__global__ void radixSortPassKernel(unsigned int* output, const unsigned int* input,
                                     const unsigned int* histogram,
                                     int n, int shift, int numBins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int bin = (input[idx] >> shift) & (numBins - 1);
        // This is just illustrative - actual radix sort needs proper scatter
    }
}

// ============================================================================
// RENDERING
// ============================================================================

__global__ void clearScreenKernel(unsigned char* pixels, unsigned char r, unsigned char g, unsigned char b) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    
    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = b;
    pixels[idx + 1] = g;
    pixels[idx + 2] = r;
    pixels[idx + 3] = 255;
}

__global__ void drawBarKernel(unsigned char* pixels, int x, int y, int w, int h,
                               unsigned char r, unsigned char g, unsigned char b) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    if (px < x || px >= x + w || py < y || py >= y + h) return;
    
    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = b;
    pixels[idx + 1] = g;
    pixels[idx + 2] = r;
}

// Draw histogram bars - fixed version
__global__ void drawHistogramBarsKernel(unsigned char* pixels, const unsigned int* histogram,
                                         int numBins, unsigned int maxVal, int offsetX, int barWidth,
                                         int maxHeight, int baseY) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    
    // Which bin does this x coordinate belong to?
    int binAreaStart = offsetX;
    int binAreaWidth = numBins * (barWidth + 2);
    
    if (px < binAreaStart || px >= binAreaStart + binAreaWidth) return;
    
    int localX = px - binAreaStart;
    int bin = localX / (barWidth + 2);
    int withinBar = localX % (barWidth + 2);
    
    if (bin < 0 || bin >= numBins) return;
    if (withinBar >= barWidth) return;  // Gap between bars
    
    // Calculate bar height
    int h = (histogram[bin] * maxHeight) / (maxVal + 1);
    int barTop = baseY - h;
    
    if (py < barTop || py >= baseY) return;
    
    // Color based on bin (rainbow)
    float t = bin / (float)numBins;
    unsigned char r = (unsigned char)(sinf(t * 3.14159f * 2.0f) * 127 + 128);
    unsigned char g = (unsigned char)(sinf(t * 3.14159f * 2.0f + 2.094f) * 127 + 128);
    unsigned char b = (unsigned char)(sinf(t * 3.14159f * 2.0f + 4.188f) * 127 + 128);
    
    int idx = (py * WIDTH + px) * 4;
    pixels[idx + 0] = b;
    pixels[idx + 1] = g;
    pixels[idx + 2] = r;
    pixels[idx + 3] = 255;
}

// Draw prime visualization - Ulam spiral style
__global__ void drawPrimesKernel(unsigned char* pixels, const int* primes, int numPrimes,
                                  int maxVal, int centerX, int centerY, int scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPrimes) return;
    
    int prime = primes[idx];
    if (prime <= 1 || prime >= maxVal) return;
    
    // Ulam spiral coordinates
    // n -> (x, y) where we spiral outward
    int n = prime;
    int x = 0, y = 0;
    int dx = 0, dy = -1;
    int segmentLen = 1;
    int segmentPassed = 0;
    int turnsMade = 0;
    
    for (int i = 1; i < n; i++) {
        x += dx;
        y += dy;
        segmentPassed++;
        
        if (segmentPassed == segmentLen) {
            segmentPassed = 0;
            // Turn left
            int temp = dx;
            dx = -dy;
            dy = temp;
            turnsMade++;
            if (turnsMade % 2 == 0) segmentLen++;
        }
    }
    
    int px = centerX + x * scale;
    int py = centerY + y * scale;
    
    if (px < 1 || px >= WIDTH - 1 || py < 1 || py >= HEIGHT - 1) return;
    
    // Color based on prime size
    float t = (float)prime / maxVal;
    unsigned char r = (unsigned char)(255 - t * 200);
    unsigned char g = (unsigned char)(200 + t * 55);
    unsigned char b = (unsigned char)(100 + t * 155);
    
    // Draw a small dot
    for (int dy2 = -1; dy2 <= 1; dy2++) {
        for (int dx2 = -1; dx2 <= 1; dx2++) {
            if (dx2*dx2 + dy2*dy2 <= 1) {  // Circle
                int pidx = ((py + dy2) * WIDTH + (px + dx2)) * 4;
                pixels[pidx + 0] = b;
                pixels[pidx + 1] = g;
                pixels[pidx + 2] = r;
            }
        }
    }
}

// Draw scan visualization - show prefix sum as bars
__global__ void drawScanVisualizationKernel(unsigned char* pixels, const int* scanResult,
                                             int n, int maxVal, int startX, int width) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= WIDTH || py >= HEIGHT) return;
    if (px < startX || px >= startX + width) return;
    
    // Map pixel x to array index
    int idx = (px - startX) * n / width;
    if (idx < 0 || idx >= n) return;
    
    // Get scan value and map to height
    int val = scanResult[idx];
    int barHeight = (val * (HEIGHT - 100)) / (maxVal + 1);
    int barTop = HEIGHT - 50 - barHeight;
    
    if (py < barTop || py >= HEIGHT - 50) return;
    
    // Gradient color
    float t = (float)idx / n;
    unsigned char r = (unsigned char)(50 + t * 200);
    unsigned char g = (unsigned char)(200 - t * 100);
    unsigned char b = (unsigned char)(255 - t * 200);
    
    int pidx = (py * WIDTH + px) * 4;
    pixels[pidx + 0] = b;
    pixels[pidx + 1] = g;
    pixels[pidx + 2] = r;
}

// Simple digit rendering
__device__ void drawDigit(unsigned char* pixels, int digit, int x, int y, 
                          unsigned char r, unsigned char g, unsigned char b) {
    // 3x5 font patterns
    const unsigned char digits[10][5] = {
        {0x7, 0x5, 0x5, 0x5, 0x7},  // 0
        {0x2, 0x6, 0x2, 0x2, 0x7},  // 1
        {0x7, 0x1, 0x7, 0x4, 0x7},  // 2
        {0x7, 0x1, 0x7, 0x1, 0x7},  // 3
        {0x5, 0x5, 0x7, 0x1, 0x1},  // 4
        {0x7, 0x4, 0x7, 0x1, 0x7},  // 5
        {0x7, 0x4, 0x7, 0x5, 0x7},  // 6
        {0x7, 0x1, 0x1, 0x1, 0x1},  // 7
        {0x7, 0x5, 0x7, 0x5, 0x7},  // 8
        {0x7, 0x5, 0x7, 0x1, 0x7}   // 9
    };
    
    if (digit < 0 || digit > 9) return;
    
    for (int dy = 0; dy < 5; dy++) {
        for (int dx = 0; dx < 3; dx++) {
            if (digits[digit][dy] & (0x4 >> dx)) {
                int px = x + dx * 2;
                int py = y + dy * 2;
                if (px >= 0 && px < WIDTH - 1 && py >= 0 && py < HEIGHT - 1) {
                    for (int sy = 0; sy < 2; sy++) {
                        for (int sx = 0; sx < 2; sx++) {
                            int idx = ((py + sy) * WIDTH + px + sx) * 4;
                            pixels[idx + 0] = b;
                            pixels[idx + 1] = g;
                            pixels[idx + 2] = r;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// MAIN PROGRAM
// ============================================================================

int main() {
    printf("=== CUDA Parallel Primitives Toybox ===\n\n");
    
    // Check GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("CUDA Cores: %d\n", prop.multiProcessorCount * 128);
    printf("Shared Memory per Block: %zu bytes\n\n", prop.sharedMemPerBlock);
    
    // X11 setup
    Display* display = XOpenDisplay(NULL);
    if (!display) {
        printf("Cannot open display\n");
        return 1;
    }
    
    int screen = DefaultScreen(display);
    Window root = RootWindow(display, screen);
    
    XVisualInfo vinfo;
    if (!XMatchVisualInfo(display, screen, 24, TrueColor, &vinfo)) {
        printf("No matching visual\n");
        return 1;
    }
    
    XSetWindowAttributes attrs;
    attrs.colormap = XCreateColormap(display, root, vinfo.visual, AllocNone);
    attrs.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
    
    Window window = XCreateWindow(display, root, 0, 0, WIDTH, HEIGHT, 0,
                                   vinfo.depth, InputOutput, vinfo.visual,
                                   CWColormap | CWEventMask, &attrs);
    
    XStoreName(display, window, "CUDA Parallel Primitives Toybox");
    XMapWindow(display, window);
    
    GC gc = XCreateGC(display, window, 0, NULL);
    
    // Wait for window
    XEvent event;
    while (1) {
        XNextEvent(display, &event);
        if (event.type == Expose) break;
    }
    
    // Allocate host pixel buffer
    unsigned char* h_pixels = (unsigned char*)malloc(WIDTH * HEIGHT * 4);
    
    XImage* ximage = XCreateImage(display, vinfo.visual, vinfo.depth, ZPixmap, 0,
                                   (char*)h_pixels, WIDTH, HEIGHT, 32, 0);
    
    // Allocate device memory
    unsigned char* d_pixels;
    CUDA_CHECK(cudaMalloc(&d_pixels, WIDTH * HEIGHT * 4));
    
    // Demo A resources
    int numParticles = 100000;
    Particle* d_particles;
    float* d_speeds;
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_particles, numParticles * sizeof(Particle)));
    CUDA_CHECK(cudaMalloc(&d_speeds, numParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temp, numParticles * sizeof(float)));
    
    // Initialize particles
    int blocks = (numParticles + 255) / 256;
    initParticlesKernel<<<blocks, 256>>>(d_particles, numParticles, time(NULL));
    
    // Demo B resources
    int sieveSize = 1000000;
    int* d_sieve;
    int* d_scanResult;
    int* d_primes;
    int* d_blockSums;
    CUDA_CHECK(cudaMalloc(&d_sieve, sieveSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scanResult, sieveSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_primes, sieveSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockSums, (sieveSize / 256 + 1) * sizeof(int)));
    
    // Demo C resources
    int sortSize = 100000;
    unsigned int* d_sortData;
    unsigned int* d_histogram;
    int numBins = 256;
    CUDA_CHECK(cudaMalloc(&d_sortData, sortSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_histogram, numBins * sizeof(unsigned int)));
    
    // State variables
    int currentDemo = 1;
    bool verbose = false;
    bool running = true;
    float dt = 0.1f;
    
    // Stats
    float minSpeed = 0, maxSpeed = 0, avgSpeed = 0;
    int primeCount = 0;
    unsigned int* h_histogram = (unsigned int*)malloc(numBins * sizeof(unsigned int));
    
    dim3 dispBlockSize(16, 16);
    dim3 dispGridSize((WIDTH + 15) / 16, (HEIGHT + 15) / 16);
    
    double lastTime = getTime();
    double lastFpsTime = lastTime;
    int frameCount = 0;
    
    printf("Controls:\n");
    printf("  1     - Demo A: Particle Statistics (Reduce)\n");
    printf("  2     - Demo B: Prime Sieve (Scan + Compact)\n");
    printf("  3     - Demo C: Histogram Demo\n");
    printf("  Space - Run/refresh demo\n");
    printf("  +/-   - Adjust size\n");
    printf("  V     - Toggle verbose\n");
    printf("  Q     - Quit\n\n");
    
    printf("Starting Demo A: Particle Speed Statistics\n");
    
    while (running) {
        // Handle events
        while (XPending(display)) {
            XNextEvent(display, &event);
            
            if (event.type == KeyPress) {
                KeySym key = XLookupKeysym(&event.xkey, 0);
                
                if (key == XK_Escape || key == XK_q) running = false;
                
                if (key == XK_1 && currentDemo != 1) {
                    currentDemo = 1;
                    printf("\n=== Demo A: Particle Speed Statistics ===\n");
                    printf("Using REDUCE to compute min/max/average speeds\n");
                }
                if (key == XK_2 && currentDemo != 2) {
                    currentDemo = 2;
                    printf("\n=== Demo B: Prime Sieve with Stream Compaction ===\n");
                    printf("Using SCAN + SCATTER for stream compaction\n");
                    
                    // Run prime sieve
                    double t0 = getTime();
                    
                    // Initialize sieve
                    int sBlocks = (sieveSize + 255) / 256;
                    initSieveKernel<<<sBlocks, 256>>>(d_sieve, sieveSize);
                    
                    // Mark composites
                    for (int p = 2; p * p < sieveSize; p++) {
                        int* h_check = (int*)malloc(sizeof(int));
                        cudaMemcpy(h_check, d_sieve + p, sizeof(int), cudaMemcpyDeviceToHost);
                        if (*h_check) {
                            int numMultiples = (sieveSize - p * p) / p + 1;
                            int mBlocks = (numMultiples + 255) / 256;
                            markCompositesKernel<<<mBlocks, 256>>>(d_sieve, sieveSize, p);
                        }
                        free(h_check);
                    }
                    
                    // Count primes using reduction
                    int* d_count;
                    CUDA_CHECK(cudaMalloc(&d_count, ((sieveSize + 511) / 512) * sizeof(int)));
                    
                    int rBlocks = (sieveSize + 511) / 512;
                    countOnesKernel<<<rBlocks, 256, 256 * sizeof(int)>>>(d_sieve, d_count, sieveSize);
                    
                    while (rBlocks > 1) {
                        int newBlocks = (rBlocks + 511) / 512;
                        countOnesKernel<<<newBlocks, 256, 256 * sizeof(int)>>>(d_count, d_count, rBlocks);
                        rBlocks = newBlocks;
                    }
                    
                    cudaMemcpy(&primeCount, d_count, sizeof(int), cudaMemcpyDeviceToHost);
                    cudaFree(d_count);
                    
                    // Compact primes using scan
                    exclusiveScan(d_scanResult, d_sieve, sieveSize, d_primes, d_blockSums);
                    scatterIndicesKernel<<<sBlocks, 256>>>(d_primes, d_sieve, d_scanResult, sieveSize);
                    
                    cudaDeviceSynchronize();
                    double elapsed = getTime() - t0;
                    
                    printf("Sieve size: %d\n", sieveSize);
                    printf("Primes found: %d\n", primeCount);
                    printf("Time: %.3f ms\n", elapsed * 1000);
                    
                    // Show first 20 primes
                    if (verbose) {
                        int* h_primes = (int*)malloc(20 * sizeof(int));
                        cudaMemcpy(h_primes, d_primes, 20 * sizeof(int), cudaMemcpyDeviceToHost);
                        printf("First 20 primes: ");
                        for (int i = 0; i < 20; i++) printf("%d ", h_primes[i]);
                        printf("\n");
                        free(h_primes);
                    }
                }
                if (key == XK_3 && currentDemo != 3) {
                    currentDemo = 3;
                    printf("\n=== Demo C: Histogram (Radix Sort Building Block) ===\n");
                    printf("Computing 8-bit histogram of random data\n");
                    
                    // Generate random data
                    unsigned int* h_data = (unsigned int*)malloc(sortSize * sizeof(unsigned int));
                    for (int i = 0; i < sortSize; i++) {
                        h_data[i] = rand() % 256;
                    }
                    cudaMemcpy(d_sortData, h_data, sortSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
                    free(h_data);
                    
                    // Compute histogram
                    double t0 = getTime();
                    cudaMemset(d_histogram, 0, numBins * sizeof(unsigned int));
                    
                    int hBlocks = (sortSize + HISTOGRAM_BLOCK_SIZE - 1) / HISTOGRAM_BLOCK_SIZE;
                    histogramSharedKernel<<<hBlocks, HISTOGRAM_BLOCK_SIZE, numBins * sizeof(unsigned int)>>>
                        (d_histogram, d_sortData, sortSize, numBins, 0, 0xFF);
                    
                    cudaDeviceSynchronize();
                    double elapsed = getTime() - t0;
                    
                    cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                    
                    printf("Data size: %d elements\n", sortSize);
                    printf("Histogram time: %.3f ms\n", elapsed * 1000);
                    
                    // Find max for display scaling
                    unsigned int maxCount = 0;
                    for (int i = 0; i < numBins; i++) {
                        if (h_histogram[i] > maxCount) maxCount = h_histogram[i];
                    }
                    printf("Max bin count: %u\n", maxCount);
                    
                    if (verbose) {
                        printf("First 16 bins: ");
                        for (int i = 0; i < 16; i++) printf("%u ", h_histogram[i]);
                        printf("\n");
                    }
                }
                
                if (key == XK_v || key == XK_V) {
                    verbose = !verbose;
                    printf("Verbose: %s\n", verbose ? "ON" : "OFF");
                }
                
                if (key == XK_space) {
                    if (currentDemo == 2) {
                        // Re-run prime sieve with current size
                        printf("Re-running prime sieve...\n");
                    }
                }
                
                if (key == XK_plus || key == XK_equal) {
                    if (currentDemo == 1) {
                        numParticles = (int)(numParticles * 1.5f);
                        if (numParticles > 500000) numParticles = 500000;
                        cudaFree(d_particles);
                        cudaFree(d_speeds);
                        cudaMalloc(&d_particles, numParticles * sizeof(Particle));
                        cudaMalloc(&d_speeds, numParticles * sizeof(float));
                        initParticlesKernel<<<(numParticles + 255) / 256, 256>>>(d_particles, numParticles, time(NULL));
                        printf("Particles: %d\n", numParticles);
                    } else if (currentDemo == 2) {
                        sieveSize *= 2;
                        if (sieveSize > 10000000) sieveSize = 10000000;
                        printf("Sieve size: %d (press 2 to recompute)\n", sieveSize);
                    } else if (currentDemo == 3) {
                        sortSize *= 2;
                        if (sortSize > 10000000) sortSize = 10000000;
                        printf("Sort size: %d (press 3 to recompute)\n", sortSize);
                    }
                }
                
                if (key == XK_minus) {
                    if (currentDemo == 1) {
                        numParticles = (int)(numParticles / 1.5f);
                        if (numParticles < 1000) numParticles = 1000;
                        cudaFree(d_particles);
                        cudaFree(d_speeds);
                        cudaMalloc(&d_particles, numParticles * sizeof(Particle));
                        cudaMalloc(&d_speeds, numParticles * sizeof(float));
                        initParticlesKernel<<<(numParticles + 255) / 256, 256>>>(d_particles, numParticles, time(NULL));
                        printf("Particles: %d\n", numParticles);
                    } else if (currentDemo == 2) {
                        sieveSize /= 2;
                        if (sieveSize < 1000) sieveSize = 1000;
                        printf("Sieve size: %d (press 2 to recompute)\n", sieveSize);
                    } else if (currentDemo == 3) {
                        sortSize /= 2;
                        if (sortSize < 1000) sortSize = 1000;
                        printf("Sort size: %d (press 3 to recompute)\n", sortSize);
                    }
                }
            }
            
            if (event.type == DestroyNotify) running = false;
        }
        
        // Update and render based on current demo
        if (currentDemo == 1) {
            // Demo A: Particle statistics
            blocks = (numParticles + 255) / 256;
            
            // Update particles
            updateParticlesKernel<<<blocks, 256>>>(d_particles, numParticles, dt);
            
            // Compute speeds
            computeSpeedsKernel<<<blocks, 256>>>(d_speeds, d_particles, numParticles);
            
            // Reduce to get statistics
            minSpeed = reduceMin(d_speeds, d_temp, numParticles);
            maxSpeed = reduceMax(d_speeds, d_temp, numParticles);
            float sumSpeed = reduceSum(d_speeds, d_temp, numParticles);
            avgSpeed = sumSpeed / numParticles;
            
            // Render
            renderParticlesKernel<<<dispGridSize, dispBlockSize>>>(d_pixels, d_particles,
                d_speeds, numParticles, minSpeed, maxSpeed);
            drawParticlesKernel<<<(numParticles + 255) / 256, 256>>>(d_pixels, d_particles,
                d_speeds, numParticles, minSpeed, maxSpeed);
                
        } else if (currentDemo == 2) {
            // Demo B: Prime sieve visualization - Ulam spiral
            clearScreenKernel<<<dispGridSize, dispBlockSize>>>(d_pixels, 10, 10, 20);
            
            // Draw primes as Ulam spiral
            if (primeCount > 0) {
                int scale = 2;  // Adjust based on sieve size
                if (sieveSize > 100000) scale = 1;
                
                int primeBlocks = (primeCount + 255) / 256;
                drawPrimesKernel<<<primeBlocks, 256>>>(d_pixels, d_primes, primeCount,
                    sieveSize, WIDTH / 2, HEIGHT / 2, scale);
            }
            
        } else if (currentDemo == 3) {
            // Demo C: Histogram visualization
            clearScreenKernel<<<dispGridSize, dispBlockSize>>>(d_pixels, 15, 15, 25);
            
            // Draw histogram bars
            cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            unsigned int maxCount = 1;
            for (int i = 0; i < numBins; i++) {
                if (h_histogram[i] > maxCount) maxCount = h_histogram[i];
            }
            
            int barWidth = 3;
            int maxHeight = HEIGHT - 100;
            
            drawHistogramBarsKernel<<<dispGridSize, dispBlockSize>>>
                (d_pixels, d_histogram, numBins, maxCount, 50, barWidth, maxHeight, HEIGHT - 50);
        }
        
        // Copy to host and display
        cudaMemcpy(h_pixels, d_pixels, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        XPutImage(display, window, gc, ximage, 0, 0, 0, 0, WIDTH, HEIGHT);
        
        // FPS calculation
        frameCount++;
        double currentTime = getTime();
        if (currentTime - lastFpsTime >= 1.0) {
            float fps = frameCount / (currentTime - lastFpsTime);
            
            char title[256];
            if (currentDemo == 1) {
                snprintf(title, sizeof(title), 
                    "Primitives - Demo A: %d particles | Min: %.1f Max: %.1f Avg: %.1f | %.1f FPS",
                    numParticles, minSpeed, maxSpeed, avgSpeed, fps);
            } else if (currentDemo == 2) {
                snprintf(title, sizeof(title),
                    "Primitives - Demo B: Ulam Spiral - %d primes up to %d | %.1f FPS",
                    primeCount, sieveSize, fps);
            } else {
                snprintf(title, sizeof(title),
                    "Primitives - Demo C: Histogram - %d elements, %d bins | %.1f FPS",
                    sortSize, numBins, fps);
            }
            XStoreName(display, window, title);
            
            frameCount = 0;
            lastFpsTime = currentTime;
        }
        
        lastTime = currentTime;
    }
    
    // Cleanup
    cudaFree(d_pixels);
    cudaFree(d_particles);
    cudaFree(d_speeds);
    cudaFree(d_temp);
    cudaFree(d_sieve);
    cudaFree(d_scanResult);
    cudaFree(d_primes);
    cudaFree(d_blockSums);
    cudaFree(d_sortData);
    cudaFree(d_histogram);
    
    free(h_histogram);
    
    XDestroyImage(ximage);
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    
    printf("\nDone!\n");
    return 0;
}
