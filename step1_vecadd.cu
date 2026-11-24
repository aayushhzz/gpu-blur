#include <cmath>           // fabsf() for floating-point epsilon comparison
#include <cstdio>          // printf(), fprintf() for printing results and errors
#include <cstdlib>         // exit(), EXIT_FAILURE to abort on CUDA errors
#include <vector>          // std::vector for safe, automatically freed host (CPU) arrays
#include <cuda_runtime.h>  // CUDA runtime API: cudaMalloc, cudaMemcpy, cudaError_t, etc.

// Reusable macro that checks the return code of any CUDA runtime call
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);            /* execute call and capture error code */ \
        if (err != cudaSuccess) {            /* cudaSuccess == 0 means no error */     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__,      /* show exact source file and line */     \
                    cudaGetErrorString(err));/* human-readable CUDA error message */   \
            exit(EXIT_FAILURE);              /* fail fast instead of using bad data */ \
        }                                                                    \
    } while (0)                              // do-while(0) makes the macro behave like one statement

// __global__ means: runs on the GPU (device), called from the CPU (host)
// const pointers tell the reader and compiler that a and b are read-only inputs
__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    // Compute this thread's unique 1D global index from its block ID, block width, and local thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Grid size is rounded up, so the last block has extra threads past n; stop them from writing out of bounds
    if (i >= n) return;
    // Core data-parallel work: each thread adds exactly one pair of elements
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 1000000;                  // 1 million floats to add in parallel
    const size_t bytes = N * sizeof(float); // CUDA allocation/copy APIs take byte counts (~4 MB per array)

    // Allocate host (CPU) arrays; h_ prefix is a common convention for host memory
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = i * 0.5f;                  // Deterministic non-trivial float values for easy debugging
        h_b[i] = (N - i) * 0.25f;           // Different slope so every element has a distinct sum
    }

    // Device (GPU) pointers; d_ prefix reminds us NOT to dereference these on the CPU
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));    // Reserve 4 MB in GPU global memory for input A
    CUDA_CHECK(cudaMalloc(&d_b, bytes));    // Reserve 4 MB in GPU global memory for input B
    CUDA_CHECK(cudaMalloc(&d_c, bytes));    // Reserve 4 MB in GPU global memory for output C

    // Copy inputs across PCIe from CPU RAM (host) into GPU VRAM (device)
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;                        // 256 threads/block = 8 warps (32 threads/warp)
    const int gridSize = (N + blockSize - 1) / blockSize; // Ceiling division: 3907 blocks = 1,000,192 threads

    // Launch kernel on the GPU asynchronously; CPU immediately moves to the next line
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());         // Catch invalid launch parameters (e.g. bad block/grid size)
    CUDA_CHECK(cudaDeviceSynchronize());    // Block CPU until GPU finishes; catches runtime faults inside kernel

    // Copy GPU results back to CPU RAM so we can inspect and verify them
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;                     // Count any elements that disagree beyond tolerance
    float maxErr = 0.0f;                    // Track worst-case absolute error across all 1M elements
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];   // Compute ground-truth reference on the CPU
        float err = fabsf(h_c[i] - expected); // Absolute difference between GPU and CPU result
        if (err > maxErr) maxErr = err;     // Keep the largest error seen so far
        if (err > 1e-5f) mismatches++;      // Flag mismatch if outside floating-point tolerance
    }

    // Print launch geometry and verification summary
    printf("N=%d | grid=%d, block=%d | mismatches=%d | maxErr=%.6f\n",
           N, gridSize, blockSize, mismatches, maxErr);

    // Release GPU memory (unlike std::vector, raw cudaMalloc pointers are not freed automatically)
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // Return 0 (success) only if every single element matched within tolerance
    return mismatches == 0 ? 0 : 1;
}
