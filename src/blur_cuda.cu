// blur_cuda.cu - CUDA 2D Box Blur (naive + shared-memory tiled) and
// Separable Gaussian Blur with CPU reference verification.
//
// Build on a CUDA-capable machine (Linux / Colab):
//   nvcc -O2 -std=c++17 -Iinclude src/blur_cuda.cu src/blur_cpu.cpp -o build/blur_cuda
// Run:
//   ./build/blur_cuda [width] [height] [radius] [sigma]

#include "blur_cpu.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

constexpr int MAX_RADIUS = 32;
constexpr int MAX_KERNEL_SIZE = 2 * MAX_RADIUS + 1;
constexpr int TILE_W = 16;
constexpr int TILE_H = 16;

// 1D Gaussian kernel stored in GPU constant memory (cached & broadcast to warps).
__constant__ float d_gauss_kernel[MAX_KERNEL_SIZE];

__device__ __forceinline__ int d_clamp(int v, int limit) {
    return v < 0 ? 0 : (v >= limit ? limit - 1 : v);
}

// ---------------------------------------------------------------------------
// Kernel 1: Naive 2D Box Blur (reads directly from global memory)
// ---------------------------------------------------------------------------
__global__ void boxBlurNaiveKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int side = 2 * radius + 1;
    const float inv_area = 1.0f / static_cast<float>(side * side);

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int dy = -radius; dy <= radius; ++dy) {
            int sy = d_clamp(y + dy, height);
            for (int dx = -radius; dx <= radius; ++dx) {
                int sx = d_clamp(x + dx, width);
                sum += src[(sy * width + sx) * channels + c];
            }
        }
        dst[(y * width + x) * channels + c] = sum * inv_area;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: Shared-Memory Tiled 2D Box Blur (per-channel tile + halo)
// Dynamic shared memory size: (TILE_W + 2*radius) * (TILE_H + 2*radius) * sizeof(float)
// ---------------------------------------------------------------------------
__global__ void boxBlurTiledKernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int width,
    int height,
    int channels,
    int radius) {
    extern __shared__ float tile[];

    const int tile_pitch = TILE_W + 2 * radius;
    const int tile_rows = TILE_H + 2 * radius;
    const int tile_origin_x = blockIdx.x * TILE_W - radius;
    const int tile_origin_y = blockIdx.y * TILE_H - radius;

    const int x = blockIdx.x * TILE_W + threadIdx.x;
    const int y = blockIdx.y * TILE_H + threadIdx.y;

    const int side = 2 * radius + 1;
    const float inv_area = 1.0f / static_cast<float>(side * side);

    for (int c = 0; c < channels; ++c) {
        // Cooperatively load interior + halo pixels into shared memory
        for (int ty = threadIdx.y; ty < tile_rows; ty += TILE_H) {
            int sy = d_clamp(tile_origin_y + ty, height);
            for (int tx = threadIdx.x; tx < tile_pitch; tx += TILE_W) {
                int sx = d_clamp(tile_origin_x + tx, width);
                tile[ty * tile_pitch + tx] = src[(sy * width + sx) * channels + c];
            }
        }
        __syncthreads();

        if (x < width && y < height) {
            float sum = 0.0f;
            const int cx = threadIdx.x + radius;
            const int cy = threadIdx.y + radius;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    sum += tile[(cy + dy) * tile_pitch + (cx + dx)];
                }
            }
            dst[(y * width + x) * channels + c] = sum * inv_area;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Kernel 3a & 3b: Separable Gaussian Blur (Horizontal + Vertical 1D passes)
// ---------------------------------------------------------------------------
__global__ void gaussianBlurHorizontalKernel(
    const float* __restrict__ src,
    float* __restrict__ tmp,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int dx = -radius; dx <= radius; ++dx) {
            int sx = d_clamp(x + dx, width);
            sum += src[(y * width + sx) * channels + c] * d_gauss_kernel[dx + radius];
        }
        tmp[(y * width + x) * channels + c] = sum;
    }
}

__global__ void gaussianBlurVerticalKernel(
    const float* __restrict__ tmp,
    float* __restrict__ dst,
    int width,
    int height,
    int channels,
    int radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        for (int dy = -radius; dy <= radius; ++dy) {
            int sy = d_clamp(y + dy, height);
            sum += tmp[(sy * width + x) * channels + c] * d_gauss_kernel[dy + radius];
        }
        dst[(y * width + x) * channels + c] = sum;
    }
}

int main(int argc, char** argv) {
    const int width = (argc > 1) ? std::atoi(argv[1]) : 1024;
    const int height = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int radius = (argc > 3) ? std::atoi(argv[3]) : 5;
    const float sigma = (argc > 4) ? static_cast<float>(std::atof(argv[4])) : 2.0f;
    const int channels = 3;

    if (radius < 0 || radius > MAX_RADIUS) {
        fprintf(stderr, "Error: radius must be in [0, %d]\n", MAX_RADIUS);
        return EXIT_FAILURE;
    }

    gpu_blur::Image h_src = gpu_blur::make_synthetic_image(width, height, channels);
    const size_t bytes = h_src.data.size() * sizeof(float);

    // CPU reference timings
    auto t0 = std::chrono::steady_clock::now();
    gpu_blur::Image ref_box = gpu_blur::box_blur_cpu(h_src, radius);
    auto t1 = std::chrono::steady_clock::now();
    gpu_blur::Image ref_gauss = gpu_blur::gaussian_blur_separable_cpu(h_src, radius, sigma);
    auto t2 = std::chrono::steady_clock::now();

    double cpu_box_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double cpu_gauss_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Allocate GPU memory
    float *d_src = nullptr, *d_tmp = nullptr, *d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data.data(), bytes, cudaMemcpyHostToDevice));

    // Upload Gaussian 1D kernel to constant memory
    std::vector<float> h_kernel = gpu_blur::make_gaussian_kernel_1d(radius, sigma);
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_gauss_kernel, h_kernel.data(), h_kernel.size() * sizeof(float)));

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width + TILE_W - 1) / TILE_W, (height + TILE_H - 1) / TILE_H);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    gpu_blur::Image h_out(width, height, channels);
    bool all_ok = true;

    // 1. Naive Box Blur
    CUDA_CHECK(cudaEventRecord(ev_start));
    boxBlurNaiveKernel<<<grid, block>>>(d_src, d_dst, width, height, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float naive_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_out.data.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    auto cmp_naive = gpu_blur::compare_images(h_out, ref_box, 1e-3f);
    all_ok &= (cmp_naive.mismatches == 0);
    printf("[box-naive]    GPU=%.3f ms | CPU=%.3f ms | speedup=%.1fx | maxErr=%.6f | mismatches=%d\n",
           naive_ms, cpu_box_ms, cpu_box_ms / naive_ms, cmp_naive.max_abs_err, cmp_naive.mismatches);

    // 2. Shared-Memory Tiled Box Blur
    size_t smem_bytes = static_cast<size_t>(TILE_W + 2 * radius) *
                        static_cast<size_t>(TILE_H + 2 * radius) * sizeof(float);
    CUDA_CHECK(cudaEventRecord(ev_start));
    boxBlurTiledKernel<<<grid, block, smem_bytes>>>(d_src, d_dst, width, height, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float tiled_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tiled_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_out.data.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    auto cmp_tiled = gpu_blur::compare_images(h_out, ref_box, 1e-3f);
    all_ok &= (cmp_tiled.mismatches == 0);
    printf("[box-tiled]    GPU=%.3f ms | CPU=%.3f ms | speedup=%.1fx | maxErr=%.6f | mismatches=%d\n",
           tiled_ms, cpu_box_ms, cpu_box_ms / tiled_ms, cmp_tiled.max_abs_err, cmp_tiled.mismatches);

    // 3. Separable Gaussian Blur
    CUDA_CHECK(cudaEventRecord(ev_start));
    gaussianBlurHorizontalKernel<<<grid, block>>>(d_src, d_tmp, width, height, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    gaussianBlurVerticalKernel<<<grid, block>>>(d_tmp, d_dst, width, height, channels, radius);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float gauss_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gauss_ms, ev_start, ev_stop));
    CUDA_CHECK(cudaMemcpy(h_out.data.data(), d_dst, bytes, cudaMemcpyDeviceToHost));
    auto cmp_gauss = gpu_blur::compare_images(h_out, ref_gauss, 1e-3f);
    all_ok &= (cmp_gauss.mismatches == 0);
    printf("[gauss-sep]    GPU=%.3f ms | CPU=%.3f ms | speedup=%.1fx | maxErr=%.6f | mismatches=%d\n",
           gauss_ms, cpu_gauss_ms, cpu_gauss_ms / gauss_ms, cmp_gauss.max_abs_err, cmp_gauss.mismatches);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_dst));

    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
