# gpu-blur

CUDA 2D image blur kernels (naive box blur, shared-memory tiled box blur, and separable Gaussian blur) paired with a portable C++17 CPU reference implementation and verification test suite.

## Project layout

```
gpu-blur/
├── include/blur_cpu.hpp   # CPU reference API, synthetic image generator, PPM I/O, metrics
├── src/blur_cpu.cpp       # Box blur, direct & separable Gaussian blur (clamp-to-edge)
├── src/blur_cuda.cu       # CUDA kernels + GPU timing & CPU verification harness
├── src/main_cpu.cpp       # Standalone CPU benchmark / PPM output CLI
├── tests/test_blur_cpu.cpp# Unit tests for kernels, boundary clamping, separability, PPM I/O
├── step1_vecadd.cu        # Step 1 CUDA warmup: 1D vector addition
└── Makefile
```

## Kernels

1. **Naive 2D Box Blur (`boxBlurNaiveKernel`)** — One thread per pixel, reading the `(2r + 1) x (2r + 1)` neighborhood directly from global memory with clamp-to-edge boundaries.
2. **Shared-Memory Tiled 2D Box Blur (`boxBlurTiledKernel`)** — Cooperatively loads a `16 x 16` tile plus `r`-pixel halo into dynamic `__shared__` memory per channel, reducing global memory traffic.
3. **Separable Gaussian Blur (`gaussianBlurHorizontalKernel`, `gaussianBlurVerticalKernel`)** — Uploads the normalized 1D Gaussian weights into `__constant__` memory (`d_gauss_kernel`) and runs two `O(r)` passes instead of one `O(r^2)` pass.

## Building and testing on macOS / Linux (CPU reference)

macOS does not include `nvcc` / CUDA, so the CPU reference and unit tests compile cleanly with `clang++` or `g++`:

```bash
make test     # builds and runs 8 CPU reference unit tests
make bench    # runs CPU benchmark (512x512 RGB, radius=5, sigma=2.0)
```

To also write synthetic source and blurred PPM images:

```bash
./build/blur_cpu_cli 512 512 5 2.0 /tmp/demo
```

## Building and running CUDA kernels (Linux / Google Colab with NVIDIA GPU)

```bash
make cuda
./build/step1_vecadd
./build/blur_cuda 1024 1024 5 2.0
```

Or directly with `nvcc` (e.g. in a Colab notebook):

```bash
nvcc -O2 -std=c++17 -Iinclude src/blur_cuda.cu src/blur_cpu.cpp -o blur_cuda
./blur_cuda 1024 1024 5 2.0
```

`blur_cuda` times each GPU kernel using `cudaEvent_t`, compares every output pixel against `box_blur_cpu` / `gaussian_blur_separable_cpu`, prints speedup and `maxErr` / `mismatches`, and exits non-zero if any pixel exceeds tolerance.
