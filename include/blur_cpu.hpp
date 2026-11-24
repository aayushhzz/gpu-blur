#ifndef GPU_BLUR_CPU_HPP
#define GPU_BLUR_CPU_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_blur {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 3;  // 1 (grayscale) or 3 (RGB)
    std::vector<float> data;  // row-major interleaved: (y * width + x) * channels + c, values in [0, 255]

    Image() = default;
    Image(int w, int h, int c = 3)
        : width(w), height(h), channels(c),
          data(static_cast<size_t>(w) * static_cast<size_t>(h) * static_cast<size_t>(c), 0.0f) {}

    float& at(int x, int y, int c) {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }

    float at(int x, int y, int c) const {
        return data[(static_cast<size_t>(y) * width + x) * channels + c];
    }
};

struct CompareResult {
    int mismatches = 0;
    float max_abs_err = 0.0f;
    double rmse = 0.0;
};

// Clamp coordinate to [0, limit - 1].
inline int clamp_coord(int v, int limit) {
    if (v < 0) return 0;
    if (v >= limit) return limit - 1;
    return v;
}

// Build a normalized 1D Gaussian kernel of length (2 * radius + 1).
std::vector<float> make_gaussian_kernel_1d(int radius, float sigma);

// 2D Box blur with clamp-to-edge boundary handling.
Image box_blur_cpu(const Image& src, int radius);

// 2D Direct (non-separable) Gaussian blur (ground-truth reference for separable).
Image gaussian_blur_2d_cpu(const Image& src, int radius, float sigma);

// Separable 2D Gaussian blur (horizontal 1D pass followed by vertical 1D pass).
Image gaussian_blur_separable_cpu(const Image& src, int radius, float sigma);

// Compare two images element-wise within tolerance.
CompareResult compare_images(const Image& a, const Image& b, float tol = 1e-4f);

// Generate a deterministic synthetic RGB test image (gradient + checkerboard).
Image make_synthetic_image(int width, int height, int channels = 3);

// Binary PPM (P6) I/O for 3-channel images (clamps floats to [0, 255]).
bool write_ppm(const std::string& path, const Image& img);
Image read_ppm(const std::string& path);

}  // namespace gpu_blur

#endif  // GPU_BLUR_CPU_HPP
