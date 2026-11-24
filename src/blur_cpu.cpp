#include "blur_cpu.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpu_blur {

std::vector<float> make_gaussian_kernel_1d(int radius, float sigma) {
    if (radius < 0) {
        throw std::invalid_argument("radius must be >= 0");
    }
    int size = 2 * radius + 1;
    std::vector<float> kernel(size, 0.0f);
    if (radius == 0 || sigma <= 0.0f) {
        kernel[radius] = 1.0f;
        return kernel;
    }

    double sum = 0.0;
    double inv_two_sigma2 = 1.0 / (2.0 * static_cast<double>(sigma) * static_cast<double>(sigma));
    for (int i = -radius; i <= radius; ++i) {
        double val = std::exp(-(i * i) * inv_two_sigma2);
        kernel[i + radius] = static_cast<float>(val);
        sum += val;
    }
    for (float& w : kernel) {
        w = static_cast<float>(w / sum);
    }
    return kernel;
}

Image box_blur_cpu(const Image& src, int radius) {
    if (radius < 0) {
        throw std::invalid_argument("radius must be >= 0");
    }
    Image dst(src.width, src.height, src.channels);
    if (radius == 0) {
        dst.data = src.data;
        return dst;
    }

    const int side = 2 * radius + 1;
    const float inv_area = 1.0f / static_cast<float>(side * side);

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = clamp_coord(y + dy, src.height);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int sx = clamp_coord(x + dx, src.width);
                        sum += src.at(sx, sy, c);
                    }
                }
                dst.at(x, y, c) = sum * inv_area;
            }
        }
    }
    return dst;
}

Image gaussian_blur_2d_cpu(const Image& src, int radius, float sigma) {
    if (radius < 0) {
        throw std::invalid_argument("radius must be >= 0");
    }
    Image dst(src.width, src.height, src.channels);
    if (radius == 0) {
        dst.data = src.data;
        return dst;
    }

    std::vector<float> k1d = make_gaussian_kernel_1d(radius, sigma);
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = clamp_coord(y + dy, src.height);
                    float wy = k1d[dy + radius];
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int sx = clamp_coord(x + dx, src.width);
                        float wx = k1d[dx + radius];
                        sum += src.at(sx, sy, c) * (wx * wy);
                    }
                }
                dst.at(x, y, c) = sum;
            }
        }
    }
    return dst;
}

Image gaussian_blur_separable_cpu(const Image& src, int radius, float sigma) {
    if (radius < 0) {
        throw std::invalid_argument("radius must be >= 0");
    }
    if (radius == 0) {
        Image dst = src;
        return dst;
    }

    std::vector<float> kernel = make_gaussian_kernel_1d(radius, sigma);
    Image tmp(src.width, src.height, src.channels);
    Image dst(src.width, src.height, src.channels);

    // Horizontal pass
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                float sum = 0.0f;
                for (int dx = -radius; dx <= radius; ++dx) {
                    int sx = clamp_coord(x + dx, src.width);
                    sum += src.at(sx, y, c) * kernel[dx + radius];
                }
                tmp.at(x, y, c) = sum;
            }
        }
    }

    // Vertical pass
    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            for (int c = 0; c < src.channels; ++c) {
                float sum = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int sy = clamp_coord(y + dy, src.height);
                    sum += tmp.at(x, sy, c) * kernel[dy + radius];
                }
                dst.at(x, y, c) = sum;
            }
        }
    }

    return dst;
}

CompareResult compare_images(const Image& a, const Image& b, float tol) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels ||
        a.data.size() != b.data.size()) {
        throw std::invalid_argument("Image dimensions do not match");
    }
    CompareResult res;
    double sq_sum = 0.0;
    for (size_t i = 0; i < a.data.size(); ++i) {
        float err = std::fabs(a.data[i] - b.data[i]);
        if (err > res.max_abs_err) {
            res.max_abs_err = err;
        }
        if (err > tol) {
            res.mismatches++;
        }
        sq_sum += static_cast<double>(err) * static_cast<double>(err);
    }
    if (!a.data.empty()) {
        res.rmse = std::sqrt(sq_sum / static_cast<double>(a.data.size()));
    }
    return res;
}

Image make_synthetic_image(int width, int height, int channels) {
    Image img(width, height, channels);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool checker = (((x / 16) + (y / 16)) % 2) == 0;
            for (int c = 0; c < channels; ++c) {
                float grad = 0.0f;
                if (c == 0) {
                    grad = (width > 1) ? (255.0f * x / (width - 1)) : 128.0f;
                } else if (c == 1) {
                    grad = (height > 1) ? (255.0f * y / (height - 1)) : 128.0f;
                } else {
                    grad = checker ? 220.0f : 35.0f;
                }
                img.at(x, y, c) = grad;
            }
        }
    }
    return img;
}

bool write_ppm(const std::string& path, const Image& img) {
    if (img.channels != 3) {
        return false;
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }
    out << "P6\n" << img.width << " " << img.height << "\n255\n";
    std::vector<uint8_t> bytes(img.data.size());
    for (size_t i = 0; i < img.data.size(); ++i) {
        float v = std::clamp(img.data[i], 0.0f, 255.0f);
        bytes[i] = static_cast<uint8_t>(std::lround(v));
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return static_cast<bool>(out);
}

Image read_ppm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open PPM file: " + path);
    }
    std::string magic;
    in >> magic;
    if (magic != "P6") {
        throw std::runtime_error("Unsupported PPM format (expected P6): " + magic);
    }
    auto skip_comments = [&in]() {
        in >> std::ws;
        while (in.peek() == '#') {
            std::string line;
            std::getline(in, line);
            in >> std::ws;
        }
    };
    int width = 0, height = 0, maxval = 0;
    skip_comments();
    in >> width;
    skip_comments();
    in >> height;
    skip_comments();
    in >> maxval;
    if (maxval != 255 || width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid PPM header");
    }
    in.get();  // consume single whitespace after maxval

    Image img(width, height, 3);
    std::vector<uint8_t> bytes(img.data.size());
    in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!in) {
        throw std::runtime_error("Truncated PPM pixel data");
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
        img.data[i] = static_cast<float>(bytes[i]);
    }
    return img;
}

}  // namespace gpu_blur
