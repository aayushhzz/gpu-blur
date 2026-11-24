#include "blur_cpu.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    int width = (argc > 1) ? std::atoi(argv[1]) : 512;
    int height = (argc > 2) ? std::atoi(argv[2]) : 512;
    int radius = (argc > 3) ? std::atoi(argv[3]) : 5;
    float sigma = (argc > 4) ? static_cast<float>(std::atof(argv[4])) : 2.0f;
    std::string out_prefix = (argc > 5) ? argv[5] : "";

    gpu_blur::Image src = gpu_blur::make_synthetic_image(width, height, 3);

    auto t0 = std::chrono::steady_clock::now();
    gpu_blur::Image box = gpu_blur::box_blur_cpu(src, radius);
    auto t1 = std::chrono::steady_clock::now();
    gpu_blur::Image gauss = gpu_blur::gaussian_blur_separable_cpu(src, radius, sigma);
    auto t2 = std::chrono::steady_clock::now();

    double box_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gauss_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

    printf("CPU reference (%dx%dx3, radius=%d, sigma=%.2f)\n", width, height, radius, sigma);
    printf("  box_blur_cpu:                %.3f ms\n", box_ms);
    printf("  gaussian_blur_separable_cpu: %.3f ms\n", gauss_ms);

    if (!out_prefix.empty()) {
        gpu_blur::write_ppm(out_prefix + "_src.ppm", src);
        gpu_blur::write_ppm(out_prefix + "_box.ppm", box);
        gpu_blur::write_ppm(out_prefix + "_gauss.ppm", gauss);
        printf("  wrote %s_{src,box,gauss}.ppm\n", out_prefix.c_str());
    }
    return 0;
}
