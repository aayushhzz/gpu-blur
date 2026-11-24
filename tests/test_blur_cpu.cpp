#include "blur_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
        g_failures++;
    }
}

#define CHECK(expr) check((expr), #expr, __FILE__, __LINE__)

void test_gaussian_kernel_normalized_and_symmetric() {
    auto k = gpu_blur::make_gaussian_kernel_1d(4, 1.5f);
    CHECK(k.size() == 9);
    float sum = std::accumulate(k.begin(), k.end(), 0.0f);
    CHECK(std::fabs(sum - 1.0f) < 1e-5f);
    for (int i = 0; i < 4; ++i) {
        CHECK(std::fabs(k[i] - k[8 - i]) < 1e-6f);
    }
    // Peak should be at the center
    CHECK(k[4] > k[3] && k[3] > k[2] && k[2] > k[1] && k[1] > k[0]);
    printf("[PASS] test_gaussian_kernel_normalized_and_symmetric\n");
}

void test_radius_zero_is_identity() {
    auto src = gpu_blur::make_synthetic_image(32, 24, 3);
    auto box = gpu_blur::box_blur_cpu(src, 0);
    auto gauss = gpu_blur::gaussian_blur_separable_cpu(src, 0, 2.0f);
    CHECK(gpu_blur::compare_images(src, box, 1e-6f).mismatches == 0);
    CHECK(gpu_blur::compare_images(src, gauss, 1e-6f).mismatches == 0);
    printf("[PASS] test_radius_zero_is_identity\n");
}

void test_constant_image_preserved() {
    gpu_blur::Image img(25, 19, 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            img.at(x, y, 0) = 42.0f;
            img.at(x, y, 1) = 128.5f;
            img.at(x, y, 2) = 250.0f;
        }
    }
    auto box = gpu_blur::box_blur_cpu(img, 3);
    auto gauss = gpu_blur::gaussian_blur_separable_cpu(img, 3, 1.8f);
    CHECK(gpu_blur::compare_images(img, box, 1e-4f).mismatches == 0);
    CHECK(gpu_blur::compare_images(img, gauss, 1e-4f).mismatches == 0);
    printf("[PASS] test_constant_image_preserved\n");
}

void test_box_blur_impulse_response() {
    const int W = 11, H = 11, R = 2;
    gpu_blur::Image img(W, H, 1);
    img.at(5, 5, 0) = 250.0f;
    auto out = gpu_blur::box_blur_cpu(img, R);

    const float expected = 250.0f / static_cast<float>((2 * R + 1) * (2 * R + 1));
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool inside = (std::abs(x - 5) <= R) && (std::abs(y - 5) <= R);
            float want = inside ? expected : 0.0f;
            CHECK(std::fabs(out.at(x, y, 0) - want) < 1e-5f);
        }
    }
    printf("[PASS] test_box_blur_impulse_response\n");
}

void test_box_blur_corner_clamping() {
    // 3x3 image with only (0,0) = 90.0f; with radius=1, corner (0,0) samples
    // (-1,-1), (0,-1), (-1,0), (0,0) which all clamp to (0,0) -> 4 of 9 samples.
    gpu_blur::Image img(3, 3, 1);
    img.at(0, 0, 0) = 90.0f;
    auto out = gpu_blur::box_blur_cpu(img, 1);
    CHECK(std::fabs(out.at(0, 0, 0) - 40.0f) < 1e-5f);
    CHECK(std::fabs(out.at(2, 2, 0) - 0.0f) < 1e-5f);
    printf("[PASS] test_box_blur_corner_clamping\n");
}

void test_separable_matches_direct_2d_gaussian() {
    auto src = gpu_blur::make_synthetic_image(48, 37, 3);
    auto direct = gpu_blur::gaussian_blur_2d_cpu(src, 4, 1.75f);
    auto sep = gpu_blur::gaussian_blur_separable_cpu(src, 4, 1.75f);
    auto cmp = gpu_blur::compare_images(direct, sep, 1e-4f);
    CHECK(cmp.mismatches == 0);
    CHECK(cmp.max_abs_err < 1e-4f);
    printf("[PASS] test_separable_matches_direct_2d_gaussian (maxErr=%.2e)\n", cmp.max_abs_err);
}

void test_channels_are_independent() {
    gpu_blur::Image img(16, 16, 3);
    img.at(8, 8, 1) = 200.0f;  // green channel only
    auto box = gpu_blur::box_blur_cpu(img, 2);
    auto gauss = gpu_blur::gaussian_blur_separable_cpu(img, 2, 1.0f);
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            CHECK(box.at(x, y, 0) == 0.0f && box.at(x, y, 2) == 0.0f);
            CHECK(gauss.at(x, y, 0) == 0.0f && gauss.at(x, y, 2) == 0.0f);
        }
    }
    CHECK(box.at(8, 8, 1) > 0.0f);
    CHECK(gauss.at(8, 8, 1) > 0.0f);
    printf("[PASS] test_channels_are_independent\n");
}

void test_ppm_round_trip() {
    gpu_blur::Image img(17, 13, 3);
    for (int y = 0; y < img.height; ++y) {
        for (int x = 0; x < img.width; ++x) {
            img.at(x, y, 0) = static_cast<float>((x * 15) % 256);
            img.at(x, y, 1) = static_cast<float>((y * 19) % 256);
            img.at(x, y, 2) = static_cast<float>(((x + y) * 7) % 256);
        }
    }
    const std::string path = "/tmp/gpu_blur_test_roundtrip.ppm";
    CHECK(gpu_blur::write_ppm(path, img));
    auto loaded = gpu_blur::read_ppm(path);
    auto cmp = gpu_blur::compare_images(img, loaded, 1e-5f);
    CHECK(cmp.mismatches == 0);
    std::remove(path.c_str());
    printf("[PASS] test_ppm_round_trip\n");
}

}  // namespace

int main() {
    test_gaussian_kernel_normalized_and_symmetric();
    test_radius_zero_is_identity();
    test_constant_image_preserved();
    test_box_blur_impulse_response();
    test_box_blur_corner_clamping();
    test_separable_matches_direct_2d_gaussian();
    test_channels_are_independent();
    test_ppm_round_trip();

    if (g_failures == 0) {
        printf("\nAll 8 CPU reference tests passed.\n");
        return EXIT_SUCCESS;
    }
    fprintf(stderr, "\n%d assertion(s) failed.\n", g_failures);
    return EXIT_FAILURE;
}
