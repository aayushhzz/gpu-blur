CXX ?= clang++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude
NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++17 -Iinclude

BUILD_DIR := build

.PHONY: all cpu test bench cuda clean

all: cpu

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/blur_cpu.o: src/blur_cpu.cpp include/blur_cpu.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/blur_cpu.cpp -o $@

$(BUILD_DIR)/test_blur_cpu: tests/test_blur_cpu.cpp $(BUILD_DIR)/blur_cpu.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_blur_cpu.cpp $(BUILD_DIR)/blur_cpu.o -o $@

$(BUILD_DIR)/blur_cpu_cli: src/main_cpu.cpp $(BUILD_DIR)/blur_cpu.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/main_cpu.cpp $(BUILD_DIR)/blur_cpu.o -o $@

cpu: $(BUILD_DIR)/test_blur_cpu $(BUILD_DIR)/blur_cpu_cli

test: $(BUILD_DIR)/test_blur_cpu
	./$(BUILD_DIR)/test_blur_cpu

bench: $(BUILD_DIR)/blur_cpu_cli
	./$(BUILD_DIR)/blur_cpu_cli 512 512 5 2.0

cuda: | $(BUILD_DIR)
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "Error: $(NVCC) not found. Install the CUDA Toolkit (Linux/Colab GPU) to build CUDA targets."; \
		exit 1; \
	}
	$(NVCC) $(NVCCFLAGS) step1_vecadd.cu -o $(BUILD_DIR)/step1_vecadd
	$(NVCC) $(NVCCFLAGS) src/blur_cuda.cu src/blur_cpu.cpp -o $(BUILD_DIR)/blur_cuda

clean:
	rm -rf $(BUILD_DIR)
