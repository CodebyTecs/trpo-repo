CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
LDLIBS ?= -lopenblas -pthread
OPENBLAS_PREFIX ?=

ifneq ($(OPENBLAS_PREFIX),)
CXXFLAGS += -I$(OPENBLAS_PREFIX)/include
LDLIBS += -L$(OPENBLAS_PREFIX)/lib
endif

BUILD_DIR := .build

.PHONY: all build test perf ci clean

all: build

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

build: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) bench.cpp my_gemm.cpp $(LDLIBS) -o $(BUILD_DIR)/bench

test: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. tests/interface_tests.cpp my_gemm.cpp $(LDLIBS) -o $(BUILD_DIR)/interface_tests
	./$(BUILD_DIR)/interface_tests

perf: $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I. tests/performance_tests.cpp my_gemm.cpp $(LDLIBS) -o $(BUILD_DIR)/performance_tests
	./$(BUILD_DIR)/performance_tests

ci: build test perf

clean:
	rm -rf $(BUILD_DIR)
