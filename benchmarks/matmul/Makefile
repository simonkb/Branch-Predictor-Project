RISCV_PREFIX ?= riscv64-linux-gnu
CXX := $(RISCV_PREFIX)-g++
CXXFLAGS := -O3 -static -march=rv64gc -mabi=lp64d -std=c++17

all: matmul_rv_static

matmul_rv_static: matmul.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f matmul_rv_static
