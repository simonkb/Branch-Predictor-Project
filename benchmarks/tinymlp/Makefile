RISCV_PREFIX ?= riscv64-linux-gnu
CXX := $(RISCV_PREFIX)-g++
CXXFLAGS := -O3 -static -march=rv64gc -mabi=lp64d -std=c++17

all: tinymlp_rv_static

tinymlp_rv_static: tinymlp.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f tinymlp_rv_static

