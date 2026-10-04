CXX      := clang++
CXXFLAGS := -std=c++20 -O3 -mcpu=apple-m3 -g
OMP_HOME := $(shell brew --prefix libomp)
OMPFLAGS := -Xpreprocessor -fopenmp -I$(OMP_HOME)/include -L$(OMP_HOME)/lib -lomp
MFRAMEWORKS := -framework Metal -framework Foundation -framework Accelerate

METAL_SDK := xcrun -sdk macosx

all: default.metallib thermal

kernels.air: kernels.metal
	$(METAL_SDK) metal -c $< -o $@

default.metallib: kernels.air
	$(METAL_SDK) metallib $< -o $@


thermal: thermal.cpp solver.cpp multigrid.cpp metal_backend.mm \
         solver.hpp multigrid.hpp metal_backend.hpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -x objective-c++ \
	    thermal.cpp solver.cpp multigrid.cpp metal_backend.mm \
	    $(MFRAMEWORKS) -o $@

clean:
	rm -f *.air *.metallib metal_test thermal
