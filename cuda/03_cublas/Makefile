INC=-I/usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCC_OPT=-std=c++17

all:
	$(NVCC) $(NVCC_OPT) -x cu 01_array_add.cpp -o 01_array_add.out -lcublas
	$(NVCC) $(NVCC_OPT) -x cu 02_matrix_mul.cpp -o 02_matrix_mul.out -lcublas -lcurand
clean:
	-rm -f main
