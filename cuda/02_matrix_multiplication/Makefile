INC=-I/usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCC_OPT=-std=c++17

all:
	$(NVCC) $(NVCC_OPT) -x cu 01_matrix_mul.cpp -o 01_matrix_mul.out
	$(NVCC) $(NVCC_OPT) -x cu 02_matrix_mul_shared_mem.cpp -o 02_matrix_mul_shared_mem.out
	$(NVCC) $(NVCC_OPT) -x cu 03_matrix_mul_rows.cpp -o 03_matrix_mul_rows.out
	$(NVCC) $(NVCC_OPT) -x cu 04_matrix_mul_cols.cpp -o 04_matrix_mul_cols.out
clean:
	-rm -f main
