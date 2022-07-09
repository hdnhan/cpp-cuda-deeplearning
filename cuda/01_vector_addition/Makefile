INC=-I/usr/local/cuda/include
NVCC=/usr/local/cuda/bin/nvcc
NVCC_OPT=-std=c++17

all:
	$(NVCC) $(NVCC_OPT) -x cu 01_vector_add.cpp -o 01_vector_add.out
	$(NVCC) $(NVCC_OPT) -x cu 02_vector_add_unified_memory.cpp -o 02_vector_add_unified_memory.out
	$(NVCC) $(NVCC_OPT) -x cu 03_vector_add_unified_memory_prefetch.cpp -o 03_vector_add_unified_memory_prefetch.out
clean:
	-rm -f main
