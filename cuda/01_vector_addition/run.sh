#!/bin/bash

make

echo
echo 01_vector_add
# sudo nvprof ./01_vector_add.out
./01_vector_add.out

echo
echo 02_vector_add_unified_memory
# sudo nvprof ./02_vector_add_unified_memory.out
./02_vector_add_unified_memory.out

echo
echo 03_vector_add_unified_memory_prefetch
# sudo nvprof ./03_vector_add_unified_memory_prefetch.out
./03_vector_add_unified_memory_prefetch.out
