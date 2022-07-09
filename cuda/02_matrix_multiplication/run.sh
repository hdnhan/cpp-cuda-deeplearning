#!/bin/bash

make

echo
echo 01_matrix_mul
# sudo nvprof ./01_matrix_mul.out
./01_matrix_mul.out

echo
echo 02_matrix_mul_shared_mem
# sudo nvprof ./02_matrix_mul_shared_mem.out
./02_matrix_mul_shared_mem.out

echo
echo 03_matrix_mul_rows
# sudo nvprof ./03_matrix_mul_rows.out
./03_matrix_mul_rows.out

echo
echo 04_matrix_mul_cols
# sudo nvprof ./04_matrix_mul_cols.out
./04_matrix_mul_cols.out
