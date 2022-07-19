#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

class Matrix {
   private:
    int64_t m_rows;
    int64_t m_cols;

   public:
    float** m_data;

   public:
    Matrix(int64_t rows, int64_t cols);
    ~Matrix();
    void randomize();
    void print();
};
