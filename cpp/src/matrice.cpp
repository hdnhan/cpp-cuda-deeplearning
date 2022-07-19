#include "matrice.h"

Matrix::Matrix(int64_t rows, int64_t cols) : m_rows(rows), m_cols(cols) {
    std::cout << "Constructor executed!" << std::endl;
    m_data = (float **)malloc(m_rows * sizeof(float *));
    for (int64_t i = 0; i < m_rows; i++) {
        m_data[i] = (float *)malloc(m_cols * sizeof(float));
    }
}

/*
malloc -> free
new    -> delete
new[]  -> delete[]
*/
Matrix::~Matrix() {
    std::cout << "Destructor executed!" << std::endl;
    for (int64_t i = 0; i < m_rows; i++) {
        free(m_data[i]);
    }
    free(m_data);
}

void Matrix::randomize() {
    srand((int64_t)time(NULL));
    for (int64_t i = 0; i < m_rows; i++) {
        for (int64_t j = 0; j < m_cols; j++) {
            m_data[i][j] = (float)rand() / (float)RAND_MAX;
        }
    }
}

void Matrix::print() {
    for (int64_t i = 0; i < m_rows; i++) {
        for (int64_t j = 0; j < m_cols; j++) {
            std::cout << m_data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}