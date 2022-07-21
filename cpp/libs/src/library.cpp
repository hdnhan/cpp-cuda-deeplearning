#include "library.h"

template <typename T>
void print_matrix(T **matrix, int64_t rows, int64_t columns) {
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < columns; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

template void print_matrix<float>(float**, int64_t, int64_t);
