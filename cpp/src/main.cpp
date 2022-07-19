#include <cstdint>
#include <iomanip>  // setprecision
#include <iostream>

#include "library.h"
#include "matrice.h"

int main(int argc, char *argv[]) {
    // std::cout.precision(4);
    std::cout << std::setprecision(4) << std::fixed;
    int64_t rows = 10, cols = 10;
    Matrix m(rows, cols);
    m.randomize();
    m.print();
    std::cout << std::endl;
    print_matrix(m.m_data, rows, cols);
    return 0;
}
