cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuBLAS LANGUAGES C CXX CUDA)

message("CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message("CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message("CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(include)
link_libraries(cublas)

add_executable(01-axpy src/01-axpy.cu src/common.cu)
add_executable(01-dot src/01-dot.cu src/common.cu)
add_executable(02-gemv src/02-gemv.cu src/common.cu)
add_executable(03-gemm src/03-gemm.cu src/common.cu)