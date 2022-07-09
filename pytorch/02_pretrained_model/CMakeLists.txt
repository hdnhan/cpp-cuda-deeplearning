cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../libtorch")

project(pretrained_model)

find_package(Torch REQUIRED)

set(EXECUTABLE_NAME main)
add_executable(${EXECUTABLE_NAME} inference.cpp)
target_link_libraries(${EXECUTABLE_NAME} ${TORCH_LIBRARIES})
set_property(TARGET ${EXECUTABLE_NAME} PROPERTY CXX_STANDARD 17)
