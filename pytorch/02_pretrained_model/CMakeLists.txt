cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(PretrainedModel LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# In case there are multiple cuda versions installed
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../libtorch")
find_package(Torch REQUIRED)
link_libraries(${TORCH_LIBRARIES})

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(EXECUTABLE_NAME main)
add_executable(${EXECUTABLE_NAME} inference.cpp)
