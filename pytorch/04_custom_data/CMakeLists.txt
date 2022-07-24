cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(CustomData LANGUAGES CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# In case there are multiple cuda versions installed
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../libtorch")
find_package(Torch REQUIRED)
link_libraries(${TORCH_LIBRARIES})

find_package(OpenCV REQUIRED)
link_libraries(${OpenCV_LIBRARIES})

include_directories(include)

# By default put binaries in build/bin (pre-install)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(TRAIN_NAME train)
add_executable(${TRAIN_NAME} src/train.cpp
                             src/network.cpp
                             src/data.cpp
                             src/common.cpp
)

set(TEST_NAME inference)
add_executable(${TEST_NAME} src/inference.cpp
                            src/network.cpp
                            src/data.cpp
                            src/common.cpp
)
