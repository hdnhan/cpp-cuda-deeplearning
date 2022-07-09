cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/../libtorch")

project(resnet)

find_package(Torch REQUIRED)

set(TEST_NAME inference)
add_executable(${TEST_NAME})
target_sources(${TEST_NAME} PRIVATE inference.cpp
                                     src/cifar10.cpp
                                     src/transform.cpp
                                     src/resnet.cpp
)
target_include_directories(${TEST_NAME} PRIVATE include)
target_link_libraries(${TEST_NAME} ${TORCH_LIBRARIES})
set_property(TARGET ${TEST_NAME} PROPERTY CXX_STANDARD 17)


set(TRAIN_NAME train)
add_executable(${TRAIN_NAME})
target_sources(${TRAIN_NAME} PRIVATE train.cpp
                                     src/cifar10.cpp
                                     src/transform.cpp
                                     src/resnet.cpp
)
target_include_directories(${TRAIN_NAME} PRIVATE include)
target_link_libraries(${TRAIN_NAME} ${TORCH_LIBRARIES})
set_property(TARGET ${TRAIN_NAME} PROPERTY CXX_STANDARD 17)
