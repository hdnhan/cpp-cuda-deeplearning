#!/bin/bash

mkdir -p build && rm -rf ./build/*
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release # Release vs Debug
make -C ./build

echo -e "\n01-axpy:"
./build/bin/01-axpy

echo -e "\n01-dot:"
./build/bin/01-dot

echo -e "\n02-gemv:"
./build/bin/02-gemv

echo -e "\n03-gemm:"
./build/bin/03-gemm