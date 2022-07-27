#!/bin/bash

mkdir -p build && rm -rf ./build/*
cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release # Release vs Debug
make -C ./build # cmake --build ./build
./build/bin/main
