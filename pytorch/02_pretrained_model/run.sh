#!/bin/bash

mkdir -p checkpoint

# arg is null or 3, execute cmake
if [ -z "$1" ] || [ $1 -eq 3 ]; then
    mkdir -p build && rm -rf ./build/*
    cmake -S . -B ./build
fi

# arg == 2, execute make
if [ -z "$1" ] || [ $1 -ge 2 ]; then
    make -C ./build
fi

# arg == 1, execute output from build
if [ -z "$1" ] || [ $1 -ge 1 ]; then
    ./build/main
fi
