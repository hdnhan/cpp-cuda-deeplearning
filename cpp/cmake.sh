#!/bin/bash

mkdir -p build && rm -rf ./build/*
cmake -S . -B ./build
make -C ./build
./build/main
