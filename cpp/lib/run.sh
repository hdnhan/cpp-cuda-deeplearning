#!/bin/bash

mkdir -p lib && rm -rf lib/*

# g++ -std=c++17 -Iinclude -c -o library.o src/library.cpp -fPIC 
# g++ -shared -o lib/liblibrary.so library.o

g++ -std=c++17 -Iinclude -shared -o lib/liblibrary.so src/library.cpp -fPIC

readelf --dyn-syms lib/liblibrary.so
