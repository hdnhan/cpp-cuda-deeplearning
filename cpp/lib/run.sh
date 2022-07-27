#!/bin/bash

mkdir -p lib && rm -rf lib/*

# g++ -std=c++17 -Iinclude -c -o library.o src/library.cpp -fPIC 
# g++ -shared -o lib/liblibrary.so library.o

# debug: -DDEBUG -O0 -g
# release: -DRELEASE -O3 -s / -DNDEBUG -O3 -s
g++ -std=c++17 -Iinclude -shared -o lib/liblibrary.so src/library.cpp -fPIC -DRELEASE -O3 -s

readelf --dyn-syms lib/liblibrary.so
