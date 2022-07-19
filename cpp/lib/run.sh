#!/bin/bash

g++ -std=c++17 -c -o library.o library.cpp -fPIC 
g++ -shared -o liblibrary.so library.o

cp library.h ../include/

readelf --dyn-syms liblibrary.so