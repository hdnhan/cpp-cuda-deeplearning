#!/bin/bash

make clean
make all
export LD_LIBRARY_PATH=libs/lib:$LD_LIBRARY_PATH
./bin/main
