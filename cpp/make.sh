#!/bin/bash

make clean
make all
export LD_LIBRARY_PATH=lib/:$LD_LIBRARY_PATH
./bin/main
