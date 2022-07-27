#!/bin/bash

make clean
make release # debug vs release
export LD_LIBRARY_PATH=lib/lib:$LD_LIBRARY_PATH
./bin/main
