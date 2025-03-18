#!/bin/bash

# export PATH="/usr/local/cuda-12/bin:$PATH"
# export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH"

export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$HOME/.local/cuda-12/lib64:$LD_LIBRARY_PATH

nvcc -cudart static test.cu -o test ; ./test
# nvcc test.cu -o test ; ./test
# nvcc -cudart shared test.cu -o test ; ./test


