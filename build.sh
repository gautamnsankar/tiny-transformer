#!/usr/bin/env bash
set -e

TORCH_DIR=${TORCH_DIR:-$PWD/libtorch}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.1}
CUDA_LIBDIR=${CUDA_LIBDIR:-$CUDA_HOME/targets/x86_64-linux/lib}
ABI_FLAG=${USE_PRE_CXX11_ABI:+-D_GLIBCXX_USE_CXX11_ABI=0}

g++ index.cpp -o main \
  -std=c++20 -O2 -fPIC $ABI_FLAG \
  -I"$TORCH_DIR/include" \
  -I"$TORCH_DIR/include/torch/csrc/api/include" \
  -L"$TORCH_DIR/lib" -L"$CUDA_LIBDIR" \
  -Wl,--no-as-needed \
  -Wl,-rpath,"$TORCH_DIR/lib" \
  -Wl,-rpath,"$CUDA_LIBDIR:/usr/lib/wsl/lib" \
  -ltorch_cuda -ltorch -ltorch_cpu -lc10_cuda -lc10 -lcudart \
  -pthread -ldl
./main
