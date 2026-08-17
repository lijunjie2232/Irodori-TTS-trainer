#!/usr/bin/env zsh

# Load user environment/aliases (like cu121)
if [ -f ~/.zsh ]; then
    source ~/.zsh
fi

if [ -f ~/.profile ]; then
    source ~/.profile
fi

# Switch CUDA environment
cu121

# Run CMake Configuration
CC=gcc-11 CXX=g++-11 cmake -B build_linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61-real \
  -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA_GRAPHS=ON \
  -DUSE_CUDA_GRAPH=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DGGML_CUDA_USE_GRAPHS=ON \
  -DGGML_CUDA_USE_CUB=ON \
  -DGGML_CCACHE=ON

# Build
cmake --build build_linux -j10