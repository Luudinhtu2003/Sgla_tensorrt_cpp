#!/bin/bash
# Build script for Jetson Xavier NX (JetPack 4.6.2, TensorRT 8.2.1)

set -e

mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ""
echo "Build complete! Run with:"
echo "  ./build/sgla_tracker --engine <path/to/model.engine> --video <path/to/video.mp4>"
