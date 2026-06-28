# Lensboy environment setup

## Environment name

lb_detect

## Purpose

This environment is used for Pipeline 1 and Pipeline 2 detection.

Pipeline 1:
Lensboy detection -> Calibcam calibration

Pipeline 2:
Lensboy detection -> Lensboy calibration -> Calibcam reprojection check

## Python version

Python 3.11

## Important build tools

Lensboy has a C++ extension, so these tools were needed:

- cmake
- ninja
- vcpkg
- clang++
- libomp

## Important environment variables used during installation

export VCPKG_ROOT="$HOME/vcpkg"
export CMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

export LIBOMP_PREFIX="$(brew --prefix libomp)"

export CPATH="$LIBOMP_PREFIX/include:$CPATH"
export LIBRARY_PATH="$LIBOMP_PREFIX/lib:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$LIBOMP_PREFIX/lib:$DYLD_LIBRARY_PATH"

export CMAKE_ARGS="-DOpenMP_CXX_FLAGS='-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include' -DOpenMP_CXX_LIB_NAMES=omp -DOpenMP_omp_LIBRARY=${LIBOMP_PREFIX}/lib/libomp.dylib"

## Successful test

python - <<'PY'
import cv2
import lensboy as lb

print("Lensboy environment OK")
print("OpenCV:", cv2.__version__)
print("Lensboy:", lb.__file__)
PY
