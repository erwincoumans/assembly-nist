#!/usr/bin/env bash
# Build the AssembleNIST SDF plugin (bolt/nut thread SDFs and connector grid SDFs) against the headers and
# library shipped with the `mujoco` Python package (x86-64; -mf16c for the half-precision SDF grids).
# Output: plugin/build/libassemblenist_sdf.so, and a copy in plugin/build/mujoco-<version>/ (sim/threads.py loads the
# copy for the running MuJoCo version when there is one, so environments with different MuJoCo versions can coexist)
#   bash plugin/build.sh                 # uses `python` on PATH
#   PYTHON=/path/to/python bash plugin/build.sh
#   MUJOCO_DIR=/path/to/site-packages/mujoco bash plugin/build.sh
set -euo pipefail
cd "$(dirname "$0")"
MJ=${MUJOCO_DIR:-$(${PYTHON:-python} -c "import mujoco, os; print(os.path.dirname(mujoco.__file__))")}
LIB=$(ls "$MJ"/libmujoco.so* 2>/dev/null | head -1)
[ -n "$LIB" ] || { echo "libmujoco not found in $MJ"; exit 1; }
mkdir -p build
${CXX:-g++} -O3 -std=c++17 -mf16c -shared -fPIC -I"$MJ/include" assemblenist_sdf.cc -o build/libassemblenist_sdf.so "$LIB" \
  -Wl,-rpath,"$MJ"
VER=$(basename "$LIB" | sed -e 's/^libmujoco\.so\.//')
mkdir -p "build/mujoco-$VER" && cp build/libassemblenist_sdf.so "build/mujoco-$VER/"
echo "built plugin/build/libassemblenist_sdf.so (and build/mujoco-$VER/) against $LIB"
