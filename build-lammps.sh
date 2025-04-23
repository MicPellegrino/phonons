#!/bin/bash

# Brief description
# 
# This is adapted from https://mace-docs.readthedocs.io/en/latest/guide/lammps.html
# 
# 'BUILD_LIB' and 'BUILD_SHARED_LIBS' create LAMMPS plungin 
# (needed to build Python API)
# 
# Make sure the CMake options are compatible with your backend 
# (e.g Kokkos_ARCH_AMPERE should be adapted based on you GPU)
#
# I have added a bunch of packages needed for my research, feel free
# to change everything below PKG_ML-MACE=ON (excluded!) 

mkdir lammps/build-mace
cd lammps/build-mace

cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=ON \
    -D BUILD_LIB=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_CUDA=ON \
    -D CMAKE_CXX_COMPILER=$(pwd)/../lib/kokkos/bin/nvcc_wrapper \
    -D Kokkos_ARCH_AMDAVX=ON \
    -D Kokkos_ARCH_AMPERE86=ON \
    -D CMAKE_PREFIX_PATH=$(pwd)/../../libtorch \
    -D PKG_ML-MACE=ON \
    -D PKG_CLASS=ON \
    -D PKG_EXTRA-COMPUTE=ON \
    -D PKG_EXTRA-DUMP=ON \
    -D PKG_FEP=ON \
    -D PKG_KSPACE=ON \
    -D PKG_MANYBODY=ON \
    -D PKG_MISC=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_RIGID=ON \
    -D PKG_REAXFF=ON \
    -D PKG_MC=ON \
    -D PKG_MEAM=ON \
    -D PKG_OPENMP=ON \
    ../cmake

make -j 56

make install-python
