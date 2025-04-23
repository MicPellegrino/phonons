Have you ever dreamed about computing phonons using Neural Network potentials? Me neither, but here we are.

## Installation of the basic components

Prerequisites:
- MPI (tested for `(Open MPI) 4.1.2`)
- CUDA (tested for `CUDA Version: 12.6`)

Building all the software needed for this project is quite tricky. Let's go step by step:

1. After cloning/downloading, create a conda/mamba environment by running `conda env create -f environment.yml` and activate it by running `conda activate phonon`.
2. Download LibTorch to store locally from [https://pytorch.org/](https://pytorch.org/). It could look something like: `wget https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.6.0%2Bcu126.zip` .
3. Clone/download MACE-compatible LAMMPS from: [https://github.com/ACEsuit/lammps](https://github.com/ACEsuit/lammps).
4. Build LAMMPS by running `build-lammps.sh`. Have a look at the file but do **not** change the CMake options! Feel free to reduce the number of processes of `make` (the `-j` flag).
5. Check that the LAMMPS Python API is working by running `python3 test-lammps.py`.

You should be good to go!

## Extras

### MACE

[...]
