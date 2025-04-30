import lammps

# These are the flags needed to run on GPU with Kokkos
lmp = lammps.lammps(cmdargs=['-k','on','g','1','-sf','kk'])
lmp.close()
lmp.finalize()

# Testing if mpi4py is using the correct backend
from mpi4py import MPI
print("MPI vendor:", MPI.get_vendor())
print("MPI version:", MPI.Get_version())

print("PASSED")
