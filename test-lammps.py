import lammps

# These are the flags needed to run on GPU with Kokkos
lmp = lammps.lammps(cmdargs=['-k','on','g','1','-sf','kk'])

lmp.close()
lmp.finalize()

print("PASSED")
