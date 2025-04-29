### We acknowledge the help of Dr. Beñat Gurrutxaga-Lerma (University of Birmingham) ###

import os
# from phonolammps import Phonolammps
from pl import Phonolammps
import matplotlib.pyplot as plt
import lammps

phlammps = Phonolammps('in.W',
                       supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                       show_log=True, show_progress=True,
                       lammps_args=['-k', 'on', 'g', '1', '-sf', 'kk']
                       )

phlammps.plot_phonon_dispersion_bands(absv=True, tag='W')

# Unfortunately, there's no way to get rid of
# "Kokkos::Cuda ERROR: Failed to call Kokkos::Cuda::finalize()"
