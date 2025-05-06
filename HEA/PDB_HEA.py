import os
# from phonolammps import Phonolammps
from pl import Phonolammps
import matplotlib.pyplot as plt
import lammps

phlammps = Phonolammps('in.HEA',
                       supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
                       show_log=True, show_progress=True,
                       lammps_args=['-k', 'on', 'g', '1', '-sf', 'kk']
                       )

phlammps.plot_phonon_dispersion_bands(absv=True, tag='MoWAlTiCr')
