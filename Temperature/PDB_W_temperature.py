import os
# import importlib.util
import sys

# spec = importlib.util.spec_from_file_location('dynaphopy', '/home/markwootton/git/DynaPhoPy/dynaphopy/')
# foo = importlib.util.module_from_spec(spec)
# sys.modules['dynaphopy'] = foo
# spec.loader.exec_module(foo)

# sys.path.insert(-1,'/home/markwootton/git/DynaPhoPy/dynaphopy/')

# os.environ['PYTHONPATH'] += ':$HOME/git/DynaPhoPy'

import math
import numpy as np
# from phonolammps import Phonolammps
from pl import Phonolammps
from dynaphopy.interface.lammps_link import generate_lammps_trajectory
from dynaphopy.interface.phonopy_link import ForceConstants
from dynaphopy import Quasiparticle
from dynaphopy.atoms import Structure

species='Cr'
lammpsF='Cr.in'
temperature=100#K
showplot=True

phlammps = Phonolammps(lammpsF, supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]], show_log=True, show_progress=True)

phlammps.plot_phonon_dispersion_bands(absv=True, tag='Cr')




################################################################################
force_constants = ForceConstants(phlammps.get_force_constants(),supercell=phlammps.get_supercell_matrix())

structure = phlammps.get_unitcell()

# define structure for dynaphopy
dp_structure = Structure(cell=structure.get_cell(),  # cell_matrix, lattice vectors in rows
                         scaled_positions=structure.get_scaled_positions(),
                         atomic_elements=structure.get_chemical_symbols(),
                         primitive_matrix=phlammps.get_primitve_matrix(),
                         force_constants=force_constants)

# calculate trajectory for dynaphopy with lammps
trajectory = generate_lammps_trajectory(dp_structure, lammpsF,
                                        total_time=20,      # ps
                                        time_step=0.001,    # ps
                                        relaxation_time=5,  # ps
                                        silent=False,
                                        supercell=[2, 2, 2],
                                        memmap=False,
                                        velocity_only=True,
                                        temperature=temperature)

# set dynaphopy calculation
calculation = Quasiparticle(trajectory)
calculation.select_power_spectra_algorithm(2)  # select FFT algorithm

calculation.get_renormalized_phonon_dispersion_bands()
renormalized_force_constants = calculation.get_renormalized_force_constants()

# Print phonon band structure
if showplot:
    calculation.plot_renormalized_phonon_dispersion_bands(plot_harmonic=False)

###
# print(calculation.get_renormalized_phonon_dispersion_bands())
bands_full_data = calculation.get_renormalized_phonon_dispersion_bands()
path = '' if len(sys.argv) == 1 else '_'+'_'.join(f'{point}' for point in sys.argv[1:])

with open(os.path.join(os.getcwd(), f'dynaphopy_{species}_{temperature}K{path}.csv'), 'w') as file:
    wv = []
    fr = [[] for i in range(6)]
    for i, path in enumerate(bands_full_data):
        # print(i, path)
        # for p in path:
        #     file.write(p)
        #     for p_ in path[p]:
        #         file.write(f',{p_}')
        #     file.write('\n')

        # print(len(path['q_path_distances']), len(np.array(list(path['renormalized_frequencies'].values())).T))
        for d, f in zip(path['q_path_distances'], np.array(list(path['renormalized_frequencies'].values())).T):
            # print(d,f, len(f), max(f))
            wv.append(d)
            for fn, ff in enumerate(f):
                fr[fn].append(ff)


    # print(max(wv), max(max(fr)))
    file.write(','.join(f'{wv_}' for wv_ in wv) + '\n')
    for fn, ff in enumerate(fr):
        file.write(','.join(f'{ff_}' for ff_ in ff) + '\n')

################################################################################

# Workaround for a problem that should have never existed
lmp = lammps.lammps(cmdargs=['-k','on','g','1','-sf','kk'])
lmp.close()
lmp.finalize()
