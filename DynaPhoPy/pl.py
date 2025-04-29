__version__ = '0.9.0'

import numpy as np
import warnings

from phonopy.file_IO import write_FORCE_CONSTANTS, write_force_constants_to_hdf5, write_FORCE_SETS
from phonolammps.arrange import get_correct_arrangement, rebuild_connectivity_tinker
from phonolammps.phonopy_link import obtain_phonon_dispersion_bands, get_phonon
from phonolammps.iofile import get_structure_from_poscar, generate_VASP_structure
from phonolammps.iofile import generate_tinker_key_file, generate_tinker_txyz_file, parse_tinker_forces
from phonolammps.iofile import get_structure_from_lammps, get_structure_from_txyz
from phonolammps.iofile import get_structure_from_g96, get_structure_from_gro
from phonolammps.iofile import generate_gro, generate_g96

from lammps import lammps

import shutil, os

import time

# define the force unit conversion factors to LAMMPS metal style (eV/Angstrom)
unit_factors = {'real': 4.336410389526464e-2,
                'metal': 1.0,
                'si': 624150636.3094,
                'gromacs': 0.00103642723,
                'tinker': 0.043  # kcal/mol to eV
}


#lmp = lammps(cmdargs=["-l", "~/Clean_MACE_LAMMPS/lammps/build/liblammps.so"])

class PhonoBase:
    """
    Base class for PhonoLAMMPS
    This class is not designed to be called directly.
    To use it make a subclass and implement the following methods:

    * __init__()
    * get_forces()

    """


    def get_path_using_seek_path(self, customPath=None):

        """ Obtain the path in reciprocal space to plot the phonon band structure

        :return: dictionary with list of q-points and labels of high symmetry points
        """

        try:
            if customPath is not None:
                raise ImportError
            import seekpath

            cell = self._structure.get_cell()
            positions = self._structure.get_scaled_positions()
            numbers = np.unique(self._structure.get_chemical_symbols(), return_inverse=True)[1]

            path_data = seekpath.get_path((cell, positions, numbers))

            labels = path_data['point_coords']

            band_ranges = []
            for set in path_data['path']:
                band_ranges.append([labels[set[0]], labels[set[1]]])
            print(band_ranges)

            return {'ranges': band_ranges,
                    'labels': path_data['path']}
        except ImportError:
            if customPath is None:
                print ('Seekpath not installed. Autopath is deactivated')
            else:
                print(f'Using custom path {customPath}')
            band_ranges = ([customPath]) #([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]])
            return {'ranges': band_ranges,
                    # 'labels': [['GAMMA', '1/2 0 1/2']]
                    'labels': [[' '.join(f'{m}' for m in customPath[0]),
                                ' '.join(f'{m}' for m in customPath[-1])]]
                   }



    def get_force_constants(self, include_data_set=False):
        """
        calculate the force constants with phonopy using lammps to calculate forces

        :return: ForceConstants type object containing force constants
        """

        if self._force_constants is None:
            phonon = get_phonon(self._structure,
                                setup_forces=False,
                                super_cell_phonon=self._supercell_matrix,
                                primitive_matrix=self._primitive_matrix,
                                NAC=self._NAC,
                                symmetrize=self._symmetrize)

            phonon.get_displacement_dataset()
            phonon.generate_displacements(distance=self._displacement_distance)
            cells_with_disp = phonon.get_supercells_with_displacements()
            data_set = phonon.get_displacement_dataset()

            # Check forces for non displaced supercell
            forces_supercell = self.get_forces(phonon.get_supercell())
            if np.max(forces_supercell) > 1e-1:
                warnings.warn('Large atomic forces found for non displaced structure: '
                              '{}. Make sure your unit cell is properly optimized'.format(np.max(forces_supercell)))

            # Get forces from lammps
            for i, cell in enumerate(cells_with_disp):
                if self._show_progress:
                    print('displacement {} / {}'.format(i+1, len(cells_with_disp)))
                forces = self.get_forces(cell)
                data_set['first_atoms'][i]['forces'] = forces

            phonon.set_displacement_dataset(data_set)
            phonon.produce_force_constants()
            self._force_constants = phonon.get_force_constants()
            self._data_set = data_set

        if include_data_set:
            return [self._force_constants, self._data_set]
        else:
            return self._force_constants



    def plot_phonon_dispersion_bands(self, absv=False, tag='', writeFile=True, wave_vector_labels=True):
        """
        Plot phonon band structure using seekpath automatic k-path
        Warning: The labels may be wrong if the structure is not standarized

        :param absv: Toggle for forcing absolute value of frequency
        :param tag: Tag for file output

        """
        tag = '_' + tag
        colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k']*3
        import matplotlib.pyplot as plt
        import math

        def replace_list(text_string):
            substitutions = {'GAMMA': u'$\Gamma$',
                             }

            for item in substitutions.items():
                text_string = text_string.replace(item[0], item[1])
            return text_string

        force_constants = self.get_force_constants()
        bands_and_labels = self.get_path_using_seek_path()#customPath=[[0,0,0],[0.5,0.5,0.5]])

        _bands = obtain_phonon_dispersion_bands(self._structure,
                                                bands_and_labels['ranges'],
                                                force_constants,
                                                self._supercell_matrix,
                                                primitive_matrix=self._primitive_matrix,
                                                band_resolution=100000)

        curves = []
        for i, freq in enumerate(_bands[1]):
            # print(f'Wave vector {i}\n', _bands[1][i], len(_bands[1][i]))
            # print(f'Frequency {i} [THz]')
            # for b2 in range(len(_bands[2][i])):
                # print(b2, _bands[2][i][b2], len(_bands[2][i][b2]))
                # for j in range(len(_bands[2][i][b2])):
                #     pb[j].append(_bands[2][i][b2][j])
                # break
            # print()
            # plt.plot(_bands[1][i], _bands[2][i], color='r')
            curveSet = plt.plot(_bands[1][i], [abs(_bands[2][i][j]) for j in range(len(_bands[2][i]))] if absv else _bands[2][i], color='black', linewidth=0.5)#, color=colour[i])
            for curve in curveSet:
                curves.append(curve)

        plt.ylabel('Frequency [THz]')
        plt.xlabel('Wave vector')
        plt.xlim([0, _bands[1][-1][-1]])
        plt.ylim([0, math.ceil(np.max(np.abs(_bands[2])))])
        plt.axhline(y=0, color='k', ls='dashed')
        plt.suptitle('Phonon dispersion')

        if 'labels' in bands_and_labels and wave_vector_labels:
            plt.rcParams.update({'mathtext.default': 'regular'})

            labels = bands_and_labels['labels']

            labels_e = []
            x_labels = []
            for i, freq in enumerate(_bands[1]):
                if labels[i][0] == labels[i - 1][1]:
                    labels_e.append(replace_list(labels[i][0]))
                else:
                    labels_e.append(
                        replace_list(labels[i - 1][1]) + '/' + replace_list(labels[i][0]))
                x_labels.append(_bands[1][i][0])
            x_labels.append(_bands[1][-1][-1])
            labels_e.append(replace_list(labels[-1][1]))
            labels_e[0] = replace_list(labels[0][0])

            plt.xticks(x_labels, labels_e, rotation='horizontal')

        if writeFile:
            # print(len(_bands[1]))
            wv = []
            # pb = [[]]*len(_bands[2][0])
            for i in range(len(_bands[1])):
                for wv_ in _bands[1][i]:
                    wv.append(wv_)
                # for j, pb_ in enumerate(_bands[2][i]):
                #     for pb_v in pb_:
                #         # print(len(pb_v))
                #         pb[j].append(pb_v)
                #     # print(pb)
                #     # exit()

            # print(len(wv))
            # for pb_ in pb:
                # print(len(pb_))
            nBands = 0
            for i, curve in enumerate(curves):
                if curve.get_xdata()[0]:
                    nBands = i
                    break
            start = -1.
            # section = 0
            wv_data = []
            # pb_data = [[]]*nBands
            pb_data = []
            for q in range(nBands):
                pb_data.append([])
            # print(pb_data)
            # print(len(curves))
            with open(os.path.join(os.getcwd(), f'phonon_dispersion_bands{tag}.csv'), 'w') as file:
                for i, curve in enumerate(curves):
                    if not nBands:
                        break
                    ii = i%nBands
                    # print(curve)
                    # file.write(','.join(f'{wv_}' for wv_ in curve.get_xdata()) + '\n')
                    # file.write(','.join(f'{pb_d}' for pb_d in curve.get_ydata()) + '\n')
                    # file.write('\n')

                    wv = list(curve.get_xdata())
                    # print(wv, type(wv))
                    pb = list(curve.get_ydata())
                    assert len(wv) == len(pb)
                    # print(i, ii, pb[0], len(pb))
                    # exit()

                    if wv[0] > start:
                    #     print(section, start, wv[0])
                    #     section += 1
                        start = wv[0]
                        wv_data += wv[(0 if i < nBands else 1):]
                    #     pb_data.append([])
                    # pb_data[-1] += pb

                    pb_data[ii] = pb_data[ii]+pb[(0 if i < nBands else 1):]
                    # print(pb_data)
                    # exit()
                    # print(i, ii, len(pb), len(pb_data), len(pb_data[ii]))
                    # pb = None
                    # assert len(wv) == len(pb_data[i%nBands])
                    # for pb_ in pb:
                    #     pb_data[i%nBands].append(pb_)
                    # print(pb_data[i%nBands], '\n')
                    # print(i%nBands)
                    # time.sleep(1)
                    # if ii == 17:
                    #     print()


                # print('wv:', wv_data)
                # print('pb:', pb_data, len(pb_data))
                file.write(','.join(f'{wv_}' for wv_ in wv_data) + '\n')
                for pb_ in pb_data:
                    file.write(','.join(f'{pb_d}' for pb_d in pb_) + '\n')
                # for n in range(nBands):
                #     for n_ in pb_data[n]:
                #         file.write(f'{n_},')
                #     file.write('\n')





                # file.write(','.join(f'{wv_}' for wv_ in wv) + '\n')
                # for pb_ in pb:
                #     file.write(','.join(f'{pb_d}' for pb_d in pb_) + '\n')


                # for i, freq in enumerate(_bands[1]):
                #     for bi in _bands[1][i]:
                #         file.write(f'{bi},')
                # file.write('\n')
                #     #####
                #     # file.write(_bands[1][i])
                #     # file.write('\n')
                #     # data = [[]]*len(freq)
                #     # for k in range(len(freq)):
                #     #     for j in range(len(_bands[2][i])):
                #     #         print(i,j,k,_bands[2][i][j][k])
                #     #         print(len(_bands[1]), len(_bands[2][i]), len(freq))
                #     #         data[k].append(_bands[2][i][j][k])
                #     #         # file.write(f'{bij}')
                #     #         # file.write(',')
                #     #     # print(data)
                #     #     # break
                #     # print(data)
                #     # for dk in data:
                #     #     for bij in dk:
                #     #         file.write(f'{bij}')
                #     #         file.write(',')
                #     #     file.write('\n')
                #     #####
                #     # data = _bands[2][i].transpose()
                #     # print(data)
                #     # for d in data:
                #     #     for dd in d:
                #     #         file.write(f'{dd},')
                #     #     file.write(f'\n')
                #     # file.write('\n')
                #     ####
                # for i, freq in enumerate(_bands[1]):
                #     data = [abs(_bands[2][i][j]) for j in range(len(_bands[2][i]))] if absv else _bands[2][i]
                #     data2 = [[]]*len(data[0])
                #     for d in data:
                #         for dd in range(len(d)):
                #             data2[dd].append(d[dd])
                #
                #     # print(len(data))
                #     for d in data2:
                #         print(len(d))
                #         for dd in d:
                #             file.write(f'{dd},')
                #             file.write('\n')

        
        plt.savefig("W_dispersion.png", dpi=300, bbox_inches='tight')
        plt.show()


    def write_force_constants(self, filename='FORCE_CONSTANTS', hdf5=False):
        """
        Write the force constants in a file in phonopy plain text format

        :param filename: Force constants filename
        """

        force_constants = self.get_force_constants()
        if hdf5:
            write_force_constants_to_hdf5(force_constants, filename=filename)
        else:
            write_FORCE_CONSTANTS(force_constants, filename=filename)



    def write_force_sets(self, filename='FORCE_SETS'):
        """
        Write the force sets in a file in phonopy plain text format

        :param filename: Force sets filename
        """

        data_set = self.get_force_constants(include_data_set=True)[1]

        write_FORCE_SETS(data_set, filename=filename)



    def get_unitcell(self):
        """
        Get unit cell structure

        :return unitcell: unit cell 3x3 matrix (lattice vectors in rows)
        """
        return self._structure



    def get_supercell_matrix(self):
        """
        Get the supercell matrix

        :return supercell: the supercell 3x3 matrix (list of lists)
        """
        return self._supercell_matrix


    def get_primitve_matrix(self):
        return self._primitive_matrix

    def get_seekpath_bands(self, band_resolution=30):
        ranges = self.get_path_using_seek_path()['ranges']
        bands =[]
        for q_start, q_end in ranges:
            band = []
            for i in range(band_resolution+1):
                band.append(np.array(q_start) + (np.array(q_end) - np.array(q_start)) / band_resolution * i)
            bands.append(band)

        return bands


    def write_unitcell_POSCAR(self, filename='POSCAR'):
        """
        Write unit cell in VASP POSCAR type file

        :param filename: POSCAR file name (Default: POSCAR)
        """
        poscar_txt = generate_VASP_structure(self._structure)

        with open(filename, mode='w') as f:
            f.write(poscar_txt)



    def get_phonopy_phonon(self):
        """
        Return phonopy phonon object with unitcell, primitive cell and
        the force constants set.

        :return:
        """

        phonon = get_phonon(self._structure,
                            setup_forces=False,
                            super_cell_phonon=self._supercell_matrix,
                            primitive_matrix=self._primitive_matrix,
                            NAC=self._NAC,
                            symmetrize=self._symmetrize)

        phonon.set_force_constants(self.get_force_constants())

        return phonon


################################
#            LAMMPS            #
################################

class Phonolammps(PhonoBase):
    def __init__(self,
                 lammps_input,
                 supercell_matrix=np.identity(3),
                 primitive_matrix=np.identity(3),
                 displacement_distance=0.01,
                 show_log=False,
                 show_progress=False,
                 use_NAC=False,
                 symmetrize=True,
                 lammps_args=[]):
        """
        Main PhonoLAMMPS class

        :param lammps_input: LAMMPS input file name or list of commands
        :param supercell_matrix:  3x3 matrix supercell
        :param primitive cell:  3x3 matrix primitive cell
        :param displacement_distance: displacement distance in Angstroms
        :param show_log: Set true to display lammps log info
        :param show_progress: Set true to display progress of calculation
        """

        # Check if input is file or list of commands
        if type(lammps_input) is str:
            # read from file name
            self._lammps_input_file = lammps_input
            self._lammps_commands_list = open(lammps_input).read().split('\n')
        else:
            # read from commands
            self._lammps_commands_list = lammps_input

        self._structure = get_structure_from_lammps(self._lammps_commands_list, show_log=show_log)

        self._supercell_matrix = supercell_matrix
        self._primitive_matrix = primitive_matrix
        self._displacement_distance = displacement_distance
        self._show_log = show_log
        self._show_progress = show_progress
        self._symmetrize = symmetrize
        self._NAC = use_NAC

        self._force_constants = None
        self._data_set = None

        self.units = self.get_units(self._lammps_commands_list)

        if not self.units in unit_factors.keys():
            print ('Units style not supported, use: {}'.format(unit_factors.keys()))
            exit()

        # Additional imput flags for LAMMPS
        self._lammps_args = lammps_args


    def get_units(self, commands_list):
        """
        Get the units label for LAMMPS "units" command from a list of LAMMPS input commands

        :param commands_list: list of LAMMPS input commands (strings)
        :return units: string containing the units
        """
        for line in commands_list:
                if line.startswith('units'):
                    return line.split()[1]
        return 'lj'



    def get_forces(self, cell_with_disp):
        """
        Calculate the forces of a supercell using lammps

        :param cell_with_disp: supercell from which determine the forces
        :return: numpy array matrix with forces of atoms [Natoms x 3]
        """

        import lammps

        supercell_sizes = np.diag(self._supercell_matrix)

        cmd_list = ['-log', 'none']
        # HACK: adding Kokkoks-on-GPU flags
        # cmd_list += ['-k', 'on', 'g', '1', '-sf', 'kk']
        if not self._show_log:
            cmd_list += ['-echo', 'none', '-screen', 'none']

        # Adding other flags (e.g. to run on GPU, or with OpenMP)
        cmd_list += self._lammps_args

        lmp = lammps.lammps(cmdargs=cmd_list)
        lmp.commands_list(self._lammps_commands_list)
        lmp.command('replicate {} {} {}'.format(*supercell_sizes))
        lmp.command('run 0')

        na = lmp.get_natoms()
        # xc = lmp.gather_atoms("x", 1, 3)
        # reference2 = np.array([xc[i] for i in range(na * 3)]).reshape((na, 3))

        id = lmp.extract_atom("id", 0)
        id = np.array([id[i]-1 for i in range(na)], dtype=int)
        # id_inverse = [list(id).index(i) for i in range(len(id))]

        xp = lmp.extract_atom("x", 3)
        reference = np.array([[xp[i][0], xp[i][1], xp[i][2]] for i in range(na)], dtype=float)

        template = get_correct_arrangement(reference, self._structure, self._supercell_matrix)
        indexing = np.argsort(template)

        coordinates = cell_with_disp.get_positions()

        for i in range(na):
            lmp.command('set atom {} x {} y {} z {}'.format(id[i]+1,
                                                            coordinates[template[i], 0],
                                                            coordinates[template[i], 1],
                                                            coordinates[template[i], 2])
                        )

        lmp.command('run 0')

        # forces2 = lmp.gather_atoms("f", 1, 3)
        # forces2 = np.array([forces2[i] for i in range(na * 3)], dtype=float).reshape((na, 3))#[indexing,:]

        fp = lmp.extract_atom("f", 3)
        forces = np.array([[fp[i][0], fp[i][1], fp[i][2]] for i in range(na)], dtype=float)

        forces = forces[indexing, :] * unit_factors[self.units]

        # elements = ['C', 'H', 'H', 'H', 'C', 'O']
        # id = lmp.extract_atom("id", 0)
        # id = np.array([id[i] for i in range(na)], dtype=int)

        # xp = lmp.extract_atom("x", 3)
        # coordinates = np.array([[xp[i][0], xp[i][1], xp[i][2]] for i in range(na)], dtype=float)
        # print(coordinates)

        # types = lmp.extract_atom("type", 0)
        # types = np.array([types[i]-1 for i in range(na)], dtype=int)
        # symbols = [elements[i] for i in types]

        # print('{}\n'.format(len(symbols)))
        # for i, s in enumerate(symbols):
        #     print(s, '{:10.5f} {:10.5f} {:10.5f}'.format(*coordinates[i]) + '  {} {}'.format(id[i], i+1))

        lmp.close()
        # Avoid Kokkos+CUDA error?
        # lmp.finalize()

        return forces



    def optimize_unitcell(self, energy_tol=0, force_tol=1e-10, max_iter=1000000, max_eval=1000000):
        """
        Optimize atoms position of the unitcell using lammps minimizer.
        Check https://docs.lammps.org/minimize.html for details

        :param energy_tol: stopping tolerance for energ
        :param force_tol: stopping tolerance for force (force units)
        :param max_iter: max iterations of minimizer
        :param max_eval: max number of force/energy evaluations
        """

        import lammps

        cmd_list =  ['-log', 'none']
        # HACK: adding Kokkoks-on-GPU flags
        # cmd_list += ['-k', 'on', 'g', '1', '-sf', 'kk']
        if not self._show_log:
            cmd_list += ['-echo', 'none', '-screen', 'none']

        cmd_list += self._lammps_args

        lmp = lammps.lammps(cmdargs=cmd_list)
        lmp.commands_list(self._lammps_commands_list)

        lmp.command(' thermo          10 ')
        lmp.command('thermo_style    custom step temp etotal press vol enthalpy')
        #lmp.command(' fix             2  all box/relax aniso 1000000 dilate all')
        lmp.command('minimize {} {} {} {} '.format(energy_tol, force_tol, max_iter, max_eval))
        #lmp.command('run 0 ')

        na = lmp.get_natoms()
        xp = lmp.extract_atom("x", 3)
        positions = np.array([[xp[i][0], xp[i][1], xp[i][2]] for i in range(na)], dtype=float)

        fp = lmp.extract_atom("f", 3)
        forces = np.array([[fp[i][0], fp[i][1], fp[i][2]] for i in range(na)], dtype=float)

        lmp.close()
        # Avoid Kokkos+CUDA error?
        # lmp.finalize()

        self._structure.set_positions(positions)

        norm = np.linalg.norm(forces.flatten())
        maxforce = np.max(np.abs(forces))

        print('Force two-norm: ', norm)
        print('Force max component: ', maxforce)

        return norm, maxforce