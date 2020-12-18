import gaptrain as gt
from gaptrain.log import logger
import gaptrain.exceptions as ex
from autode.input_output import xyz_file_to_atoms
from autode.atoms import Atom
from multiprocessing import Pool
from copy import deepcopy
from time import time
import numpy as np
import os


class Configuration:

    def ase_atoms(self):
        """Get ASE atoms from this configuration"""
        from ase import Atoms
        return Atoms(symbols=[atom.label for atom in self.atoms],
                     positions=[atom.coord for atom in self.atoms],
                     pbc=True,
                     cell=self.box.size)

    def all_atoms_in_box(self):
        """Are all the atoms in the box?

        :return: (bool)
        """
        coords = self.coordinates()
        max_xyz = np.max(coords, axis=0)
        # The maximum x, y, z values nee dto be smaller than the box and all >0
        return max(max_xyz - self.box.size) < 0 and np.min(coords) > 0.0

    def add_perturbation(self, sigma=0.05, max_length=0.2):
        """Add a random perturbation to all atoms in the configuration

        ----------------------------------------------------------------------
        :param sigma: (float) Variance of the normal distribution used to
                      generate displacements in Å

        :param max_length: (float) Maximum length of the random displacement
                           vector in Å
        """
        logger.info(f'Displacing all atoms in the system using a random '
                    f'displacments from a normal distribution:\n'
                    f'σ        = {sigma} Å\n'
                    f'max(|v|) = {max_length} Å')

        for atom in self.atoms:

            # Generate random vectors until one has length < threshold
            while True:
                vector = np.random.normal(loc=0.0, scale=sigma, size=3)

                if np.linalg.norm(vector) < max_length:
                    atom.translate(vector)
                    break

        return None

    def coordinates(self):
        """
        Atomic positions for this configuration

        :return: (np.ndarray) matrix of coordinates shape = (n, 3)
        """
        return np.array([atom.coord for atom in self.atoms])

    def copy(self):
        return deepcopy(self)

    def wrap(self, max_wraps=100):
        """
        Wrap all the atoms into the box

        :param max_wraps: (int) Maximum number of recursive calls
        """
        logger.info('Wrapping all atoms back into the box')

        if self.all_atoms_in_box():
            logger.info('All atoms in the box - nothing to be done')
            return None

        for atom in self.atoms:
            for i, _ in enumerate(['x', 'y', 'z']):

                # Atom is not in the upper right octant of 3D space
                if atom.coord[i] < 0:
                    atom.coord[i] += self.box.size[i]

                # Atom is further than the upper right quadrant
                if atom.coord[i] > self.box.size[i]:
                    atom.coord[i] -= self.box.size[i]

        while not self.all_atoms_in_box():
            logger.info('All atoms are still not in the box')
            self.n_wraps += 1

            # Prevent an overflow in the recursive call by setting a threshold
            if self.n_wraps > max_wraps:
                return None

            return self.wrap()

        # Reset the number of wraps performed on this configuration(?)
        return None

    def set_atoms(self, xyz_filename=None, atoms=None):
        """
        Set self.atoms from either an xyz file or a list of atoms

        :param xyz_filename: (str)
        :param atoms: (list(autode.atoms.Atom))
        """
        if xyz_filename is not None:
            self.atoms = xyz_file_to_atoms(xyz_filename)

        if atoms is not None:
            self.atoms = atoms

        self.forces = None
        return None

    def run_dftb(self, max_force=None, n_cores=None):
        """
        Run a DFTB+ calculation, either a minimisation or optimisation

        :param max_force: (float) Maximum force in eV Å-1. If None then a
                          single point energy and force evaluation is performed
        """
        from gaptrain.calculators import run_dftb, set_threads
        set_threads(n_cores)

        return run_dftb(self, max_force)

    def run_gap(self, gap, max_force=None, n_cores=None):
        """Run GAP to predict energy and forces"""
        from gaptrain.calculators import run_gap, set_threads
        set_threads(n_cores)

        return run_gap(self, max_force=max_force, gap=gap)

    def run_gpaw(self, max_force=None, n_cores=None):
        """Run a GPAW DFT calculation, either a minimisation or optimisation

        :param max_force: (float) Maximum force in eV Å-1. If None then a
                          single point energy and force evaluation is performed
        """
        from gaptrain.calculators import run_gpaw, set_threads
        set_threads(n_cores)

        return run_gpaw(self, max_force)

    def run_orca(self, max_force=None, n_cores=None):
        """Run an ORCA calculation on this configuration"""
        from gaptrain.calculators import run_orca, GTConfig
        assert max_force is None

        return run_orca(self, n_cores=n_cores if n_cores is not None
                        else GTConfig.n_cores)

    def print_gro_file(self, system):
        filename = 'input.gro'
        with open(filename, 'w') as f:
            print(f'{str(system)}', file=f)
            print(f'{len(self.atoms)}', file=f)

            n = 0
            for i, molecule in enumerate(system.molecules):
                atom_list = []
                for atom in molecule.atoms:
                    x, y, z = self.atoms[n].coord / 10
                    atom_list.append(atom.label)
                    print(f'{i+1:>5}'
                          f'{molecule.name:<5}'
                          f'{atom.mm_type.strip():>5}'
                          f'{n+1:>5}'
                          f'{x:>8.3f}'
                          f'{y:>8.3f}'
                          f'{z:>8.3f}', file=f)
                    n += 1
            a, b, c = system.box.size / 10
            print(f'{a} {b} {c}', file=f)

        return None

    def load(self, filename=None, file_lines=None, box=None, charge=None,
             mult=None):
        """
        Load a configuration from a file or a list of file lines

        ----------------------------------------------------------------------
        :param filename: (str) Filename to load configurations from if
                         None defaults to "name.xyz"

        :param file_lines: (list(str)) List of extended xyz file lines to read
                           from

        :param box: (gaptrain.box.Box)

        :param charge: (int)

        :param mult: (int)
        """
        if filename is None and file_lines is None:
            try:
                file_lines = open(f'{self.name}.xyz', 'r').readlines()
            except IOError:
                raise ex.LoadingFailed('Could not load no file or file lines')

        if filename is not None and file_lines is None:
            file_lines = open(filename, 'r').readlines()

        self.charge = charge if charge is not None else self.charge
        self.mult = mult if mult is not None else self.mult
        self.box = box if box is not None else self.box

        # Atoms, true forces and energy
        n_atoms, atoms, forces = None, [], []

        # Grab the coordinates, energy and forces 0->n_atoms + 2 inclusive
        for j, line in enumerate(file_lines):

            if j == 0:
                # First thing should be the number of atoms
                try:
                    n_atoms = int(line.split()[0])
                except (IndexError, TypeError):
                    raise ex.LoadingFailed('Line 1 of the xyz file '
                                           'malformatted')

            elif j == 1:
                if 'dft_energy' in line:
                    self.energy = float(line.split()[-1].lstrip('dft_energy='))

                # Try and load the box
                if 'Lattice="' in line and box is None:
                    try:
                        # Remove anything before or after the quotes
                        vec_string = line.split('"')[1].split('"')[0]
                        components = [float(val) for val in vec_string.split()]
                        # Expecting all the components of the lattice
                        # vectors, so for an orthorhombic box take the
                        # diagonal elements of the a, b, c vectors
                        self.box = gt.Box(size=[components[0],
                                                components[4],
                                                components[8]])

                    except (TypeError, ValueError, IndexError):
                        raise ex.LoadingFailed('Failed to load the box')

            elif len(line.split()) < 4:
                logger.warning('Unexpected line break, assuming end of atoms')
                break

            else:
                atom_label, x, y, z = line.split()[:4]
                atoms.append(Atom(atom_label, x=x, y=y, z=z))

                if len(line.split()) != 7:
                    continue

                # System has forces
                fx, fy, fz = line.split()[4:]
                forces.append(np.array([float(fx), float(fy), float(fz)]))

        # Default charge and multiplicity if there is a box but no charge
        if self.box is not None:
            if charge is None and self.charge is None:
                logger.warning('Found a box but no charge, defaulting to 0')
                self.charge = 0
            if mult is None and self.mult is None:
                logger.warning('Found a box but no multiplicity, '
                               'defaulting to 1')
                self.mult = 1

        if len(atoms) == 0 or n_atoms is None:
            raise ex.LoadingFailed('Found no atoms in the file')

        if len(atoms) != n_atoms:
            raise ex.LoadingFailed(f'Number of atoms declared {n_atoms} not '
                                   f'equal to the number found {len(atoms)}')

        self.set_atoms(atoms=atoms)

        # Set the  forces if there are some
        if len(forces) > 0:
            self.forces = np.array(forces)

        return None

    def save(self, filename=None, append=False):
        """
        Print this configuration as an extended xyz file where the first 4
        columns are the atom symbol, x, y, z and, if this configuration
        contains forces then add the x, y, z components of the force on as
        columns 4-7.

        -----------------------------------------------------------------------
        :param filename: (str)

        :param append: (bool) Append to the end of this exyz file?
        """
        if filename is None:
            filename = f'{self.name}.xyz'
            logger.info(f'Saving configuration as {filename}')

        a, b, c = self.box.size

        energy_str = ''
        if self.energy is not None:
            energy_str += f'dft_energy={self.energy:.8f}'

        prop_str = 'Properties=species:S:1:pos:R:3'
        if self.forces is not None:
            prop_str += ':dft_forces:R:3'

        if not filename.endswith('.xyz'):
            logger.warning('Filename had no .xyz extension - adding')
            filename += '.xyz'

        with open(filename, 'a' if append else 'w') as exyz_file:
            print(f'{len(self.atoms)}\n'
                  f'Lattice="{a:.6f} 0.000000 0.000000 '
                  f'0.000000 {b:.6f} 0.000000 '
                  f'0.000000 0.000000 {c:.6f}" '
                  f'{prop_str} '
                  f'{energy_str}',
                  file=exyz_file)

            for i, atom in enumerate(self.atoms):
                x, y, z = atom.coord
                line = f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '

                if self.forces is not None:
                    fx, fy, fz = self.forces[i]
                    line += f'{fx:.5f} {fy:.5f} {fz:.5f}'

                print(line, file=exyz_file)

        return None

    def __init__(self, filename=None, system=None, box=None, charge=None,
                 mult=None, name='config'):
        """
        A configuration consisting of a set of atoms suitable to run DFT
        or GAP on to set self.energy and self.forces

        ------------------------------------------------------------------
        :param system: (gaptrain.system.System)

        :param box: (gaptrain.box.Box)

        :param charge: (int)

        :param mult: (int)
        """

        self.name = name
        self.atoms = []

        if system is not None:
            for molecule in system.molecules:
                self.atoms += molecule.atoms

        self.forces = None                                  # eV Å-1
        self.energy = None                                  # eV

        self.box = system.box if system is not None else box
        self.charge = system.charge if system is not None else charge
        self.mult = system.mult if system is not None else mult

        self.n_wraps = 0

        if filename is not None:

            self.load(filename)


class ConfigurationSet:

    def __len__(self):
        """Get the number of configurations in this set"""
        return len(self._list)

    def __getitem__(self, item):
        """Get an indexed configuration from this set"""
        return self._list[item]

    def __setitem__(self, key, value):
        """Set a new indexed configuration"""
        self._list[key] = value
        return None

    def __iter__(self):
        """Iterate through these configurations"""
        return iter(self._list)

    def __add__(self, other):
        """Add another configuration or set of configurations onto this one"""

        if isinstance(other, Configuration):
            self._list.append(other)

        elif isinstance(other, ConfigurationSet):
            self._list += other._list

        else:
            raise TypeError('Can only add a Configuration or'
                            f' ConfigurationSet, not {type(other)}')

        logger.info(f'Current number of configurations is {len(self)}')
        return self

    def add(self, other):
        """Add another configuration to this set of configurations"""
        return self.__add__(other)

    def copy(self):
        return deepcopy(self)

    def load(self, filename=None, system=None,
             box=None, charge=None, mult=None):
        """
        Load a set of configurations from an extended xyz file - needs to have
        a system to be able to assign a charge, multiplicity and box size.
        Will set the *true* values

        ----------------------------------------------------------------------
        :param system: (gaptrain.systems.System |
                        gaptrain.configuration.Configuration)

        :param filename: (str) Filename to load configurations from if
                         None defaults to "name.xyz"

        :param box: (gaptrain.box.Box)

        :param charge: (int)

        :param mult: (int)
        """
        filename = f'{self.name}.xyz' if filename is None else filename

        if not os.path.exists(filename):
            raise ex.LoadingFailed(f'XYZ file for {filename} did not exist')

        if system is not None:
            if all(prm for prm in (system.box, system.charge, system.mult)):
                logger.info('Setting box, charge and multiplicity from a conf')
                box, charge, mult = system.box, system.charge, system.mult

        logger.info(f'Loading configuration set from {filename}')
        lines = open(filename, 'r').readlines()

        # Stride through the file and add configuration for each
        i = 0
        while i < len(lines):

            # Configurations may have different numbers of atoms
            try:
                n_atoms = int(lines[i].split()[0])
            except (TypeError, IndexError):
                raise ex.LoadingFailed('Could not read the number of atoms in'
                                       f'{filename}')

            stride = n_atoms + 2

            configuration = Configuration()
            configuration.load(file_lines=lines[i:i+stride],
                               box=box, charge=charge, mult=mult)

            self._list.append(configuration)
            i += stride

        if self.name is None or self.name == 'data':
            self.name = filename.rstrip('.xyz')
            logger.warning(f'Set self.name to {self.name}')

        return None

    def save(self, filename=None, override=True):
        """Save an extended xyz file for this set of configurations"""

        # Ensure the name is unique
        if filename is None:
            filename = f'{self.name}.xyz'

            if not override and os.path.exists(f'{self.name}.xyz'):
                n = 0
                while os.path.exists(f'{self.name}{n}.xyz'):
                    n += 1
                filename = f'{self.name}{n}.xyz'

            if override:  # Empty the file
                open(f'{self.name}.xyz', 'w').close()

        # Add all of the configurations to the extended xyz file
        for config in self._list:
            # Print either the ground truth or predicted values
            config.save(filename, append=True)

        return None

    def _run_parallel_method(self, method, max_force, **kwargs):
        """Run a set of electronic structure calculations on this set
        in parallel

        :param method: (function) A method to calculate energy and forces
                       on a configuration

        :param max_force: (float) Maximum force on an atom within a
                          configuration. If None then only a single point
                          energy evaluation is performed
        """
        logger.info(f'Running calculations over {len(self)} configurations\n'
                    f'Using {gt.GTConfig.n_cores} total cores')
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MLK_NUM_THREADS'] = '1'

        start_time = time()
        results = []

        with Pool(processes=gt.GTConfig.n_cores) as pool:

            # Apply the method to each configuration in this set
            for i, config in enumerate(self._list):
                result = pool.apply_async(func=method,
                                          args=(config, max_force),
                                          kwds=kwargs)
                results.append(result)

            # Reset all the configurations in this set with updated energy
            # and forces (each with .true)
            for i, result in enumerate(results):
                self._list[i] = result.get(timeout=None)

        logger.info(f'Calculations done in {(time() - start_time)/60:.1f} m')
        return None

    def parallel_gpaw(self, max_force=None):
        """Run single point or optimisation up to a F threshold using GPAW"""
        from gaptrain.calculators import run_gpaw
        return self._run_parallel_method(run_gpaw, max_force=max_force)

    def parallel_gap(self, gap, max_force=None):
        """Run single point or optimisation up to a F threshold using a GAP"""
        from gaptrain.calculators import run_gap
        return self._run_parallel_method(run_gap, max_force=max_force,
                                         gap=gap)

    def parallel_dftb(self, max_force=None):
        """Run periodic DFTB+ on these configurations"""
        from gaptrain.calculators import run_dftb
        return self._run_parallel_method(run_dftb, max_force=max_force)

    def parallel_orca(self):
        """Run parallel ORCA on these configurations"""
        from gaptrain.calculators import run_orca
        return self._run_parallel_method(run_orca, max_force=None, n_cores=1)

    def remove_first(self, n):
        """
        Remove the first n configurations

        :param n: (int)
        """
        self._list = self._list[n:]
        return None

    def remove_random(self, n=None, remainder=None):
        """Randomly remove some configurations

        :param n: (int) Number of configurations to remove
        :param remainder: (int) Number of configurations left in these data
        """
        # Number to choose is the total minus the number to remove
        if n is not None:
            remainder = len(self) - int(n)

        elif remainder is not None:
            remainder = int(remainder)

        else:
            raise ValueError('No configurations to remove')

        self._list = list(np.random.choice(self._list, size=remainder))
        return None

    def remove_above_e(self, threshold, min_energy=None):
        """
        Remove configurations above an energy threshold

        :param threshold: (float) Energy threshold in eV
        :param min_energy: (float or None) Minimum/reference energy to use to
                           calculate the relative energy of a configuration. If
                           None then the lowest energy in this set is used
        """
        if any(config.energy is None for config in self._list):
            raise ValueError('Cannot truncate: a config had energy=None')

        if min_energy is None:
            min_energy = min(config.energy for config in self._list)

        logger.info(f'Truncating {len(self._list)} configurations with an '
                    f'energy threshold of {threshold:.3f} eV above a reference'
                    f' energy of {min_energy:.3f} eV')

        self._list = [config for config in self._list
                      if (config.energy - min_energy) < threshold]

        logger.info(f'Truncated on energies to {len(self._list)}')
        return None

    def remove_above_f(self, threshold):
        """
        Remove configurations with an atomic force component above a threshold
        value

        :param threshold: (float) Force threshold in eV Å-1
        """
        if any(config.forces is None for config in self._list):
            raise ValueError('Cannot truncate: a config had forces=None')

        logger.info(f'Truncating {len(self._list)} configurations with a '
                    f'force threshold of {threshold:.3f} eV Å-1')

        self._list = [config for config in self._list
                      if np.max(config.forces) < threshold]

        logger.info(f'Truncated on forces to {len(self._list)}')
        return None

    def remove_highest_e(self, n):
        """
        Remove the highest energy n configurations from this set

        :param n: (int) Number of configurations to remove
        """
        energies = [config.energy for config in self._list]

        if any(energy is None for energy in energies):
            raise ValueError('Cannot remove highest energy from a set '
                             'with some undefined energies')

        if n > len(self._list):
            raise ValueError('The number of configurations needs to be larger '
                             'than the number removed')

        logger.info(f'Removing the least stable {n} configurations')

        idxs = np.argsort(energies)
        print(idxs)
        print(idxs[-1])
        self._list = [self._list[i] for i in idxs[:-n]]
        return None

    def truncate(self, n, method='random', **kwargs):
        """
        Truncate this set of configurations to a n configurations

        :param n: (int) Number of configurations to truncate to
        :param method: (str) Name of the method to use
        :param kwargs: ensemble (gaptrain.gap.GAPEnsemble)
        """
        implemented_methods = ['random', 'cur', 'ensemble', 'higher', 'cur_k']

        if method.lower() not in implemented_methods:
            raise NotImplementedError(f'Methods are {implemented_methods}')

        if n > len(self._list):
            raise ValueError(f'Cannot truncate a set of {len(self)} '
                             f'configurations to {n}')

        if method.lower() == 'random':
            return self.remove_random(remainder=n)

        if 'cur' in method.lower():
            if method.lower() == 'cur':
                matrix = gt.descriptors.soap(self)
            elif method.lower() == 'cur_k':
                matrix = gt.descriptors.soap_kernel_matrix(self)
            else:
                raise NotImplementedError

            cur_idxs = gt.cur.rows(matrix, k=n, return_indexes=True)
            self._list = [self._list[idx] for idx in cur_idxs]

        if method.lower() == 'ensemble':
            logger.info(f'Truncating {len(self)} configurations based on the'
                        f' largest errors using a ensemble of GAP models')

            if 'ensemble' not in kwargs:
                raise KeyError('A GAPEnsemble must be provided to use'
                               'ensemble truncation i.e.: '
                               'truncate(.., ensemble=ensemble_instance)')

            errors = kwargs['ensemble'].predict_energy_error(self._list)

            # Use the n configurations with the largest error, where numpy
            # argsort sorts from smallest->largest so take the last n values
            self._list = [self._list[i] for i in np.argsort(errors)[-n:]]

        if method.lower() == 'higher':
            return self.remove_highest_e(n=len(self) - n)

        return None

    def __init__(self, *args, name='data'):
        """Set of configurations

        :param args: (gaptrain.configurations.Configuration)
        :param name: (str)
        """

        self.name = name
        self._list = []

        for arg in args:
            if isinstance(arg, Configuration):
                self._list.append(arg)

            elif type(arg) is str and arg.endswith('.xyz'):
                self.load(filename=arg)

            else:
                raise AssertionError('Unsupported argument type')
