from gaptrain.gtconfig import GTConfig
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
        """Are all the atoms in the box? """
        coords = self.coordinates()
        return np.max(coords) < max(self.box.size) and np.min(coords) > 0.0

    def coordinates(self):
        return np.array([atom.coord for atom in self.atoms])

    def wrap(self, max_wraps=100):
        """Wrap all the atoms into the box"""
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
        """Set the coordinates """
        if xyz_filename is not None:
            self.atoms = xyz_file_to_atoms(xyz_filename)

        if atoms is not None:
            self.atoms = atoms

        self.forces = None
        return None

    def run_dftb(self, max_force=None):
        """Run a DFTB+ calculation, either a minimisation or optimisation

        :param max_force: (float) Maximum force in eV Å-1. If None then a
                          single point energy and force evaluation is performed
        """
        from gaptrain.calculators import run_dftb

        os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)
        return run_dftb(self, max_force)

    def run_gap(self, gap, max_force=None):
        """Run GAP to predict energy and forces"""
        from gaptrain.calculators import run_gap
        os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)

        return run_gap(self, max_force=max_force, gap=gap)

    def run_gpaw(self, max_force=None):
        """Run a GPAW DFT calculation, either a minimisation or optimisation

        :param max_force: (float) Maximum force in eV Å-1. If None then a
                          single point energy and force evaluation is performed
        """
        from gaptrain.calculators import run_gpaw

        os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)
        os.environ['MLK_NUM_THREADS'] = str(GTConfig.n_cores)
        return run_gpaw(self, max_force)

    def save(self, filename, append=False):
        """Print this configuration as an extended xyz file

        -----------------------------------------------------------------------
        :param filename: (str)

        :param append: (bool) Append to the end of this exyz file
        """
        a, b, c = self.box.size

        # Energy needs to be formattable
        energy = self.energy if self.energy is not None else 0.0

        if energy == 0.0:
            logger.warning('Printing configuration with no energy')

        with open(filename, 'a' if append else 'w') as exyz_file:
            print(f'{len(self.atoms)}\n'
                  f'Lattice="{a:.6f} 0.000000 0.000000 '
                  f'0.000000 {b:.6f} 0.000000 '
                  f'0.000000 0.000000 {c:.6f}" '
                  f'Properties=species:S:1:pos:R:3:dft_forces:R:3 '
                  f'dft_energy={energy:.8f}',
                  file=exyz_file)

            for i, atom in enumerate(self.atoms):
                x, y, z = atom.coord
                line = f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '

                if self.forces is not None:
                    fx, fy, fz = self.forces[i]
                    line += f'{fx:.5f} {fy:.5f} {fz:.5f}'

                print(line, file=exyz_file)

        return None

    def __init__(self, system=None, box=None, charge=None, mult=None):
        """
        A configuration consisting of a set of atoms suitable to run DFT
        or GAP on to set self.energy and self.forces

        ------------------------------------------------------------------
        :param system: (gaptrain.system.System)

        :param box: (gaptrain.box.Box)

        :param charge: (int)

        :param mult: (int)
        """

        self.atoms = []

        if system is not None:
            for molecule in system.molecules:
                self.atoms += molecule.atoms

        self.forces = None                                  # eV Å-1
        self.energy = None                                  # eV

        self.box = system.box if system is not None else box
        self.charge = system.charge() if system is not None else charge
        self.mult = system.mult() if system is not None else mult

        self.n_wraps = 0


class ConfigurationSet:

    def __len__(self):
        """Get the number of configurations in this set"""
        return len(self._list)

    def __getitem__(self, item):
        """Get an indexed configuration from this set"""
        return self._list[item]

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
            raise ex.CannotAdd('Can only add a Configuration or'
                               f' ConfigurationSet, not {type(other)}')

        return self

    def copy(self):
        return deepcopy(self)

    def load(self, filename=None, system=None,
             box=None, charge=None, mult=None):
        """
        Load a set of configurations from an extended xyz file - needs to have
        a system to be able to assign a charge, multiplicity and box size.
        Will set the *true* values

        ----------------------------------------------------------------------
        :param system: (gaptrain.systems.System)

        :param filename: (str) Filename to load configurations from if
                         None defaults to "name.xyz"

        :param box: (gaptrain.box.Box)

        :param charge: (int)

        :param mult: (int)
        """
        filename = f'{self.name}.xyz' if filename is None else filename

        if not os.path.exists(filename):
            raise ex.LoadingFailed(f'XYZ file for {self.name} did not exist')

        if system is None and not all((box, charge, mult)):
            raise ex.LoadingFailed('Configurations must be loaded with either'
                                   'a system or box, charge & multiplicity')

        lines = open(filename, 'r').readlines()

        # Number of atoms should be the first item in the file
        n_atoms = int(lines[0].split()[0])
        stride = int(n_atoms + 2)

        # Stride through the file and add configuration for each
        for i, _ in enumerate(lines[::stride]):

            # Atoms, true forces and energy
            atoms, forces = [], []
            energy = None

            # Grab the coordinates, energy and forces 0->n_atoms + 2 inclusive
            for j, line in enumerate(lines[i*stride:(i+1)*stride]):

                if j == 0:
                    # First thing should be the number of atoms
                    assert len(line.split()) == 1

                elif j == 1:
                    if 'dft_energy' in line:
                        energy = float(line.split()[-1].lstrip('dft_energy='))

                else:
                    atom_label, x, y, z = line.split()[:4]
                    atoms.append(Atom(atom_label, x=x, y=y, z=z))

                    if len(line.split()) != 7:
                        continue

                    # System has forces
                    fx, fy, fz = line.split()[4:]
                    forces.append(np.array([float(fx), float(fy), float(fz)]))

            # Add the configuration
            configuration = Configuration(system, box, charge, mult)
            configuration.set_atoms(atoms=atoms)

            configuration.energy = energy

            # Set the  forces if there are some
            if len(forces) > 0:
                configuration.forces = np.array(forces)

            configuration.wrap()
            self._list.append(configuration)

        return None

    def save(self, override=True):
        """Save an extended xyz file for this set of configurations"""

        # Ensure the name is unique
        if not override and os.path.exists(f'{self.name}.xyz'):
            n = 0
            while os.path.exists(f'{self.name}{n}.xyz'):
                n += 1

            self.name = f'{self.name}{n}'

        # Add all of the configurations to the extended xyz file
        for config in self._list:
            # Print either the ground truth or predicted values
            config.save(f'{self.name}.xyz', append=True)

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
                    f'Using {GTConfig.n_cores} total cores')

        start_time = time()
        results = []

        with Pool(processes=GTConfig.n_cores) as pool:

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
            logger.warning('No configurations to remove')
            return None

        self._list = np.random.choice(self._list, size=remainder)
        return None

    def __init__(self, *args, name='data'):
        """Set of configurations

        :param args: (gaptrain.configurations.Configuration)
        :param name: (str)
        """

        self.name = name
        self._list = list(args)
