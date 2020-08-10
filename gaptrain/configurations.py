from gaptrain.ef import Energy, Forces
from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
import gaptrain.exceptions as ex
from autode.input_output import atoms_to_xyz_file
from autode.input_output import xyz_file_to_atoms
from autode.atoms import Atom
from multiprocessing import Pool
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

    def coordinates(self):
        return np.array([atom.coord for atom in self.atoms])

    def set_atoms(self, xyz_filename=None, atoms=None):
        """Set the coordinates """
        if xyz_filename is not None:
            self.atoms = xyz_file_to_atoms(xyz_filename)

        if atoms is not None:
            self.atoms = atoms

        # Reset the forces to None
        self.forces = Forces(n_atoms=len(self.atoms))
        return None

    def run_dftb(self):
        from gaptrain.calculators import run_dftb
        return run_dftb(self, n_cores=GTConfig.n_cores)

    def run_gap(self):
        raise NotImplementedError

    def run_gpaw(self):
        raise NotImplementedError

    def print_xyz_file(self, filename):
        """Print a standard .xyz file of this configuration"""
        return atoms_to_xyz_file(self.atoms, filename=filename)

    def print_gro_file(self, filename, system):
        assert filename.endswith('.gro')
        with open(filename, 'w') as f:
            print(f'{str(system)}', file=f)
            print(f'{len(self.atoms)}', file=f)
            n = 0
            for i, molecule in enumerate(system.molecules):
                for atom in molecule.atoms:
                    white_space = 5-len(molecule.name)
                    print(f'{"":<5}{i+1}{molecule.name}{"":{white_space}}'
                          f'{atom.mm_atom_type}'                    # atom type (5 characters)
                          f'{atom.label}'
                          f'{n+1}',                             # atom number (5 positions, integer)
                          f'{self.atoms[n].coord}', file=f)     # position (in nm, x y z in 3 columns, each 8 positions with 3 decimal places)
                    n += 1

    def print(self, exyz_file, true_values):
        """Print this configuration to a extended xyz file"""

        a, b, c = self.box.size
        energy = self.energy.true if true_values else self.energy.predicted

        if energy is None:
            raise ex.NoEnergy

        print(f'{len(self.atoms)}\n'
              f'Lattice="{a:.6f} 0.000000 0.000000 0.000000 {b:.6f} 0.000000 '
              f'0.000000 0.000000 {c:.6f}" Properties=species:S:1:pos:R:3:'
              f'dft_forces:R:3 dft_energy={energy:.8f}',
              file=exyz_file)

        # Print the coordinates and the forces
        forces = self.forces.true() if true_values else self.forces.predicted()

        for i, atom in enumerate(self.atoms):
            x, y, z = atom.coord
            fx, fy, fz = forces[i]
            print(f'{atom.label} {x:.5f} {y:.5f} {z:.5f} '
                  f'{fx:.5f} {fy:.5f} {fz:.5f}', file=exyz_file)

        return None

    def __init__(self, system):
        """
        A configuration consisting of a set of atoms suitable to run DFT
        or GAP on to set self.energy and self.forces

        :param system: (gaptrain.system.System)
        """

        self.atoms = []
        for molecule in system.molecules:
            self.atoms += molecule.atoms

        self.forces = Forces(n_atoms=len(self.atoms))
        self.energy = Energy()

        self.box = system.box
        self.charge = system.charge()
        self.mult = system.mult()


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

    def load(self, system, filename=None):
        """
        Load a set of configurations from an extended xyz file - needs to have
        a system to be able to assign a charge, multiplicity and box size.
        Will set the *true* values

        :param system: (gaptrain.systems.System)

        :param filename: (str) Filename to load configurations from if
                         None defaults to "name.xyz"
        """
        filename = f'{self.name}.xyz' if filename is None else filename

        if not os.path.exists(filename):
            raise ex.LoadingFailed(f'XYZ file for {self.name} did not exist')

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
            configuration = Configuration(system)
            configuration.set_atoms(atoms=atoms)

            configuration.energy.true = energy
            configuration.forces.set_true(forces=np.array(forces))

            self._list.append(configuration)

        return None

    def _save(self, true_values, override=True):
        """Save an extended xyz file for this set of configurations"""

        # Ensure the name is unique
        if not override and os.path.exists(f'{self.name}.xyz'):
            n = 0
            while os.path.exists(f'{self.name}{n}.xyz'):
                n += 1

            self.name = f'{self.name}{n}'

        # Add all of the configurations to the extended xyz file
        with open(f'{self.name}.xyz', 'w') as exyz_file:
            for config in self._list:
                # Print either the ground truth or predicted values
                config.print(exyz_file, true_values=true_values)

        return None

    def save_true(self, override=True):
        return self._save(true_values=True, override=override)

    def save_predicted(self, override=True):
        return self._save(true_values=False, override=override)

    def _run_parallel_est(self, method):
        """Run a set of electronic structure calculations on this set
        in parallel
        """
        logger.info(f'Running calculations over {len(self)} configurations\n'
                    f'Using {GTConfig.n_cores} total cores')

        start_time = time()
        results = []

        with Pool(processes=GTConfig.n_cores) as pool:
            # Apply the method to each configuration in this set
            for i, config in enumerate(self._list):
                result = pool.apply_async(func=method,
                                          args=(config, 1))

                # Reset all the configurations in this set with updated energy
                # and forces (each with .true)
                self._list[i] = result.get(timeout=None)

        logger.info(f'Calculations done in {(time() - start_time)/60:.1f} m')
        return None

    def async_gpaw(self):
        raise NotImplementedError

    def async_gap(self):
        raise NotImplementedError

    def async_dftb(self):
        """Run periodic DFTB+ on these configurations"""
        from gaptrain.calculators import run_dftb
        return self._run_parallel_est(method=run_dftb)

    def __init__(self, *args, name='configs'):
        """Set of configurations

        :param args: (gaptrain.configurations.Configuration)
        :param name: (str)
        """

        self.name = name
        self._list = list(args)
