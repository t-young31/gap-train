from gaptrain.ef import Energy, Forces
from gaptrain.exceptions import NoEnergy
from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
from autode.input_output import atoms_to_xyz_file
from autode.input_output import xyz_file_to_atoms
from multiprocessing import Pool
from time import time
import os


class Configuration:

    def ase_atoms(self):
        """Get ASE atoms from this configuration"""
        from ase import Atoms
        return Atoms(symbols=[atom.label for atom in self.atoms],
                     positions=[atom.coord for atom in self.atoms],
                     pbc=True,
                     cell=self.box.size)

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

    def print(self, exyz_file, true_values):
        """Print this configuration to a extended xyz file"""

        a, b, c = self.box.size
        energy = self.energy.true if true_values else self.energy.predicted

        if energy is None:
            raise NoEnergy

        print(f'Lattice="{a:.6f} 0.000000 0.000000 0.000000 {b:.6f} 0.000000 '
              f'0.000000 0.000000 {c:.6f}" Properties=species:S:1:pos:R:3:'
              f'dft_forces:R:3 dft_energy={energy:.6f}',
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

    def __iter__(self):
        """Iterate through these configurations"""
        return iter(self._list)

    def __add__(self, other):
        """Add another configuration or set of configurations onto this one"""

        if isinstance(other, Configuration):
            self._list.append(other)

        if isinstance(other, ConfigurationSet):
            self._list += other._list

        return self

    def _save(self, true_values):
        """Save an extended xyz file for this set of configurations"""

        # Ensure the name is unique
        if os.path.exists(f'{self.name}.exyz'):
            n = 0
            while os.path.exists(f'{self.name}{n}.exyz'):
                n += 1

            self.name = f'{self.name}{n}'

        # Add all of the configurations to the extended xyz file
        with open(f'{self.name}.exyz', 'w') as exyz_file:
            for config in self._list:
                # Print either the ground truth or predicted values
                config.print(exyz_file, true_values=true_values)

        return None

    def save_true(self):
        return self._save(true_values=True)

    def save_predicted(self):
        return self._save(true_values=False)

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
            for config in self._list:
                result = pool.apply_async(func=method,
                                          args=(config, 1))
                results.append(result)

            for result in results:
                result.get(timeout=None)

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
