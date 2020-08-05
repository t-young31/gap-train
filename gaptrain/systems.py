from gaptrain.molecules import Molecule
from gaptrain.box import Box
from autode.input_output import atoms_to_xyz_file
from gaptrain.log import logger
from copy import deepcopy
from scipy.spatial.distance import cdist
import numpy as np


class System:

    def __len__(self):
        return len(self.molecules)

    def __add__(self, other):
        """Add another list or molecule to the system"""

        if type(other) is list or type(other) is tuple:
            assert all(isinstance(m, Molecule) for m in other)
            self.molecules += list(other)

        elif isinstance(other, Molecule):
            self.molecules.append(other)

        elif isinstance(other, System):
            assert other.charge == self.charge
            assert other.mult == self.mult

            self.molecules += other.molecules

        else:
            raise Exception(f'Cannot add {other} to system')

        return self

    def print_xyz_file(self, filename):
        """Print a standard .xyz file of this configuration"""

        all_atoms = []
        for molecule in self.molecules:
            all_atoms += molecule.atoms

        atoms_to_xyz_file(all_atoms, filename=filename)
        return None

    def randomise(self, min_dist_threshold=1.5):
        """Randomise the configuration

        :param min_dist_threshold: (float) Minimum distance in Å that a
                                   molecule is permitted to be to another atom
        """
        logger.info(f'Randomising all {len(self)} molecules in the box')

        coords = np.empty(shape=(0, 3))

        for molecule in np.random.permutation(self.molecules):

            # Shift the molecule so the centroid is at the origin
            centroid = np.average(molecule.get_coordinates(), axis=0)
            molecule.translate(vec=-centroid)

            # Randomly rotate the molecule
            molecule.rotate(axis=np.random.uniform(-1.0, 1.0, size=3),
                            theta=np.random.uniform(0.0, 2*np.pi))

            # Translate to a random position in the box..
            while True:
                point = self.box.random_point()

                # Can always add the first molecule
                if len(coords) == 0:
                    molecule.translate(point)
                    break

                # Ensure that this point is far enough away from the other
                # atoms in the system
                distances = cdist(coords, point.reshape(1, 3))
                if np.min(distances) < min_dist_threshold:
                    continue

                # Calculate the minimum distance from this molecule to the rest
                molecule.translate(vec=point)
                distances = cdist(coords, molecule.get_coordinates())

                # If the minimum is larger than the threshold then this
                # molecule can be translated to this point
                if np.min(distances) > min_dist_threshold:
                    break

                # Can't be added - translate back to the origin
                molecule.translate(vec=-point)

            # Add the coordinates to the full set
            coords = np.vstack((coords, molecule.get_coordinates()))

        logger.info('Randomised all molecules in the system')
        return None

    def add_water(self):
        """Add water to the system to generate a ~1 g cm-3 density"""
        raise NotImplementedError

    def add_molecules(self, molecule, n=1):
        """Add a number of the same molecule to the system"""
        self.molecules += [deepcopy(molecule) for _ in range(n)]
        return None

    def density(self):
        """Calculate the density of the system"""
        raise NotImplementedError

    def __init__(self, *args, box_size, charge, spin_multiplicity=1):
        """
        System containing a set of molecules.

        e.g. pd_1water = (Pd, water, box_size=[10, 10, 10], charge=2)
        for a system containing a Pd(II) ion and one water in a 10 Å^3 box

        ----------------------------------------------------------------------
        :param args: (gaptrain.molecules.Molecule) Molecules that comprise
                     the system

        :param box_size: (list(float)) Dimensions of the box that the molecules
                        occupy. e.g. [10, 10, 10] for a 10 Å cubic box.

        :param charge: (int) Total Charge on the system e.g. 0 for a water box
                       or 2 for a Pd(II)(aq) system

        :param spin_multiplicity: (int) Spin multiplicity on the whole system
                                  2S + 1 where S is the number of unpaired
                                  electrons
        """
        self.molecules = list(args)

        self.box = Box(box_size)
        self.charge = int(charge)
        self.mult = int(spin_multiplicity)

        logger.info(f'Initalised a system\n'
                    f'Number of molecules = {len(self.molecules)}\n'
                    f'Charge              = {self.charge} e\n'
                    f'Spin multiplicity   = {self.mult}')


class MMSystem(System):

    def generate_topology(self):
        """Generate a GROMACS topology for this system"""
        assert all(m.itp_filename is not None for m in self.molecules)

        raise NotImplementedError

    def __init__(self, *args, box_size, charge, spin_multiplicity=1):
        """System that can be simulated with molecular mechanics"""

        super().__init__(*args,
                         box_size=box_size,
                         charge=charge,
                         spin_multiplicity=spin_multiplicity)
