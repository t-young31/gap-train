from gaptrain.box import Box
from gaptrain.log import logger
from gaptrain.configurations import Configuration
import gaptrain.exceptions as ex
from autode.species.species import Species
from copy import deepcopy
import numpy as np


class System:
    """
    A mutable collection of molecules/ions/atoms within a periodic cuboidal box
    species can be added and the coordinates randomised to generate different
    configurations

       ________________
     / |       Zn2+   /|
    /________________/ |
    |  |             | |
    |  |  H2O   H2O  | |
    |  |             | |
    |  |_____________|_|
    | /    H2O       | /
    |________________|/

    must be initialised with at least a box size.

    Example:

    water_box = System(box_size=[10, 10, 10])
    water_box.add_molecules(h2o, n=33)

    to generate a box of water at ~1 g cm-3 density.
    """

    def __len__(self):
        return len(self.molecules)

    def __add__(self, other):
        """Add another list or molecule to the system"""

        if type(other) is list or type(other) is tuple:
            assert all(isinstance(m, Species) for m in other)
            self.molecules += list(other)

        elif isinstance(other, Species):
            self.molecules.append(other)

        elif isinstance(other, System):
            assert other.charge == self.charge
            assert other.mult == self.mult

            self.molecules += other.molecules

        else:
            raise Exception(f'Cannot add {other} to system')

        return self

    def add_perturbation(self, sigma=0.05, max_length=0.2):
        """Add a random perturbation to all atoms in the system

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

        for molecule in self.molecules:
            for atom in molecule.atoms:

                # Generate random vectors until one has length < threshold
                while True:
                    vector = np.random.normal(loc=0.0, scale=sigma, size=3)

                    if np.linalg.norm(vector) < max_length:
                        atom.translate(vector)
                        break

        return None

    def random(self, min_dist_threshold=1.5, with_intra=False, on_grid=False,
               max_attempts=10000, **kwargs):
        """Randomise the configuration

        -----------------------------------------------------------------------
        :param min_dist_threshold: (float) Minimum distance in Å that a
                                   molecule is permitted to be to another atom


        :param with_intra: (bool) Randomise both the inter (i.e. molecules)
                           and also the intramolecular DOFs (i.e. bond
                           lengths and angles)

        :param on_grid: (bool)

        :param max_attempts: (int) Maximum number of times a molecule can be
                             placed randomly without raising RandomiseFailed
        """
        logger.info(f'Randomising all {len(self)} molecules in the box')

        coords = np.empty(shape=(0, 3))
        system = deepcopy(self)

        # Because of periodic boundary conditions the distances to
        # other molecules may be less than expected, so molecules
        # need to be generated in a smaller sub-box
        sub_box = Box(size=self.box.size - min_dist_threshold)

        def random_translate():
            """Randomly translate a molecule either on a grid or not"""
            if on_grid:
                vec = sub_box.random_grid_point(spacing=molecule.radius)
                molecule.translate(vec=vec)
            else:
                molecule.translate(vec=sub_box.random_point())

        for molecule in np.random.permutation(system.molecules):

            # Randomly rotate the molecule around the molecules centroid
            molecule.rotate(axis=np.random.uniform(-1.0, 1.0, size=3),
                            theta=np.random.uniform(0.0, 2*np.pi),
                            origin=molecule.centroid())

            random_translate()

            # Keep track of the number of times that this molecule has been
            # placed randomly in the box
            n_attempts = 0

            # Translate to a random position in the box that is also not too
            # close to any other molecule
            while (not molecule.in_box(sub_box)
                   or molecule.min_distance(coords) < min_dist_threshold):

                # Shift back to the origin then to a random point
                molecule.translate(vec=-molecule.centroid())
                random_translate()
                n_attempts += 1

                if n_attempts > max_attempts:
                    raise ex.RandomiseFailed('Maximum attempts exceeded')

            # Add the coordinates to the full set
            coords = np.vstack((coords, molecule.get_coordinates()))

        logger.info('Randomised all molecules in the system')

        if with_intra:
            system.add_perturbation(**kwargs)

        return system.configuration()

    def add_solvent(self, solvent_name):
        """Add water to the system to generate a ~1 g cm-3 density"""
        # assert solvent in solvent library
        raise NotImplementedError

    def add_molecules(self, molecule, n=1):
        """Add a number of the same molecule to the system"""
        assert isinstance(molecule, Species)
        self.molecules += [deepcopy(molecule) for _ in range(n)]
        return None

    def density(self):
        """Calculate the density of the system"""
        raise NotImplementedError

    def atom_symbols(self):
        """Get all the atom labels/atomic symbols in this system"""
        return [atom.label for m in self.molecules for atom in m.atoms]

    def charge(self):
        """Get the total charge on the system"""
        return sum(molecule.charge for molecule in self.molecules)

    def mult(self):
        """Get the total spin multiplicity on the system"""
        return sum(molecule.mult for molecule in self.molecules)

    def configuration(self):
        return Configuration(self)

    def __init__(self, *args, box_size):
        """
        System containing a set of molecules.

        e.g. pd_1water = (Pd, water, box_size=[10, 10, 10], charge=2)
        for a system containing a Pd(II) ion and one water in a 1 nm^3 box

        ----------------------------------------------------------------------
        :param args: (gaptrain.molecules.Molecule) Molecules that comprise
                     the system

        :param box_size: (list(float)) Dimensions of the box that the molecules
                        occupy. e.g. [10, 10, 10] for a 10 Å cubic box.

        """
        self.molecules = list(args)

        self.box = Box(box_size)

        logger.info(f'Initalised a system\n'
                    f'Number of molecules = {len(self.molecules)}\n'
                    f'Charge              = {self.charge()} e\n'
                    f'Spin multiplicity   = {self.mult()}')


class MMSystem(System):

    def generate_topology(self):
        """Generate a GROMACS topology for this system"""
        assert all(m.itp_filename is not None for m in self.molecules)

        raise NotImplementedError

    def __init__(self, *args, box_size):
        """System that can be simulated with molecular mechanics"""
        super().__init__(*args, box_size=box_size)
