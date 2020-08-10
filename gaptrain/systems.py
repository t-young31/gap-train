from gaptrain.molecules import Molecule
from gaptrain.solvents import solvents
from gaptrain.box import Box
from autode.input_output import atoms_to_xyz_file
from gaptrain.log import logger
from gaptrain.configurations import Configuration
from copy import deepcopy
from scipy.spatial.distance import cdist
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
               **kwargs):
        """Randomise the configuration

        -----------------------------------------------------------------------
        :param min_dist_threshold: (float) Minimum distance in Å that a
                                   molecule is permitted to be to another atom

        :param on_grid: (bool)

        :param with_intra: (bool) Randomise both the inter (i.e. molecules)
                           and also the intramolecular DOFs (i.e. bond
                           lengths and angles)
        """
        logger.info(f'Randomising all {len(self)} molecules in the box')

        coords = np.empty(shape=(0, 3))
        system = deepcopy(self)

        # Because of periodic boundary conditions the distances to
        # other molecules may be less than expected, so molecules
        # need to be generated in a smaller sub-box
        sub_box = Box(size=self.box.size - min_dist_threshold)

        for molecule in np.random.permutation(system.molecules):

            molecule.translate_to_origin()

            # Randomly rotate the molecule
            molecule.rotate(axis=np.random.uniform(-1.0, 1.0, size=3),
                            theta=np.random.uniform(0.0, 2*np.pi))

            # Translate to a random position in the box..
            while (not molecule.in_box(sub_box)
                   or molecule.min_distance(coords) < min_dist_threshold):

                molecule.translate_to_origin()

                if on_grid:
                    vec = sub_box.random_grid_point(spacing=2*molecule.radius)
                    molecule.translate(vec=vec)

                else:
                    molecule.translate(vec=sub_box.random_point())

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
        for a system containing a Pd(II) ion and one water in a 10 Å^3 box

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

    def generate_topology(self, path_to_ff, system_name):
        """Generate a GROMACS topology for this system"""
        for solvent in solvents:  # remove later
            print(solvent.name)   # remove later
        assert all(m.itp_filename is not None for m in solvents)

        with open('topol.top', 'x') as f:
            print(f'; Include force field parameters',
                  f'#include {path_to_ff}', file=f, sep='\n')  # indented too much
            set_itp = set([mol.itp_filename for mol in solvents])

            for itp in set_itp:
                print(f'#include {itp}', file=f)
            print(f'[ system ]\n'
                    f'; Name\n'
                    f'{system_name}\n'
                    f'[ molecules ]\n'
                    f'; Compound' + 10*' ' + '#mols' + '\n', file=f) #  use string formatting

            mol_names = [m.name for m in solvents]
            for mol_name in set(mol_names):
                print(f'{mol_name:<10s}{mol_names.count(mol_name)}', file=f)

    def __init__(self, *args, box_size):
        """System that can be simulated with molecular mechanics"""
        super().__init__(*args, box_size=box_size)
