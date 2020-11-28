from gaptrain.molecules import Species
from gaptrain.solvents import get_solvent
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

    def __str__(self):
        """Name of this system using the number of molecules contained in it"""
        if len(self.molecules) == 0:
            return 'system'

        system_str = ''
        mol_names = [mol.name for mol in self.molecules]

        for name in set(mol_names):
            num = mol_names.count(name)
            system_str += f'{name}_{num if num > 0 else ""}_'

        # Remove the final underscore
        return system_str.rstrip('_')

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

        for molecule in system.molecules:

            # Randomly rotate the molecule around the molecules centroid
            molecule.translate(vec=-molecule.centroid())
            molecule.rotate(axis=np.random.uniform(-1.0, 1.0, size=3),
                            theta=np.random.uniform(0.0, 2*np.pi))

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
        config = system.configuration()

        if with_intra:
            config.add_perturbation(**kwargs)

        return config

    def add_solvent(self, solvent_name, n):
        """Add water to the system to generate a ~1 g cm-3 density

        :param solvent_name: (str) e.g. 'h2o'
        :param n: (int) number of solvent molecules e.g. 10
        """
        solvent = get_solvent(solvent_name)

        return self.add_molecules(molecule=solvent, n=n)

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
        n_unpaired = sum((mol.mult - 1) / 2 for mol in self.molecules)
        return 2 * n_unpaired + 1

    def configuration(self):
        return Configuration(system=self)

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
        for molecule in self.molecules:
            molecule.set_mm_atom_types()

        assert all(m.itp_filename is not None for m in self.molecules)

        def print_types(file, atoms=False, molecules=False):

            itp_names = [mol.itp_filename for mol in self.molecules]
            for itp in sorted(set(itp_names), key=itp_names.index):

                itp_file = open(itp, 'r')
                print_flag = False

                for line in itp_file:
                    if atoms:
                        if 'moleculetype' not in line:
                            print(f'{line}', file=file)
                        else:
                            break

                    elif molecules:
                        if print_flag or 'moleculetype' in line:
                            print(f'{line}', file=file)
                            print_flag = True

        with open('topol.top', 'w') as topol_file:
            print(f'[ defaults ]',
                  f'{"; nbfunc":<16}{"comb-rule":<16}'
                  f'{"gen-pairs":<16}{"fudgeLJ":<8}{"fudgeQQ":<8}',
                  f'{"1":<16}{"2":<16}{"yes":<16}{"0.5":<8}{"0.8333":<8}\n'
                  , file=topol_file, sep='\n')

            print_types(topol_file, atoms=True)
            print_types(topol_file, molecules=True)

            print(f'\n[ system ]',
                  f'; Name',
                  f'{str(self)}\n',
                  f'[ molecules ]',
                  f'; Compound{"":<7s}#mols', file=topol_file, sep='\n')

            mol_names = [m.name for m in self.molecules]
            for mol_name in sorted(set(mol_names), key=mol_names.index):
                print(f'{mol_name:<15s}{mol_names.count(mol_name)}', file=topol_file)

    def __init__(self, *args, box_size):
        """System that can be simulated with molecular mechanics"""
        super().__init__(*args, box_size=box_size)
