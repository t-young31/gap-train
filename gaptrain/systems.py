from gaptrain.molecules import Species, UniqueMolecule
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

    def random(self,
               min_dist_threshold=1.7,
               with_intra=False,
               on_grid=False,
               max_attempts=10000,
               **kwargs):
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
            coords = np.vstack((coords, molecule.coordinates))

        logger.info('Randomised all molecules in the system')
        config = system.configuration()

        if with_intra:
            config.add_perturbation(**kwargs)

        return config

    def grid(self,
             min_dist_threshold=1.5,
             max_attempts=10000):
        """
        Generate molecules on an evenly spaced grid

        :return:
        """
        n_molecules = len(self.molecules)
        system = deepcopy(self)
        sub_box = Box(size=system.box.size - min_dist_threshold)

        def n_x(x, y, z):
            """Calculate the number of atoms in the x direction, given
            two others (y, z)"""
            return int(np.ceil(np.power(x**2 * n_molecules / (y * z), 1/3)))

        grid_points = []
        a, b, c = sub_box.size
        n_a, n_b, n_c = n_x(a, b, c), n_x(b, a, c), n_x(c, a, b)

        # Add all the grid points in 3D over the orthorhombic box
        for i in range(n_a):
            for j in range(n_b):
                for k in range(n_c):
                    vec = np.array([i * a / n_a, j * b / n_b, k * c / n_c])
                    grid_points.append(vec)

        # Need to have fewer molecules than grid points to put them on
        assert len(grid_points) >= n_molecules

        def random_rotate(vec):
            molecule.translate(vec=-molecule.centroid())
            molecule.rotate(axis=np.random.uniform(-1.0, 1.0, size=3),
                            theta=np.random.uniform(0.0, 2 * np.pi))
            molecule.translate(vec)
            return

        n_attempts = 0
        coords = np.empty(shape=(0, 3))

        for i, molecule in enumerate(system.molecules):
            point_idx = int(np.random.randint(0, len(grid_points)))

            if i == 0:
                random_rotate(vec=grid_points[point_idx])
                coords = np.vstack((coords, molecule.coordinates))
                continue

            # Try to add the remaining molecules
            while molecule.min_distance(coords) < min_dist_threshold:

                # Shift back to the origin then to the grid point
                point_idx = int(np.random.randint(1, len(grid_points)))
                random_rotate(vec=grid_points[point_idx])
                n_attempts += 1

                if n_attempts > max_attempts:
                    raise ex.RandomiseFailed('Maximum attempts exceeded')

            # Add the coordinates to the full set
            coords = np.vstack((coords, molecule.coordinates))
            grid_points.pop(point_idx)

        config = system.configuration()

        return config

    def configuration(self):
        return Configuration(system=self)

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

    @property
    def atom_symbols(self):
        """Get all the atom labels/atomic symbols in this system"""
        return [atom.label for m in self.molecules for atom in m.atoms]

    @property
    def n_atoms(self):
        """Get all the atom labels/atomic symbols in this system"""
        return len(self.atom_symbols)

    @property
    def n_unique_molecules(self):
        """Get the number of unique molecules in a system"""
        return len(set([str(mol) for mol in self.molecules]))

    @property
    def density(self):
        """Calculate the density of the system in g cm-3"""

        total_mw = sum([atom.weight.to('amu') for m in self.molecules
                        for atom in m.atoms])  # g mol-1

        # ρ = m / V  ->  ρ = (mw / Na) / (V) * 1E-3
        n_a = 6.02214086E23                            # thing mol^-1
        a_m, b_m, c_m = self.box.size * 1E-10          # m
        per_m3_to_per_cm3 = 1E-6

        rho = (total_mw / n_a) / (a_m * b_m * c_m)     # g m^-3
        rho_g_per_cm3 = per_m3_to_per_cm3 * rho        # g cm^-3

        return rho_g_per_cm3

    @property
    def charge(self):
        """Get the total charge on the system"""
        return sum(molecule.charge for molecule in self.molecules)

    @property
    def mult(self):
        """Get the total spin multiplicity on the system"""
        n_unpaired = sum((mol.mult - 1) / 2 for mol in self.molecules)
        return 2 * n_unpaired + 1

    @property
    def unique_molecules(self):
        """
        Unique molecules that comprise this system populating their atom
        indexes. e.g. For a system of two water molecules:
            mol_idxs = [[0, 1, 2], [3, 4, 5]]
        """
        unq_mols = []
        start_idx = 0

        for mol in self.molecules:

            if not any(str(mol) == m.name for m in unq_mols):
                unq_mols.append(UniqueMolecule(mol))

            for unq_mol in unq_mols:
                if unq_mol.name != str(mol):
                    continue

                end_idx = start_idx + unq_mol.molecule.n_atoms
                unq_mol.atom_idxs.append(list(range(start_idx, end_idx)))

                start_idx += unq_mol.molecule.n_atoms

        return unq_mols

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

        logger.info(f'Initialised a system\n'
                    f'Number of molecules = {len(self.molecules)}\n'
                    f'Charge              = {self.charge} e\n'
                    f'Spin multiplicity   = {self.mult}')


class MMSystem(System):

    def generate_topology(self):
        """Generate a GROMACS topology for this system"""
        for molecule in self.molecules:
            molecule.set_mm_atom_types()

        assert all(m.itp_filename is not None for m in self.molecules)
        itp_names = [mol.itp_filename for mol in self.molecules]

        def print_types(top_file, print_atoms=False, print_molecules=False):

            for itp_filename in sorted(set(itp_names), key=itp_names.index):

                print_flag = False

                for line in open(itp_filename, 'r'):
                    if print_atoms:
                        if 'moleculetype' not in line:
                            print(line, file=top_file)
                        else:
                            break

                    elif print_molecules:
                        if print_flag or 'moleculetype' in line:
                            print(line, file=top_file)
                            print_flag = True
            return None

        with open('topol.top', 'w') as topol_file:
            print(f'[ defaults ]',
                  f'{"; nbfunc":<16}{"comb-rule":<16}'
                  f'{"gen-pairs":<16}{"fudgeLJ":<8}{"fudgeQQ":<8}',
                  f'{"1":<16}{"2":<16}{"yes":<16}{"0.5":<8}{"0.8333":<8}\n',
                  file=topol_file, sep='\n')

            print_types(topol_file, print_atoms=True)
            print_types(topol_file, print_molecules=True)

            print(f'\n[ system ]',
                  f'; Name',
                  f'{str(self)}\n',
                  f'[ molecules ]',
                  f'; Compound{"":<7s}#mols', file=topol_file, sep='\n')

            mol_names = [m.name for m in self.molecules]
            for mol_name in sorted(set(mol_names), key=mol_names.index):
                print(f'{mol_name:<15s}{mol_names.count(mol_name)}',
                      file=topol_file)

    def __init__(self, *args, box_size):
        """System that can be simulated with molecular mechanics"""
        super().__init__(*args, box_size=box_size)
