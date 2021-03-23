import autode as ade
from autode.input_output import xyz_file_to_atoms
from autode.geom import calc_rmsd
from autode.atoms import Atom
from autode.atoms import get_vdw_radius
from gaptrain.log import logger
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import numpy as np


class Species(ade.species.Species):

    def __repr__(self):
        return f'Species(name={self.name}, n_atoms={len(self.atoms)})'

    def __eq__(self, other):
        """Are two molecules the same?"""
        if str(other) != str(self) or len(self.atoms) != len(other.atoms):
            return False

        return True

    def __str__(self):
        """Chemical formula for this species e.g. H2O"""
        name = ""
        atom_symbols = [atom.label for atom in self.atoms]

        for atom_symbol in set(atom_symbols):
            count = atom_symbols.count(atom_symbol)
            name += f'{atom_symbol}{count if count > 1 else ""}'

        return name

    def in_box(self, box):
        """Is this molecule totally inside a box with an origin at
        (0,0,0) and top right corner (a, b, c) = box.size

        :param box: (gaptrain.box.Box)
        """
        coords = self.coordinates

        if np.min(coords) < 0.0:
            return False

        # Maximum x, y, z component of all atoms should be < a, b, c
        # respectively
        if max(np.max(coords, axis=0) - box.size) > 0:
            return False

        return True

    def min_distance(self, coords):
        """Calculate the minimum distance from this molecule to a set
        of coordinates

        :param coords: (np.ndarray) shape = (n, 3)
        """
        # Infinite distance to the other set if there are no coordinates
        if len(coords) == 0:
            return np.inf

        return np.min(cdist(coords, self.coordinates))

    def centroid(self):
        """
        Get the centroid of this molecule

        :return: (np.ndarray) shape = (3,)
        """
        return np.average(self.coordinates, axis=0)

    @property
    def radius(self):
        """
        Calculate the radius of this species as half the maximum distance
        between two atoms plus the van der Walls radius of H if there are >1
        atoms otherwise

        :return: (float) Radius in Å
        """
        if self.n_atoms == 1:
            return get_vdw_radius(atom_label=self.atoms[0].label)

        coords = self.coordinates
        max_distance = np.max(distance_matrix(coords, coords))

        logger.warning('Assuming hydrogen on the exterior in calculating the '
                       f'radius of {self.name}')
        return max_distance / 2.0 + get_vdw_radius('H')

    def set_mm_atom_types(self):
        """Set the molecular mechanics (MM) atoms types for this molecule"""
        assert self.itp_filename is not None
        logger.info(f'Setting MM atom types from {self.itp_filename}')

        atom_types = []
        lines = open(self.itp_filename, 'r').readlines()

        for i, line in enumerate(lines):
            if "atoms" in line:
                n = 0
                while n < len(self.atoms):
                    n += 1
                    split_atoms = lines[i + n].split()

                    # Assumes atomtype is 5th entry
                    atom_types.append(split_atoms[4])
                break

        for j, atom in enumerate(self.atoms):
            atom.mm_type = atom_types[j]

        return None

    def __init__(self, name="mol", atoms=None, charge=0, spin_multiplicity=1,
                 gmx_itp_filename=None):
        super().__init__(name=name, atoms=atoms, charge=charge,
                         mult=spin_multiplicity)

        self.itp_filename = gmx_itp_filename


class Molecule(Species):

    def __init__(self,  xyz_filename=None,  charge=0, spin_multiplicity=1,
                 gmx_itp_filename=None, atoms=None):
        """Molecule e.g. H2O

        -----------------------------------------------------------------------
        :param xyz_filename: (str)

        :param charge: (int)

        :param spin_multiplicity: (int)

        :param gmx_itp_filename: (str) Filename(path) of the GROMACS .itp file
                                 containing MM parameters required to simulate

        :param atoms: (list(autode.atoms.Atom))
        """
        if xyz_filename is not None:
            atoms = xyz_file_to_atoms(xyz_filename)

        super().__init__(charge=charge,
                         spin_multiplicity=spin_multiplicity,
                         atoms=atoms)

        self.itp_filename = gmx_itp_filename
        self.name = str(self)

        logger.info(f'Initialised {self.name}\n'
                    f'Number of atoms      = {self.n_atoms}\n'
                    f'GROMACS itp filename = {self.itp_filename}')


class Ion(Species):

    def __init__(self, label, charge, spin_multiplicity=1,
                 gmx_itp_filename=None):
        """Ion

        -----------------------------------------------------------------------
        :param label: (str) e.g. 'Pd'

        :param charge: (int)

        :param spin_multiplicity: (int)
        """
        super().__init__(name=label,
                         charge=charge,
                         spin_multiplicity=spin_multiplicity,
                         atoms=[Atom(label)],
                         gmx_itp_filename=gmx_itp_filename)

