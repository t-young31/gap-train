from gaptrain.systems import System, MMSystem
from gaptrain.molecules import Molecule
from scipy.spatial import distance_matrix
from autode.input_output import xyz_file_to_atoms
import numpy as np
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_system():

    system = System(box_size=[5, 5, 5], charge=0)
    # No molecules yet in the system
    assert len(system) == 0

    system += h2o

    assert len(system) == 1

    system.add_molecules(h2o, 10)
    assert len(system) == 11

    two_waters = [h2o, h2o]
    system += two_waters
    assert len(system) == 13

    # Should be able to print an xyz file
    system.print_xyz_file(filename='test.xyz')
    assert os.path.exists('test.xyz')
    os.remove('test.xyz')


def test_random_positions():

    system = System(box_size=[10, 10, 10], charge=0)
    h2o = Molecule(os.path.join(here, 'data', 'methane.xyz'))
    system.add_molecules(h2o, n=45)

    system.randomise()
    system.print_xyz_file(filename='test_random.xyz')

    # Minimum pairwise distance should be ~ the C-H distance (1.109 Å)
    atoms = xyz_file_to_atoms('test_random.xyz')
    coords = np.array([atom.coord for atom in atoms])
    dist_matrix = distance_matrix(coords, coords)

    # Distance matrix has zeros along the diagonals so add the identity
    assert np.min(dist_matrix + 9 * np.identity(len(coords))) > 1.1

    os.remove('test_random.xyz')


def test_mm_system():

    system = MMSystem(box_size=[5, 5, 5], charge=0)
    assert hasattr(system, 'generate_topology')
