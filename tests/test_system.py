from gaptrain.systems import System, MMSystem
from gaptrain.molecules import Molecule
from gaptrain.solvents import get_solvent
from scipy.spatial import distance_matrix
from autode.input_output import xyz_file_to_atoms
import numpy as np
import os


here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_random_distance():

    system = System(box_size=[10, 10, 10])
    methane = Molecule(os.path.join(here, 'data', 'methane.xyz'))
    system.add_molecules(methane, n=4)

    config = system.random(min_dist_threshold=1.5)

    for coord in config.coordinates():
        assert np.min(coord) > 0.0
        assert np.max(coord) < 10 - 1.5


def test_system():

    system = System(box_size=[5, 5, 5])
    # No molecules yet in the system
    assert len(system) == 0

    system += h2o

    assert len(system) == 1

    system.add_molecules(h2o, 10)
    assert len(system) == 11

    two_waters = [h2o, h2o]
    system += two_waters
    assert len(system) == 13

    # Should be able to print an xyz file of the configuration
    system.configuration().save(filename='test.xyz')
    assert os.path.exists('test.xyz')
    os.remove('test.xyz')


def test_random_positions():

    system = System(box_size=[15, 15, 15])
    methane = Molecule(os.path.join(here, 'data', 'methane.xyz'))
    system.add_molecules(methane, n=20)

    config = system.random()
    config.save(filename='test_random.xyz')

    # Minimum pairwise distance should be ~ the C-H distance (1.109 Å)
    atoms = xyz_file_to_atoms('test_random.xyz')
    coords = np.array([atom.coord for atom in atoms])
    dist_matrix = distance_matrix(coords, coords)

    # Distance matrix has zeros along the diagonals so add the identity
    assert np.min(dist_matrix + 9 * np.identity(len(coords))) > 1.1

    os.remove('test_random.xyz')


def test_random_grid_positions():

    system = System(box_size=[10, 12, 14])
    methane = Molecule(os.path.join(here, 'data', 'methane.xyz'))
    system.add_molecules(methane, n=5)

    config = system.random(grid=True)
    config.save(filename='test_random.xyz')

    # Minimum pairwise distance should be ~ the C-H distance (1.109 Å)
    atoms = xyz_file_to_atoms('test_random.xyz')
    coords = np.array([atom.coord for atom in atoms])
    dist_matrix = distance_matrix(coords, coords)

    # Distance matrix has zeros along the diagonals so add the identity
    assert np.min(dist_matrix + 9 * np.identity(len(coords))) > 1.1

    os.remove('test_random.xyz')


def test_perturbation():

    system = System(box_size=[7, 7, 7])
    system.add_molecules(h2o, n=5)
    config = system.random(min_dist_threshold=1.5,
                           sigma=0.1,
                           max_length=0.2)

    pcoords = config.coordinates()

    # Adding a displacement to each atom should still afford a reasonably
    # sensible structure (no short distances)
    dist_matrix = distance_matrix(pcoords, pcoords)
    dist_matrix += np.identity(len(pcoords))
    assert np.min(dist_matrix) > 0.6


def test_mm_system():

    system = MMSystem(box_size=[5, 5, 5])
    assert hasattr(system, 'generate_topology')


def test_generate_topology():

    water_solvent = get_solvent('h2o')
    system = MMSystem(water_solvent, box_size=[5, 5, 5])
    MMSystem.generate_topology(system)
    assert os.stat("topol.top").st_size != 0
