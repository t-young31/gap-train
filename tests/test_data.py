from gaptrain.data import Data
from gaptrain.molecules import Molecule
from gaptrain.systems import System
from scipy.spatial import distance_matrix
from gaptrain.exceptions import NoEnergy
import numpy as np
import pytest
import os


here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_gap():

    water_dimer = System(box_size=[7, 7, 7])
    water_dimer.add_molecules(h2o, n=2)

    data = Data(name='test')
    for _ in range(2):
        data += water_dimer.random()

    assert len(data) == 2

    coords1 = data[0].coordinates()
    assert coords1.shape == (6, 3)

    coords2 = data[1].coordinates()

    # Coordinates should be somewhat different
    assert np.min(distance_matrix(coords1, coords2)) > 1E-3


def test_histogram():

    water_dimer = System(box_size=[7, 7, 7])
    water_dimer.add_molecules(h2o, n=2)

    data = Data(name='test')
    data += water_dimer.random()

    # Can't histogram energies and forces with no energies
    with pytest.raises(NoEnergy):
        data.histogram(name='test')

    data[0].energy = 1
    data[0].forces = np.zeros(shape=(6, 3))

    data.histogram(name='test')
    assert os.path.exists('test.png')
    os.remove('test.png')
