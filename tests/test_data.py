from gaptrain.data import Data
from gaptrain.molecules import Molecule
from gaptrain.systems import System
from scipy.spatial import distance_matrix
import numpy as np
import os


here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_gap():

    water_dimer = System(box_size=[7, 7, 7])
    water_dimer.add_molecules(h2o, n=2)

    print(water_dimer.random())

    data = Data(name='test')
    for _ in range(2):
        data += water_dimer.random()

    assert len(data) == 2

    coords1 = data[0].coordinates()
    assert coords1.shape == (6, 3)

    coords2 = data[1].coordinates()

    # Coordinates should be somewhat different
    assert np.min(distance_matrix(coords1, coords2)) > 1E-3
