from gaptrain.box import Box
from gaptrain.molecules import Molecule
import numpy as np
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_random_grid_point():

    box = Box(size=[10, 10, 10])
    point = box.random_grid_point(spacing=h2o.radius)

    for _ in range(10):
        new_point = box.random_grid_point(spacing=h2o.radius)
        dist = np.linalg.norm(point - new_point)

        # Could be the same point or something larger
        assert dist < 1E-6 or dist > h2o.radius - 1E-4
