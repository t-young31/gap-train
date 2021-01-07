from gaptrain.systems import System
from gaptrain.molecules import Molecule
import pytest
import os
from autode.methods import XTB

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_run_xtb():

    calculator = XTB()
    assert isinstance(calculator, XTB)

    if 'GT_XTB' not in os.environ or not os.environ['GT_XTB'] == 'True':
        return

    # Check that xtb runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=5)

    config = water.random()
    config.run_xtb()
