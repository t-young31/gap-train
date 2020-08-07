from gaptrain.systems import System
from gaptrain.molecules import Molecule, Ion
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_with_charges():

    # Check that charges work
    na_water = System(Ion('Na', charge=1), box_size=[5, 5, 5])
    na_water.add_molecules(h2o, n=5)

    assert na_water.charge() == 1

    na_water.randomise()
    config = na_water.configuration()

    config.run_dftb()
