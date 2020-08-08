from gaptrain.systems import System
from gaptrain.molecules import Molecule, Ion
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_with_charges():

    if 'GT_DFTB' not in os.environ or not os.environ['GT_DFTB'] == 'True':
        return

    # Check that charges work
    na_water = System(box_size=[5, 5, 5])
    na_water.add_molecules(Ion('Na', charge=1), n=1)
    na_water.add_molecules(h2o, n=5)

    assert na_water.charge() == 1

    config = na_water.random()
    config.run_dftb()
