from gaptrain.systems import System
from gaptrain.molecules import Molecule, Ion
from gaptrain.calculators import DFTB
from ase.calculators.dftb import Dftb
import pytest
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_with_charges():

    # Slightly modified ASE calculator to be compatible with the latest DFTB+
    calculator = DFTB()
    assert isinstance(calculator, Dftb)

    # Can't get the fermi level with no calculation run
    with pytest.raises(TypeError):
        calculator.read_fermi_levels()

    if 'GT_DFTB' not in os.environ or not os.environ['GT_DFTB'] == 'True':
        return

    # Check that charges work
    na_water = System(box_size=[10, 10, 10])
    na_water.add_molecules(Ion('Na', charge=1), n=1)
    na_water.add_molecules(h2o, n=5)

    assert na_water.charge == 1

    config = na_water.random()
    config.run_dftb()
