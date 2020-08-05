from gaptrain.systems import System
from gaptrain.molecules import Molecule
import os

here = os.path.abspath(os.path.dirname(__file__))


def test_system():

    system = System(box_size=[5, 5, 5], charge=0)
    # No molecules yet in the system
    assert len(system) == 0

    h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))
    system += h2o

    assert len(system) == 1

    system.add_molecules(h2o, 10)
    assert len(system) == 11

    two_waters = [h2o, h2o]
    system += two_waters
    assert len(system) == 13
