from gaptrain.systems import System
from gaptrain.molecules import Molecule
from gaptrain import GTConfig
from gaptrain import Data
import os
from autode.wrappers.keywords import GradientKeywords

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2co.xyz'))
GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])


def test_autode_orca():

    if 'GT_ORCA' not in os.environ or not os.environ['GT_ORCA'] == 'True':
        return

    # Check that orca runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)

    config = water.random()
    config.run_orca()

    assert config.energy is not None
    assert config.forces is not None


def test_autode_xtb():

    if 'GT_XTB' not in os.environ or not os.environ['GT_XTB'] == 'True':
        return

    # Check that orca and xtb runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=5)

    config = water.random()
    config.run_xtb()

    assert config.energy is not None
    assert config.forces is not None


def test_autode_parallel_orca():

    if 'GT_ORCA' not in os.environ or not os.environ['GT_ORCA'] == 'True':
        return

    # Check that orca runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)

    configs = Data()
    for _ in range(3):
        configs += water.random(with_intra=True)

    configs.parallel_orca()


def test_autode_parallel_xtb():

    if 'GT_XTB' not in os.environ or not os.environ['GT_XTB'] == 'True':
        return

    # Check that xtb runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)
    water.add_molecules(h2o, n=2)

    configs = Data()
    for _ in range(3):
        configs += water.random(with_intra=True)

    configs.parallel_xtb()
