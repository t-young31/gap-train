from gaptrain.systems import System
from gaptrain.molecules import Molecule
from gaptrain import GTConfig
from gaptrain import Data
import pytest
import os
from autode.methods import ORCA
from autode.wrappers.keywords import GradientKeywords
from autode.methods import XTB

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))
GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])


def test_autode_orca():

    orca_calculator = ORCA()
    assert isinstance(orca_calculator, ORCA)

    if 'GT_ORCA' not in os.environ or not os.environ['GT_ORCA'] == 'True':
        return

    # Check that orca runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)

    config = water.random()

    config.run_orca()


def test_autode_xtb():

    xtb_calculator = XTB()
    assert isinstance(xtb_calculator, XTB)

    if 'GT_XTB' not in os.environ or not os.environ['GT_XTB'] == 'True':
        return

    # Check that orca and xtb runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=5)

    config = water.random()

    config.run_xtb()


def test_autode_parallel_orca():

    orca_calculator = ORCA()
    assert isinstance(orca_calculator, ORCA)

    if 'GT_ORCA' not in os.environ or not os.environ['GT_ORCA'] == 'True':
        return

    # Check that orca runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)
    monomer = water.random()

    configs = Data()
    for _ in range(3):
        monomer_pert = monomer.copy()
        monomer_pert.add_perturbation(sigma=0.05, max_length=1)
        configs += monomer_pert

    configs.parallel_orca()


def test_autode_parallel_xtb():
    xtb_calculator = XTB()
    assert isinstance(xtb_calculator, XTB)

    if 'GT_XTB' not in os.environ or not os.environ['GT_XTB'] == 'True':
        return

    # Check that xtb runs work
    water = System(box_size=[10, 10, 10])
    water.add_molecules(h2o, n=2)
    monomer = water.random()

    configs = Data()
    for _ in range(3):
        monomer_pert = monomer.copy()
        monomer_pert.add_perturbation(sigma=0.05, max_length=1)
        configs += monomer_pert

    configs.parallel_xtb()
