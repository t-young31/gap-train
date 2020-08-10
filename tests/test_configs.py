from gaptrain.configurations import ConfigurationSet, Configuration
from gaptrain.systems import System
from gaptrain.molecules import Molecule
from gaptrain.exceptions import NoEnergy
import numpy as np
import ase
import pytest
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))

system = System(box_size=[5, 5, 5])
system.add_molecules(h2o, n=3)


def test_print_exyz():

    configs = ConfigurationSet(name='test')

    for _ in range(5):
        configs += system.random()

    # Should not be able to save ground truth without calculating
    # energies or forces
    with pytest.raises(NoEnergy):
        configs.save_true()

    os.remove('test.xyz')

    # If the energy and forces are set for all the configurations an exyz
    # should be able to be printed
    for config in configs:
        config.energy.true = 1.0
        for i in range(9):
            config.forces[i].true = np.zeros(3)

    configs.save_true()

    assert os.path.exists('test.xyz')
    os.remove('test.xyz')


def test_ase_atoms():

    ase_atoms = Configuration(system).ase_atoms()

    assert isinstance(ase_atoms, ase.Atoms)
    # Periodic in x y and z
    assert all(ase_atoms.pbc)
    # Cell vectors should all be ~ 5 Å
    assert all(4.9 < np.linalg.norm(vec) < 5.1 for vec in ase_atoms.cell)


def test_dftb_plus():

    water_box = System(box_size=[5, 5, 5])

    config = water_box.configuration()
    config.set_atoms(xyz_filename=os.path.join(here, 'data', 'h2o_10.xyz'))

    if 'GT_DFTB' not in os.environ or not os.environ['GT_DFTB'] == 'True':
        return

    config.run_dftb()
    assert config.energy.true is not None
    assert config.energy.predicted is None

    forces = config.forces.true()
    assert type(forces) is np.ndarray
    assert forces.shape == (30, 3)

    # Should all be non-zero length force vectors in ev Å^-1
    assert all(0 < np.linalg.norm(force) < 70 for force in forces)
