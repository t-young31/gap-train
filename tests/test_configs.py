from gaptrain.configurations import ConfigurationSet, Configuration
from gaptrain.systems import System
from gaptrain.molecules import Molecule
import gaptrain as gt
import numpy as np
import ase
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))

side_length = 7.0
system = System(box_size=[side_length, side_length, side_length])
system.add_molecules(h2o, n=3)


def test_print_exyz():

    configs = ConfigurationSet(name='test')

    for _ in range(5):
        configs += system.random()

    configs.save()
    assert os.path.exists('test.xyz')
    os.remove('test.xyz')

    # If the energy and forces are set for all the configurations an exyz
    # should be able to be printed
    for config in configs:
        config.energy = 1.0
        config.forces = np.zeros(shape=(9, 3))

    configs.save()

    assert os.path.exists('test.xyz')
    for line in open('test.xyz', 'r'):

        items = line.split()
        # Number of atoms in the configuration
        if len(items) == 1:
            assert int(items[0]) == 9

        if len(items) == 7:
            atomic_symbol, x, y, z, fx, fy, fz = items

            # Atomic symbols should be letters
            assert all(letter.isalpha() for letter in atomic_symbol)

            # Positions should be float-able and inside the box
            for component in (x, y, z):
                assert 0.0 < float(component) < side_length

            # Forces should be ~0, as they were set above
            assert all(-1E-6 < float(fk) < 1E-6 for fk in (fx, fy, fz))

    os.remove('test.xyz')


def test_wrap():

    config = system.random(on_grid=True)
    for atom in config.atoms[:3]:
        atom.translate(vec=np.array([10.0, 0, 0]))

    # One water molecule should be outside of the box
    assert np.max(config.coordinates()) > 7.0

    # Wrapping should put all the atoms back into the box
    config.wrap()
    assert np.max(config.coordinates()) < 7.0

    # An atom several box lengths outside the box should still be able to be
    # wrapped into the box
    for atom in config.atoms[:3]:
        atom.translate(vec=np.array([30.0, 0, 0]))

    config.wrap()
    assert np.max(config.coordinates()) < 7.0

    # With a coordinate at infinity then the function should not overflow
    config.atoms[0].translate(vec=np.array([np.inf, 0, 0]))
    config.wrap()
    assert config.atoms[0].coord[0] == np.inf


def test_ase_atoms():

    ase_atoms = Configuration(system).ase_atoms()

    assert isinstance(ase_atoms, ase.Atoms)
    # Periodic in x y and z
    assert all(ase_atoms.pbc)
    # Cell vectors should all be ~ 5 Å
    for vec in ase_atoms.cell:
        assert side_length - 0.1 < np.linalg.norm(vec) < side_length + 0.1


def test_dftb_plus():

    water_box = System(box_size=[5, 5, 5])

    config = water_box.configuration()
    config.set_atoms(xyz_filename=os.path.join(here, 'data', 'h2o_10.xyz'))

    if 'GT_DFTB' not in os.environ or not os.environ['GT_DFTB'] == 'True':
        return

    config.run_dftb()
    assert config.energy is not None

    forces = config.forces
    assert type(forces) is np.ndarray
    assert forces.shape == (30, 3)

    # Should all be non-zero length force vectors in ev Å^-1
    assert all(0 < np.linalg.norm(force) < 70 for force in forces)


def test_remove():

    configs = ConfigurationSet()
    for _ in range(10):
        configs += Configuration()

    configs[0].energy = 1

    # Removing the
    configs.remove_first(n=1)
    assert configs[0].energy is None

    configs.remove_random(n=2)
    assert len(configs) == 7

    configs.remove_random(remainder=2)
    assert len(configs) == 2


def test_load_no_box():

    data = gt.Data()
    data.load(filename=os.path.join(here, 'data', 'rnd_training.xyz'))
    assert len(data) > 0

    for config in data:

        assert config.energy is not None
        assert config.charge == 0
        assert config.mult == 1


def test_remove_energy_threshold():

    configs = ConfigurationSet(system.random(),
                               system.random())

    assert len(configs) == 2
    configs[0].energy = -1000
    configs[1].energy = -1000 + 10

    # Should keep both configurations
    configs.remove_above_e(threshold=20)
    assert len(configs) == 2

    configs.remove_above_e(threshold=5)
    assert len(configs) == 1
    assert configs[0].energy == -1000

    configs.remove_above_e(threshold=10, min_energy=-1020)
    assert len(configs) == 0


def test_remove_force_threshold():

    configs = ConfigurationSet(system.random(),
                               system.random())

    configs[0].forces = np.ones(shape=(9, 3))
    configs[1].forces = 3 * np.ones(shape=(9, 3))

    assert len(configs) == 2

    configs.remove_above_f(threshold=2)
    assert len(configs) == 1

    configs.remove_above_f(threshold=0.5)
    assert len(configs) == 0


def test_remove_higher():

    configs = ConfigurationSet(system.random(),
                               system.random())

    assert len(configs) == 2
    configs[0].energy = -1000
    configs[1].energy = -1000 + 10

    configs.truncate(n=1, method='higher')
    assert len(configs) == 1
    assert configs[0].energy == -1000

    for i in range(1, 10):
        config = system.random()
        config.energy = -1000 + i
        configs += config

    configs.truncate(n=5, method='higher')
    assert len(configs) == 5
    for i in range(5):
        assert configs[i].energy == -1000 + i


# TODO this function
def FIXME_gap_ensemble_truncate():

    return
    os.chdir(os.path.join(here, 'data', 'gap_ensemble'))

    water_box = System(box_size=[7, 7, 7])
    water_box.add_molecules(h2o, n=5)

    # Add data to the box
    configs = ConfigurationSet(name='water_configs')
    for _ in range(30):
        configs += water_box.random()

    configs.parallel_dftb()

    # Don't attempt without the GAP flag
    if 'GT_GAP' not in os.environ or not os.environ['GT_GAP'] == 'True':
        return

    gt.GTConfig.gap_default_soap_params['n_sparse'] = 10
    gt.GTConfig.gap_default_2b_params['n_sparse'] = 10

    ensemble = gt.GAPEnsemble(name='water_ensemble',
                              system=water_box,
                              n=3)
    ensemble.train(configs)

    # Add a third configuration and predict the error
    new_config = water_box.random()

    configs += new_config
    gt.GTConfig.n_cores = 8

    errors = ensemble.predict_energy_error(configs)
    print(errors)

    # Error on the newly added configuration should be the largest
    assert all(errors[-1] > error for error in errors[:-1])

    # Truncate the configurations by removing the largest predicted error
    configs.truncate(n=1, method='ensemble', ensemble=ensemble)

    # Truncation should leave only the configuration that is most disimillar
    # to the training data, i.e. the newly added one
    dist = np.linalg.norm(new_config.coordinates() - configs[0].coordinates())
    assert dist < 1E-6

    os.chdir(here)
