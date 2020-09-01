from gaptrain.plotting import histogram, correlation
import numpy as np
import pytest
import os


def test_histogram():

    energies_forces = [([1.0], [1.0]),
                       ([1.0], None),
                       (None, [1.0])]

    # Should be able to plot energies and/or forces
    for energies, forces in energies_forces:
        print(energies, forces)

        histogram(energies, forces, name='test')
        assert os.path.exists('test.png')
        os.remove('test.png')

    with pytest.raises(AssertionError):
        histogram(name='test')


def test_correlation():

    true_energies = np.random.uniform(-1, 1, size=10)
    predicted_energies = true_energies

    correlation(true_energies, predicted_energies, name='energies')
    assert os.path.exists('energies.png')
    os.remove('energies.png')

    forces = np.random.uniform(-1, 1, size=500)
    correlation(true_forces=forces, predicted_forces=forces, name='forces')
    assert os.path.exists('forces.png')
    os.remove('forces.png')
