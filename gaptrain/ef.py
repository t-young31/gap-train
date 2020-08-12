import numpy as np


class Value:

    def __str__(self):
        return f'True = {self.true}, Predicted = {self.predicted}'

    def __init__(self, true=None, predicted=None):
        self.true = true
        self.predicted = predicted


class Energy(Value):
    """Energy in eV"""


class Force(Value):
    """Force in eV Å^-1"""


class Forces:

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __getitem__(self, item):
        return self._list[item]

    def _set(self, forces, true_values):
        """Set the forces on this system from a numpy array"""
        assert type(forces) is np.ndarray
        assert forces.shape == (len(self), 3)

        for i, force in enumerate(forces):

            if true_values:
                self._list[i].true = force
            else:
                self._list[i].predicted = force

        return None

    def true_magnitudes(self):
        """Get a numpy array true magnitudes of the vectors"""
        return np.array([np.linalg.norm(force.true) for force in self._list])

    def set_true(self, forces):
        """Set the true forces on this system from a numpy array"""
        return self._set(forces, true_values=True)

    def true(self):
        """Get a numpy array n_atoms x 3 of the true forces"""
        assert all(force.true is not None for force in self._list)
        return np.array([force.true for force in self._list])

    def predicted(self):
        """Get a numpy array n_atoms x 3 of the predicted forces"""
        assert all(force.predicted is not None for force in self._list)
        return np.array([force.predicted for force in self._list])

    def __init__(self, n_atoms=0):
        """
        Set of forces on a system/configuration

        :param n_atoms: (int) Number of atoms
        """

        self._list = [Force() for _ in range(n_atoms)]
