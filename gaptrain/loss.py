import numpy as np


class Loss:

    def __str__(self):
        return (f'Loss: energy = {self.energy:.4f} eV, '
                f'force  = {self.force:.4f} eV Å-1')

    def function(self, deltas):
        """Function that transforms [∆v1, ∆v2, ..] into a float e.g. MSE"""
        raise NotImplementedError

    def loss(self, cfs_a, cfs_b, attr):
        """
        Compute the root mean squared loss between two sets of configurations

        -----------------------------------------------------------------------
        :param cfs_a: (gaptrain.configurations.ConfigurationSet)

        :param cfs_b: (gaptrain.configurations.ConfigurationSet)

        :return: (gaptrain.loss.RMSE)
        """

        assert len(cfs_a) == len(cfs_b)

        deltas = []

        for (ca, cb) in zip(cfs_a, cfs_b):
            deltas.append(getattr(ca, attr) - getattr(cb, attr))

        return self.function(deltas)

    def __init__(self, cfs_a, cfs_b):
        """
        A loss on energies (eV) and forces (eV Å-1). Force loss is computed
        element-wise

        :param cfs_a: (gaptrain.configurations.ConfigurationSet)
        :param cfs_b: (gaptrain.configurations.ConfigurationSet)
        """
        self.energy = self.loss(cfs_a, cfs_b, attr='energy')
        self.force = self.loss(cfs_a, cfs_b, attr='forces')


class RMSE(Loss):
    """Root mean squared error"""

    def function(self, deltas):
        return np.sqrt(np.mean(np.square(deltas)))
