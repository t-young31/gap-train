from gaptrain.log import logger
import numpy as np


class DeltaError:
    """A error based on difference between two sets of configurations, one
    (probably) predicted with GAP and one ground truth"""

    def __str__(self):
        return (f'Loss: energy = {self.energy:.4f} eV, '
                f'force  = {self.force:.4f} eV Å-1')

    def function(self, deltas):
        """Function that transforms [∆v1, ∆v2, ..] into a float e.g. MSE"""
        raise NotImplementedError

    def loss(self, configs_a, configs_b, attr):
        """
        Compute the root mean squared loss between two sets of configurations

        -----------------------------------------------------------------------
        :param configs_a: (gaptrain.configurations.ConfigurationSet)

        :param configs_b: (gaptrain.configurations.ConfigurationSet)

        :param attr: (str) Attribute of a configuration to calculate use in
                           the loss function e.g. energy or froca

        :return: (gaptrain.loss.RMSE)
        """
        assert len(configs_a) == len(configs_b)

        deltas = []

        for (ca, cb) in zip(configs_a, configs_b):
            val_a, val_b = getattr(ca, attr), getattr(cb, attr)

            if val_a is None or val_b is None:
                logger.warning(f'Cannot calculate loss for {attr} at least '
                               f'one value was None')
                return None

            # Append the difference between the floats
            deltas.append(val_a - val_b)

        return self.function(np.array(deltas))

    def __init__(self, configs_a, configs_b):
        """
        A loss on energies (eV) and forces (eV Å-1). Force loss is computed
        element-wise

        :param configs_a: (gaptrain.configurations.ConfigurationSet)
        :param configs_b: (gaptrain.configurations.ConfigurationSet)
        """
        logger.info(f'Calculating loss between {len(configs_a)} configurations')

        self.energy = self.loss(configs_a, configs_b, attr='energy')
        self.force = self.loss(configs_a, configs_b, attr='forces')


class RMSE(DeltaError):
    """Root mean squared error"""

    def function(self, deltas):
        return np.sqrt(np.mean(np.square(deltas)))


class MSE(DeltaError):
    """Mean squared error"""

    def function(self, deltas):
        return np.mean(np.square(deltas))


class Tau:

    def calculate(self, gap, method_name):

        raise NotImplementedError

    def __init__(self, configs, e_lower=0.1, e_thresh=1, max_fs=1000,
                 interval_fs=20):
        """
        τ_acc prospective error metric in fs

        ----------------------------------------------------------------------
        :param configs:

        :param e_lower:

        :param e_thresh:

        :param max_fs:

        :param interval_fs:
        """

        self.value = 0
