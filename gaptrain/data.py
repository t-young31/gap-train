from gaptrain.configurations import ConfigurationSet
from gaptrain.log import logger
import gaptrain.exceptions as ex
from gaptrain.plotting import histogram
import numpy as np


class Data(ConfigurationSet):

    def energies(self):
        """Get a list of all the energies in these data"""

        energies = [config.energy for config in self._list]

        if any(energy is None for energy in energies):
            raise ex.NoEnergy('Cannot histogram data - some energies = None')

        return np.array(energies)

    def force_components(self):
        """Get a 1D numpy array of all F_k in these data"""
        fs = []

        for config in self._list:
            fs += [component for force in config.forces for component in force]

        return np.array(fs)

    def force_magnitudes(self):
        """Get a 1D numpy array of all |F| in these data"""
        mod_fs = []

        for config in self._list:
            force_magnitudes = np.linalg.norm(config.forces, axis=1)
            mod_fs += force_magnitudes.tolist()

        return np.array(mod_fs)

    def histogram(self, name=None, ref_energy=None):
        """Generate a histogram of the energies and forces in these data"""
        logger.info('Plotting histogram of energies and forces')
        # Histogram |F_ij| rather than the components F_ijk for a force F in on
        # an atom j in a configuration i
        return histogram(self.energies(), self.force_magnitudes(),
                         name=name, ref_energy=ref_energy)

    def __init__(self, *args, name='data'):
        super().__init__(*args, name=name)
