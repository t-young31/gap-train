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

    def force_magnitudes(self):
        """Get a list of all |F| in these data"""
        mod_fs = []

        for config in self._list:
            force_magnitudes = np.linalg.norm(config.forces, axis=1)
            mod_fs += force_magnitudes.tolist()

        return mod_fs

    def histogram(self, name=None, ref_energy=None):
        """Generate a histogram of the energies and forces in these data"""
        logger.info('Plotting histogram of energies and forces')
        # Histogram |F_ij| rather than the components F_ijk for a force F in on
        # an atom j in a configuration i
        if ref_energy is not None:
            logger.info(f'Referencing energy to {ref_energy} eV')
            energies = self.energies() - ref_energy

        else:
            energies = self.energies()

        return histogram(energies, self.force_magnitudes(), name=name)

    def __init__(self, *args, name=None):
        super().__init__(*args, name=name)
