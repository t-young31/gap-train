from gaptrain.configurations import ConfigurationSet
from gaptrain.log import logger
import matplotlib.pyplot as plt
import numpy as np


class Data(ConfigurationSet):

    def histogram(self, name=None):
        """Generate a histogram of the energies and forces in these data"""
        logger.info('Plotting histogram of energies and forces')

        fig, (ax_e, ax_f) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        # Total energy histogram
        energies = [config.energy.true for config in self._list]
        ax_e.hist(energies, bins=np.linspace(min(energies), max(energies), 30),
                  alpha=0.5,
                  edgecolor='black',
                  linewidth=0.2)

        # Energy histogram formatting
        ax_e.ticklabel_format(style='sci', scilimits=(0,0))
        ax_e.set_xlabel('Energy / eV')

        # Force magnitude histogram
        force_magnitudes = tuple(c.forces.true_magnitudes() for c in self._list)
        all_mod_f = np.concatenate(force_magnitudes)
        ax_f.hist(all_mod_f,
                  bins=np.linspace(min(all_mod_f), max(all_mod_f), 100),
                  color='orange',
                  alpha=0.5,
                  edgecolor='black',
                  linewidth=0.2)

        # Force histogram formatting
        ax_f.set_xlabel('|$F$| / ev Å$^{-1}$')

        for ax in (ax_e, ax_f):
            ax.set_ylabel('Frequency')

        if name is None:
            plt.show()

        else:
            plt.savefig(f'{name}.png', dpi=300)

        return None

    def __init__(self, *args, name):
        super().__init__(*args, name=name)
