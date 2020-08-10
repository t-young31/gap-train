from gaptrain.configurations import ConfigurationSet
from gaptrain.log import logger
import matplotlib.pyplot as plt
import numpy as np


class Data(ConfigurationSet):

    def histogram(self):
        """Generate a histogram of the energies and forces in these data"""
        raise NotImplementedError

    def remove_first(self, n):
        """
        Remove the first n configurations from these data

        :param n: (int)
        """
        self._list = self._list[n:]
        return None

    def remove_random(self, n=None, remainder=None):
        """Randomly remove some configurations from these data

        :param n: (int) Number of configurations to remove
        :param remainder: (int) Number of configurations left in these data
        """
        # Number to choose is the total minus the number to remove
        if n is not None:
            remainder = len(self) - int(n)

        elif remainder is not None:
            remainder = int(remainder)

        else:
            logger.warning('No configurations to remove')
            return None

        self._list = np.random.choice(self._list, size=remainder)
        return None

    def __init__(self, *args, name):
        super().__init__(*args, name=name)
