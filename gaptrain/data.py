from gaptrain.configurations import ConfigurationSet
import matplotlib.pyplot as plt


class Data(ConfigurationSet):

    def histogram(self):
        """Generate a histogram of the energies and forces in these data"""
        raise NotImplementedError

    def __init__(self, *args, name):
        super().__init__(*args, name=name)
