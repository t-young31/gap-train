
# Default parameters for a GAP potential
default_params = {}


class GAP:

    def predict(self, data, plot_energy=True, plot_force=True):
        """
        Predict energies and forces for a set of data

        -----------------------------------------------------------------------
        :param data: (gaptrain.data.Data)

        :param plot_energy: (bool) Plot an energy correlation: predicted v true

        :param plot_force: (bool) Plot a force correlation
        """
        data.run_gap(self)
        raise NotImplementedError

    def train(self, data):
        """
        Train this GAP

        :param data: (gaptrain.data.Data)
        :return:
        """
        raise NotImplementedError

    def __init__(self, name):
        """A Gaussian Approximation Potential"""

        self.name = name
        self.params = default_params
