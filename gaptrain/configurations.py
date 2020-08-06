from gaptrain.ef import Energy, Forces


class Configuration:

    def __init__(self, system):
        """
        A configuration consisting of a set of atoms suitable to run DFT
        or GAP on to set self.energy and self.forces

        :param system: (gaptrain.system.System)
        """

        self.atoms = []
        for molecule in system.molecules:
            self.atoms += molecule.atoms

        self.forces = Forces()
        self.energy = Energy()

        self.box = system.box
        self.charge = system.charge
        self.mult = system.mult


class ConfigurationSet:

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        return iter(self._list)

    def __init__(self, *args):
        """Set of configurations

        :args: (gaptrain.configurations.Configuration)
        """

        self._list = args
