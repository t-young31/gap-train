class NoEnergy(Exception):
    """Exception for a configuration not having an energy set"""


class NoForces(Exception):
    """Exception for a configuration having no forces"""


class LoadingFailed(Exception):
    """Exception for failure to load a data structure"""


class MethodFailed(Exception):
    """Exception for an electronic structure or ML method failure"""


class RandomiseFailed(Exception):
    """Exception for where randomising a configuration is not possible"""


class PlottingFailed(Exception):
    """Exception for not being able to plot a set of data"""


class GAPFailed(Exception):
    """Exception for where a GAP cannot be run or energies/forces predicted"""
