class NoEnergy(Exception):
    """Exception for a configuration not having an energy set"""


class NoForces(Exception):
    """Exception for a configuration having no forces"""


class CannotAdd(Exception):
    """Exception for not being able to add a configuration"""


class LoadingFailed(Exception):
    """Exception for failure to load a data structure"""


class MethodFailed(Exception):
    """Exception for an electronic structure or ML method failure"""
