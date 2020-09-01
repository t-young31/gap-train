from gaptrain.configurations import ConfigurationSet
from gaptrain.log import logger


class Trajectory(ConfigurationSet):
    """MD trajectory frames"""

    def extract_from_dftb(self, init_config):
        """Load a DFTB+ trajectory - must have an associated configuration"""
        assert init_config is not None

        # Initial box size, charge and multiplicity are all retained in NVT
        # AIMD dynamics
        return self.load(filename='geo_end.xyz',
                         box=init_config.box,
                         charge=init_config.charge,
                         mult=init_config.mult)

    def __init__(self, filename, init_configuration=None, charge=None,
                 mult=None, box=None):
        super().__init__()

        if filename == 'geo_end.xyz':
            self.extract_from_dftb(init_config=init_configuration)

        if all(prm is not None for prm in (charge, mult, box)):
            self.load(filename, box=box, charge=charge, mult=mult)

        if len(self) == 0:
            logger.warning('Loaded an empty trajectory')
