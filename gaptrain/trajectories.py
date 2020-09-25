from gaptrain.log import logger
import gaptrain as gt
import ase.io.trajectory as ase_traj


class Trajectory(gt.ConfigurationSet):
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

    def extract_from_ase(self, filename, init_config):
        """Load an ASE trajectory as a gt Trajectory by extracting positions"""
        assert init_config is not None

        traj = ase_traj.Trajectory(filename)

        # Iterate through each frame (set of atoms) in the trajectory
        for atoms in traj:
            config = init_config.copy()

            # Set the coordinate of evert atom in the configuration
            for i, position in enumerate(atoms.get_positions()):
                config.atoms[i].coord = position

            self._list.append(config)

        return None

    def __init__(self, filename, init_configuration=None, charge=None,
                 mult=None, box=None):
        super().__init__()

        if filename == 'geo_end.xyz':
            self.extract_from_dftb(init_config=init_configuration)

        elif filename.endswith('.traj'):
            self.extract_from_ase(filename, init_config=init_configuration)

        elif all(prm is not None for prm in (charge, mult, box)):
            self.load(filename, box=box, charge=charge, mult=mult)

        if len(self) == 0:
            logger.warning('Loaded an empty trajectory')
