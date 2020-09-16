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

    def extract_from_gro(self, gro_traj, system):
        """Convert a GROMACS .gro trajectory to a .xyz trajectory"""
        with open('nvt_traj.xyz', 'w') as output_xyz:

            lines = open(gro_traj, 'r').readlines()

            # Number of atoms is second line of gro file
            num_of_atoms = int(lines[1])
            stride = (num_of_atoms + 3)

            # Create an ordered list of atoms from system.molecules
            atom_list = []
            for molecule in system.molecules:
                for atom in molecule.atoms:
                    atom_list.append(atom.label)

            # Iterate through the entire gro file
            for i, _ in enumerate(lines[::stride]):

                print(f'{num_of_atoms}\n'
                      f'Comment line', file=output_xyz)

                # Iterate through the coordinate lines of each frame only
                for j, line in enumerate(
                        lines[i * stride + 2:(i + 1) * stride - 1]):

                    # Extract the x, y, z coordinates and atom names
                    x_nm, y_nm, z_nm = line.split()[3:6]
                    x, y, z = 10 * float(x_nm), \
                              10 * float(y_nm), 10 * float(z_nm)

                    print(f'{atom_list[j]:<4} {x:<7.3f} {y:<7.3f} {z:<7.3f}'
                          , file=output_xyz)

    def __init__(self, filename, init_configuration=None, charge=None,
                 mult=None, box=None):
        super().__init__()

        if filename == 'geo_end.xyz':
            self.extract_from_dftb(init_config=init_configuration)

        if all(prm is not None for prm in (charge, mult, box)):
            self.load(filename, box=box, charge=charge, mult=mult)

        if len(self) == 0:
            logger.warning('Loaded an empty trajectory')
