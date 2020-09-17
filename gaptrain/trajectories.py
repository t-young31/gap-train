from gaptrain.log import logger
import gaptrain as gt
import ase.io.trajectory as ase_traj
import os


def gro2xyz(gro_traj, config):
    """Convert a GROMACS .gro trajectory to a .xyz trajectory"""
    with open('nvt_traj.xyz', 'w') as output_xyz:

        lines = open(gro_traj, 'r').readlines()

        # Number of atoms is second line of gro file
        num_of_atoms = int(lines[1])
        stride = (num_of_atoms + 3)

        # Create an ordered list of atoms from system.molecules
        atom_list = [atom.label for atom in config.atoms]

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

        return None


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

    def extract_from_gmx(self, filename, init_config):
        """Load a GAP train trajectory from .gro file"""
        assert init_config is not None

        gro2xyz(filename, init_config)
        self.load(filename='nvt_traj.xyz', system=init_config)
        os.remove(filename)

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

        elif filename.endswith('.gro'):
            self.extract_from_gmx(filename, init_configuration)

        if len(self) == 0:
            logger.warning('Loaded an empty trajectory')
