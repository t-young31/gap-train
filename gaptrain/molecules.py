from autode.input_output import xyz_file_to_atoms
from autode.species import Species
from gaptrain.log import logger


class Molecule(Species):

    def __init__(self, xyz_filename, gmx_itp_filename=None):
        """Molecule e.g. H2O


        :param xyz_filename: (str)
        """
        assert xyz_filename.endswith('.xyz')

        super().__init__(name='mol',
                         charge=0,
                         mult=1,
                         atoms=xyz_file_to_atoms(xyz_filename))

        self.itp_filename = gmx_itp_filename

        logger.info(f'Initialised {xyz_filename.rstrip(".xyz")}\n'
                    f'Number of atoms      = {self.n_atoms}\n'
                    f'GROMACS itp filename = {self.itp_filename}')
