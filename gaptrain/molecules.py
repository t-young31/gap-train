from autode.species import Species
from autode.input_output import xyz_file_to_atoms
from autode.atoms import Atom
from gaptrain.log import logger


class Molecule(Species):

    def __init__(self,  xyz_filename,  charge=0, spin_multiplicity=1,
                 gmx_itp_filename=None):
        """Molecule e.g. H2O

        -----------------------------------------------------------------------
        :param xyz_filename: (str)

        :param charge: (int)

        :param spin_multiplicity: (int)

        :param gmx_itp_filename: (str) Filename(path) of the GROMACS .itp file
                                 containing MM parameters required to simulate
        """
        super().__init__(name='mol',
                         charge=charge,
                         mult=spin_multiplicity,
                         atoms=xyz_file_to_atoms(xyz_filename))

        self.itp_filename = gmx_itp_filename

        logger.info(f'Initialised {xyz_filename.rstrip(".xyz")}\n'
                    f'Number of atoms      = {self.n_atoms}\n'
                    f'GROMACS itp filename = {self.itp_filename}')


class Ion(Species):

    def __init__(self, label, charge, spin_multiplicity=1,
                 gmx_itp_filename=None):
        """Ion

        -----------------------------------------------------------------------
        :param label: (str) e.g. 'Pd'

        :param charge: (int)

        :param spin_multiplicity: (int)
        """
        super().__init__(name=label,
                         charge=charge,
                         mult=spin_multiplicity,
                         atoms=[Atom(label)])

        self.itp_filename = gmx_itp_filename
