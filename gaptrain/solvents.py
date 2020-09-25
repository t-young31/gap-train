from gaptrain.molecules import Molecule
from gaptrain.log import logger
import os

here = os.path.abspath(os.path.dirname(__file__))
solvent_dir = os.path.join(here, 'solvent_lib')


class Solvent(Molecule):
    """Solvent e.g. H2O"""


def get_solvent(name):
    """Gets solvent molecule from solvent list"""
    for solvent in solvents:
        if solvent.name == name:
            return solvent
    return None


# Generate Solvent objects for all molecules in solvent_lib
solvents = []

for filename in os.listdir(solvent_dir):
    if not filename.endswith('.xyz'):
        continue

    itp_filename = filename.replace('.xyz', '.itp')
    itp_filepath = os.path.join(solvent_dir, itp_filename)

    if not os.path.exists(itp_filepath):
        logger.warning(f'Found solvent xyz file without associated '
                       f'itp {filename}')
        continue

    solvent = Solvent(xyz_filename=os.path.join(solvent_dir, filename),
                      gmx_itp_filename=itp_filepath)

    solvent.name = os.path.basename(itp_filepath.rstrip('.itp'))

    solvents.append(solvent)
