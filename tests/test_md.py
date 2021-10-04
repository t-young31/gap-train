import numpy as np
import pytest
import gaptrain as gt
from gaptrain.systems import MMSystem
from gaptrain.md import run_mmmd, run_gapmd, set_momenta
from gaptrain.molecules import Ion
from autode.species import Molecule
from gaptrain.utils import work_in_tmp_dir
import os

here = os.path.abspath(os.path.dirname(__file__))


def test_run_mmmd():
    """Tests running MD as well as converting from go to xyz."""

    itp_filepath = os.path.join(here, 'data', 'zn.itp')

    water_box = MMSystem(box_size=[12, 12, 12])
    water_box.add_molecules(Ion('Zn', charge=2,
                                gmx_itp_filename=itp_filepath), n=1)
    water_box.add_solvent('h2o', n=52)

    config = water_box.random()

    if 'GT_GMX' not in os.environ or not os.environ['GT_GMX'] == 'True':
        return

    run_mmmd(water_box, config, temp=300, dt=1, interval=100, ps=10)

    return None


def test_ase_momenta_string():

    system = gt.System(box_size=[10, 10, 10])
    system.add_solvent('h2o', n=1)

    configuration = system.configuration()

    bbond_energy = {(1, 2): 10}
    ase_atoms = configuration.ase_atoms()
    set_momenta(configuration,
                ase_atoms,
                temp=100,
                bbond_energy=bbond_energy,
                fbond_energy={})

    momenta = ase_atoms.get_momenta()
    assert np.linalg.norm(momenta[1, :]) > np.linalg.norm(momenta[0, :])
    assert np.linalg.norm(momenta[2, :]) > np.linalg.norm(momenta[0, :])


@work_in_tmp_dir()
def test_run_gapmd_no_box():

    h2o = Molecule(smiles='O')
    h2o.print_xyz_file(filename='tmp.xyz')

    with open('some_params.xml', 'w') as xml_file:
        print('not a real xml', file=xml_file)

    # Without a box defined for the configuration dynamics can't be run
    with pytest.raises(ValueError):
        _ = run_gapmd(configuration=gt.Configuration('tmp.xyz'),
                      gap=gt.GAP('some_params.xml'),
                      temp=298.15,
                      dt=0.5,
                      interval=1)
