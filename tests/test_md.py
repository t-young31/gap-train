import pytest
import gaptrain as gt
from gaptrain.systems import MMSystem
from gaptrain.md import run_mmmd, run_gapmd
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

    bbond_energy = {(1, 2): 0.1}
    fbond_energy = {(1, 2): 0.1}

    momenta_string = gt.md.ase_momenta_string(configuration, 300, bbond_energy,
                                              fbond_energy)

    assert type(momenta_string) is str

    return None


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
