import os
from gaptrain.trajectories import gro2xyz, Trajectory
from gaptrain.systems import System
from gaptrain.configurations import Configuration
from gaptrain.molecules import Molecule, Atom


here = os.path.abspath(os.path.dirname(__file__))


def test_gro_conversion():

    cwd = os.getcwd()
    os.chdir(os.path.join(here, 'data'))

    n2 = Molecule(atoms=[Atom('N'), Atom('N', z=1.0)])
    n2_config = System(n2, box_size=[5, 5, 5]).configuration()

    gro2xyz(filename='test.gro',
            config=n2_config,
            out_filename='test_gro_conv.xyz')

    assert os.path.exists('test_gro_conv.xyz')
    gen_config = Configuration(filename='test_gro_conv.xyz')

    assert len(gen_config.atoms) == 2
    assert gen_config.atoms[0].label == 'N'
    assert gen_config.atoms[1].label == 'N'
    assert gen_config.atoms[1].coord[0] == 37.76

    traj = Trajectory('test.gro', init_configuration=n2_config)
    assert len(traj) == 1

    os.remove('test_gro_conv.xyz')
    os.chdir(cwd)
