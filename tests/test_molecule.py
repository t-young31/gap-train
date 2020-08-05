from gaptrain.molecules import Molecule
import os

here = os.path.abspath(os.path.dirname(__file__))


def test_mol():

    h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))
    assert h2o.n_atoms == 3
    assert set([atom.label for atom in h2o.atoms]) == {'O', 'H'}
