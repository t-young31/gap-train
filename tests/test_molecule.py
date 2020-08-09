from gaptrain.molecules import Molecule, Ion
import os

here = os.path.abspath(os.path.dirname(__file__))


def test_mol():

    h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))
    assert h2o.n_atoms == 3
    assert set([atom.label for atom in h2o.atoms]) == {'O', 'H'}


def test_ion():

    na = Ion('Na', charge=1)

    assert na.n_atoms == 1
    assert na.charge == 1
    assert na.mult == 1

    # By default VdW radii are not added to the molecule's size
    assert na.radius == 0.0
