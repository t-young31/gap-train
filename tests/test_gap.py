from gaptrain.gap import GAP, IntraGAP
from gaptrain.molecules import Molecule
from gaptrain.systems import System
from gaptrain.data import Data
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_gap():

    water_dimer = System(box_size=[3.0, 3.0, 3.0])
    water_dimer.add_molecules(h2o, n=2)

    gap = GAP(name='test', system=water_dimer)

    assert hasattr(gap, 'name')
    assert hasattr(gap, 'params')
    assert gap.training_data is None

    assert hasattr(gap.params, 'general')
    assert hasattr(gap.params, 'pairwise')
    assert hasattr(gap.params, 'soap')

    # Should only have three pairwise descriptors
    assert (('H', 'O') in gap.params.pairwise.keys() or
            ('O', 'H') in gap.params.pairwise.keys())
    assert len(list(gap.params.pairwise)) == 3

    # By default should only add a SOAP to non-hydrogen elements
    assert 'O' in gap.params.soap.keys()
    assert len(list(gap.params.soap)) == 1


def test_gap_train():

    system = System(box_size=[10, 10, 10])

    training_data = Data(name='test')
    training_data.load(system=system,
                       filename=os.path.join(here, 'data', 'rnd_training.xyz'))

    assert len(training_data) == 10
    assert len(training_data[0].atoms) == 31

    if 'GT_GAP' not in os.environ or not os.environ['GT_GAP'] == 'True':
        return

    # Run GAP train with the training data
    gap = GAP(name='test', system=system)
    gap.train(training_data)


def test_intra_gap():

    system = System(box_size=[10, 10, 10])
    system.add_solvent('h2o', n=3)

    gap = IntraGAP(name='test', system=system)
    assert gap.mol_idxs == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
