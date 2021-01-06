import gaptrain as gt
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = gt.Molecule(os.path.join(here, 'data', 'h2o.xyz'))
methane = gt.Molecule(os.path.join(here, 'data', 'methane.xyz'))


def test_gap():

    water_dimer = gt.System(box_size=[3.0, 3.0, 3.0])
    water_dimer.add_molecules(h2o, n=2)

    gap = gt.GAP(name='test', system=water_dimer)

    assert hasattr(gap, 'name')
    assert hasattr(gap, 'params')
    assert gap.training_data is None

    assert hasattr(gap.params, 'general')
    assert hasattr(gap.params, 'pairwise')
    assert hasattr(gap.params, 'soap')

    # By default should only add a SOAP to non-hydrogen elements
    assert 'O' in gap.params.soap.keys()
    assert len(list(gap.params.soap)) == 1


def test_gap_train():

    system = gt.System(box_size=[10, 10, 10])

    training_data = gt.Data(name='test')
    training_data.load(system=system,
                       filename=os.path.join(here, 'data', 'rnd_training.xyz'))

    assert len(training_data) == 10
    assert len(training_data[0].atoms) == 31

    if 'GT_GAP' not in os.environ or not os.environ['GT_GAP'] == 'True':
        return

    # Run GAP train with the training data
    gap = gt.GAP(name='test', system=system)
    gap.train(training_data)


def test_intra_gap():

    system = gt.System(box_size=[10, 10, 10])
    system.add_solvent('h2o', n=3)

    gap = gt.gap.SolventIntraGAP(name='test', system=system)
    assert gap.mol_idxs == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    system.add_molecules(methane, n=1)
    gap = gt.gap.SoluteIntraGAP(name='test', system=system, molecule=methane)
    assert gap.mol_idxs == [[9, 10, 11, 12, 13]]
