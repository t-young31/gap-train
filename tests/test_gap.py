import numpy as np
from scipy.spatial import distance_matrix
import gaptrain as gt
from gaptrain.ase_calculators import expanded_atoms
import os


here = os.path.abspath(os.path.dirname(__file__))
h2o = gt.Molecule(os.path.join(here, 'data', 'h2o.xyz'))
methane = gt.Molecule(os.path.join(here, 'data', 'methane.xyz'))


def _test_gap():

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


def _test_gap_train():

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


def test_intra_gap1():

    system = gt.System(box_size=[10, 10, 10])
    system.add_solvent('h2o', n=3)

    gap = gt.gap.IntraGAP(name='test',
                          unique_molecule=system.unique_molecules[0])
    assert gap.mol_idxs == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    ase_atoms = system.random().ase_atoms()
    new_atoms = expanded_atoms(ase_atoms,
                               expansion_factor=10,
                               mol_idxs=np.array(gap.mol_idxs, dtype=int))
    coords = new_atoms.positions
    for i, coord in enumerate(coords):
        if i < 3:
            # Bonded atoms to the first atom should be close by
            assert np.linalg.norm(coords[0] - coord) < 3
        else:
            # while the non-bonded counterparts should all be far away
            assert np.linalg.norm(coords[0] - coord) > 10


def test_intra_gap2():

    system = gt.System(box_size=[10, 10, 10])
    system.add_solvent('h2o', n=3)
    system.add_molecules(methane, n=1)

    gap = gt.gap.IntraGAP(name='methane',
                          unique_molecule=system.unique_molecules[1])
    assert gap.mol_idxs == [list(range(9, 14))]

    ase_atoms = system.random().ase_atoms()
    new_atoms = expanded_atoms(ase_atoms,
                               expansion_factor=10,
                               mol_idxs=np.array(gap.mol_idxs, dtype=int))
    dist_mat = distance_matrix(new_atoms.positions,
                               new_atoms.positions)

    # Far away non-bonded atoms and close bonded atoms
    assert np.min(dist_mat[-1, :][:-5]) > 10
    assert np.min(dist_mat[-1, :][-5:]) < 3
