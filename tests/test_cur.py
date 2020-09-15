from gaptrain.cur import rows
import gaptrain as gt
import numpy as np
import os

here = os.path.abspath(os.path.dirname(__file__))
h2o = gt.Molecule(os.path.join(here, 'data', 'h2o.xyz'))


def test_cur():

    # Check several times that the CUR matrix is the correct size for
    # different random input
    for _ in range(100):

        matrix = np.random.random(size=(10, 2))

        cur_rows_mat = rows(matrix)
        n_rows, n_cols = cur_rows_mat.shape

        assert 0 < n_rows < 10
        assert n_cols == 2


def test_cur_soap_matrix():
    """Test CUR decomposition on a matrix of SOAP vectors to select the
    most different configurations"""

    h2o_dimer = gt.System(box_size=[10, 10, 10])
    h2o_dimer.add_molecules(h2o, n=2)

    config0 = h2o_dimer.random()

    # Configuration 2 is very similar to configuration 1 – only a small
    # perturbation to the coordinates
    config1 = config0.copy()
    # config1.add_perturbation(sigma=0.01)

    # Configuration 3 is (usually) much more different
    config2 = config0.copy()
    config2.add_perturbation(sigma=2.0)

    soap_matrix = gt.descriptors.soap(config0, config1, config2)

    # The similar configurations 0 and 1 should not form the complete set
    # of indexes i.e. both be chosen
    for _ in range(100):
        idxs = rows(soap_matrix, return_indexes=True)
        assert list(idxs) != [0, 1]
        assert list(idxs) != [1, 0]
