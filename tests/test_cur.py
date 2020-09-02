from gaptrain.cur import rows
import numpy as np


def test_cur():

    # Check several times that the CUR matrix is the correct size for
    # different random input
    for _ in range(100):

        matrix = np.random.random(size=(10, 2))

        cur_rows_mat = rows(matrix)
        n_rows, n_cols = cur_rows_mat.shape

        assert 0 < n_rows < 10
        assert n_cols == 2
