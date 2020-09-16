"""
CUR matrix decompostion based on:

CUR matrix decompositions for improved data analysis
Michael W. Mahoney, Petros Drineas
Proceedings of the National Academy of Sciences Jan 2009, 106 (3) 697-702;
DOI: 10.1073/pnas.0803205106
"""
from gaptrain.log import logger
import numpy as np
from scipy.linalg import svd


def rows(matrix, k=2, epsilon=1, return_indexes=False, n_iters=100):
    """
    Get the most leveraging rows of a matrix using the CUR algorithm

    ---------------------------------------------------------------------------
    :param matrix: (np.ndarray) shape = (n, m)

    :param k: (int) Rank of the SVD decomposition

    :param epsilon: (float) Error parameter, smaller values will generate
                    better approximations

    :param return_indexes: (bool) Only return the indexes of the rows in the
                           matrix

    :param n_iters: (int) Number of iterations to calculate whether a row is
                    chosen from the list of rows in the matrix
    :return:
    """
    logger.info('Calculating partial CUR decomposition for rows on '
                f'a matrix with dimensions {matrix.shape}')

    if k < 1:
        raise ValueError('Rank must be at least 1')

    # Requesting the rows so run COLUMNSELECT on the transpose on the matrix
    A = matrix.T
    m, n = A.shape

    if k > n:
        raise ValueError(f'Cannot find {k} rows in a matrix with only {n} rows')

    # Approximate number of rows to take from A
    c = k * np.log(k) / epsilon ** 2

    # Singular value decomposition of the matrix A:
    # A = u^T s v
    u, s, v = svd(A)

    # Compute the pi matrix of probabilities
    pi = (1.0 / k) * np.sum(v[:int(k), :] ** 2, axis=0)

    # COLUMNSELECT algorithm from Mahoney & Drineas----

    # Dictionary of row indexes and the number of times they are selected by
    # the CUR decomposition
    rows_p = {j: 0 for j in range(n)}

    for _ in range(n_iters):
        for j in range(n):

            # Accept this column with probability min{1,cπ_j}
            if np.random.uniform(0, 1) < min(1, c * pi[j]):
                rows_p[j] += 1

    # List of row indexes from most significant -> least significant will
    # have the most chance of being selected i.e a large value rows_p[j]
    row_indexes = [j for j in sorted(rows_p, key=rows_p.get)][::-1]

    if return_indexes:
        logger.info(f'Returning the indexes of the {k} most leveraging rows')
        return row_indexes[:k]

    R = np.array([A[:, j] for j in row_indexes[:k]])

    logger.info(f'Returning the matrix R with dimension {R.shape}')
    return R
