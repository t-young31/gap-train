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


def rows(matrix, k=2, epsilon=1, return_indexes=False):
    """
    Get the most leveraging rows of a matrix using the CUR algorithm.
    WARNING: This function is not deterministic in the shape of the output
    for a (n, m) input matrix a (o, m) matrix is returned, where o varies based
    on the probability given k and epsilon

    ---------------------------------------------------------------------------
    :param matrix: (np.ndarray) shape = (n, m)

    :param k: (int) Rank of the SVD decomposition

    :param epsilon: (float) Error parameter, smaller values will generate
                    better approximations but

    :param return_indexes: (bool) Only return the indexes of the rows in the
                           matrix
    :return:
    """
    logger.info('Calculating partial CUR decomposition for rows on'
                f'a matrix with dimensions {matrix.shape}')

    if k < 1:
        raise ValueError('Rank must be at least 1')

    # Requesting the rows so run COLUMNSELECT on the transpose on the matrix
    A = matrix.T
    m, n = A.shape

    # Approximate number of rows to take from A
    c = k * np.log(k) / epsilon ** 2

    # Singular value decomposition of the matrix A:
    # A = u^T s v
    u, s, v = svd(A)

    # Compute the pi matrix of probabilities
    pi = (1.0 / k) * np.sum(v[:int(k), :] ** 2, axis=0)

    # COLUMNSELECT from Mahoney & Drineas
    rows_list = []
    indexes_list = []

    for j in range(n):

        # Accept this column with probability min{1,cπ_j}
        if np.random.uniform(0, 1) < min(1, c * pi[j]):
            rows_list.append(A[:, j])
            indexes_list.append(j)

    if return_indexes:
        logger.info('Returning the indexes of the leveraging rows')
        return indexes_list

    # R matrix has ~c rows which are a the most significant leveraging rows
    R = np.array(rows_list)

    if R.shape[0] == 0:
        # Return at least one row of the matrix...
        return rows(matrix, k, epsilon)

    logger.info(f'Returning the matrix R with dimension {R.shape}')
    return R


if __name__ == '__main__':

    print(rows(matrix=np.random.random(size=(10, 5)),
               k=2,
               epsilon=1))
