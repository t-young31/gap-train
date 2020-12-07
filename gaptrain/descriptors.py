from collections.abc import Iterable
from gaptrain.log import logger
import numpy as np


def soap(*args):
    """
    Create a SOAP vector using dscribe (https://github.com/SINGROUP/dscribe)
    for a set of configurations

    soap(config)           -> [[v0, v1, ..]]
    soap(config1, config2) -> [[v0, v1, ..], [u0, u1, ..]]
    soap(configset)        -> [[v0, v1, ..], ..]

    ---------------------------------------------------------------------------
    :param args: (gaptrain.configurations.Configuration) or
                 (gaptrain.configurations.ConfigurationSet)

    :return: (np.ndarray) shape = (len(args), n) where n is the length of the
             SOAP descriptor
    """
    from dscribe.descriptors import SOAP

    configurations = args

    # If a configuration set is specified then use that as the list of configs
    if len(args) == 1 and isinstance(args[0], Iterable):
        configurations = args[0]

    logger.info(f'Calculating SOAP descriptor for {len(configurations)}'
                f' configurations')

    unique_elements = list(set(atom.label for atom in configurations[0].atoms))

    # Compute the average SOAP vector where the expansion coefficients are
    # calculated over averages over each site
    soap_desc = SOAP(species=unique_elements,
                     rcut=5,             # Distance cutoff (Å)
                     nmax=6,             # Maximum component of the radials
                     lmax=6,             # Maximum component of the angular
                     average='inner')

    soap_vec = soap_desc.create([conf.ase_atoms() for conf in configurations])

    logger.info('SOAP calculation done')
    return soap_vec


def soap_kernel_matrix(configs, zeta=4):
    """
    Calculate the kernel matrix between a set of configurations where the
    kernel is

    K(p_a, p_b) = (p_a . p_b / (p_a.p_a x p_b.p_b)^1/2 )^ζ

    :param configs: (gaptrain.configurations.ConfigurationSet)
    :param zeta: (float) Power to raise the kernel matrix to
    :return: (np.ndarray) shape = (len(configs), len(configs))
    """
    soap_vecs = soap(configs)
    n, _ = soap_vecs.shape

    # Normalise each soap vector (row)
    soap_vecs = soap_vecs / np.linalg.norm(soap_vecs, axis=1).reshape(n, 1)

    k_mat = np.matmul(soap_vecs, soap_vecs.T)
    k_mat = np.power(k_mat, zeta)
    return k_mat
