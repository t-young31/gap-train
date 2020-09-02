from collections.abc import Iterable
from gaptrain.log import logger


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
