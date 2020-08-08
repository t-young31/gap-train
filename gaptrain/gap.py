from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
from autode.atoms import elements
from subprocess import Popen, PIPE
from itertools import combinations
from time import time
import os


def atomic_number(symbol):
    """Elements in order indexed from 0 so the atomic number is the index+1"""
    return elements.index(symbol) + 1


class GAP:

    def train_command(self):
        """Generate the teach_sparse function call for this system of atoms"""

        general = self.params.general
        params = ('default_sigma={'
                  f'{general["sigma_E"]:.6f} {general["sigma_F"]:.6f} 0.0 0.0'
                  '} gap={')

        # Iterate thorough the dictionary of two-body pairwise parameters
        for (symbol_a, symbol_b), pairwise in self.params.pairwise.items():
            logger.info(f'Adding two-body pairwise: {symbol_a}-{symbol_b}')

            params += ('distance_2b '
                       f'delta={pairwise["delta"]} '
                       'covariance_type=ard_se '
                       f'n_sparse={pairwise["n_sparse"]} '
                       f'cutoff={pairwise["cutoff"]} '
                       'theta_uniform=1.0 '
                       'sparse_method=uniform '
                       'n_Z=1 '
                       'n_species=1 '
                       'species_Z={{'
                       f'{atomic_number(symbol_a)}'
                       '}} '
                       f'Z={atomic_number(symbol_b)}: ')

        # Likewise with all the SOAPs to be added
        for symbol, soap in self.params.soap.items():
            logger.info(f'Adding SOAP:              {symbol}')
            other_atomic_ns = [atomic_number(s) for s in soap["other"]]

            params += ('soap sparse_method=cur_points '
                       f'n_sparse={soap["n_sparse"]} '
                       f'covariance_type=dot_product '
                       f'zeta=4 '
                       f'atom_sigma=0.5 '
                       f'cutoff={soap["cutoff"]} '
                       f'delta={soap["delta"]} '
                       f'n_Z=1 '
                       f'n_species={len(soap["other"])} '
                       'species_Z={{'
                       # Remove the brackets from the ends of the list
                       f'{str(other_atomic_ns)[1:-1]}'
                       '}} '
                       f'Z={atomic_number(symbol)} '
                       f'n_max={soap["order"]} '
                       f'l_max={soap["order"]}: ')

        # Remove the final unnecessary colon
        params = params.rstrip(': ')

        # Reference energy and forces labels  and don't separate xml files
        params += ('} energy_parameter_name=dft_energy '
                   'force_parameter_name=dft_forces '
                   'sparse_separate_file=F')

        # GAP needs the training data, some parameters and a file to save to
        return [f'at_file={self.training_data.name}.xyz', params,
                f'gp_file={self.name}.xml']

    def predict(self, data, plot_energy=True, plot_force=True):
        """
        Predict energies and forces for a set of data

        -----------------------------------------------------------------------
        :param data: (gaptrain.data.Data)

        :param plot_energy: (bool) Plot an energy correlation: predicted v true

        :param plot_force: (bool) Plot a force correlation
        """
        data.run_gap(self)
        raise NotImplementedError

    def train(self, data):
        """
        Train this GAP on some data

        :param data: (gaptrain.data.Data)
        """
        logger.info('Training a Gaussian Approximation potential on '
                    f'*{len(data)}* training data points')

        start_time = time()

        self.training_data = data
        self.training_data.save_true()

        # Run the training using a specified number of total cores
        os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)
        p = Popen(GTConfig.gap_fit_command + self.train_command(),
                  shell=False, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()

        delta_time = time() - start_time
        print(f'GAP training ran in {delta_time/60:.1f} m')

        if delta_time < 0.5 or b'SYSTEM ABORT' in err:
            raise Exception(f'GAP train errored with:\n '
                            f'{err.decode()}\n'
                            f'{" ".join(self.train_command())}')
        return None

    def save(self):
        raise NotImplementedError

    def __init__(self, name, system):
        """A Gaussian Approximation Potential"""

        self.name = name
        self.params = Parameters(atom_symbols=set(system.atom_symbols()))
        self.training_data = None


class Parameters:

    def __init__(self, atom_symbols):
        """
        Parameters for a GAP potential

        :param atom_symbols: (set(str)) Unique atomic symbols used to
                             to generate parameters for a GAP potential
        """

        self.general = GTConfig.gap_default_params

        self.pairwise = {pair: GTConfig.gap_default_2b_params
                         for pair in combinations(atom_symbols, r=2)}

        self.soap = {}

        for symbol in atom_symbols:
            params = GTConfig.gap_default_soap_params.copy()

            # Add all the atomic symbols that aren't this one
            params["other"] = [s for s in atom_symbols if s != symbol]

            self.soap[symbol] = params

        if 'H' in self.soap:
            logger.warning('H found in SOAP descriptor  - removing')
            self.soap.pop('H')
