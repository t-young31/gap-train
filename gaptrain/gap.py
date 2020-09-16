from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
from gaptrain.plotting import correlation
from gaptrain.exceptions import GAPFailed
from gaptrain.calculators import run_gap
from autode.atoms import elements
from subprocess import Popen, PIPE
from itertools import combinations
from multiprocessing import Pool
import numpy as np
import pickle
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

        # Reference energy and forces labels and don't separate xml files
        params += ('} energy_parameter_name=dft_energy '
                   'force_parameter_name=dft_forces '
                   'sparse_separate_file=F')

        # GAP needs the training data, some parameters and a file to save to
        return [f'at_file={self.training_data.name}.xyz', params,
                f'gp_file={self.name}.xml']

    def plot_correlation(self, true_data, predicted_data, rel_energies=True):
        """
        Plot predicted vs. true for a GAP predicted values

        :param true_data: (gaptrain.data.Data)
        :param predicted_data: (gaptrain.data.Data)
        :param rel_energies: (bool)
        """
        true_energies = true_data.energies()
        pred_energies = predicted_data.energies()

        # If required calculate the energies relative to the lowest true value
        if rel_energies:
            min_energy = min(true_energies)
            true_energies -= min_energy
            pred_energies -= min_energy

        # Plot the correlation for energies and forces
        correlation(true_energies,
                    pred_energies,
                    true_data.force_components(),
                    predicted_data.force_components(),
                    name=self.name)

        return None

    def predict(self, data):
        """
        Predict energies and forces for a set of data

        :param data: (gaptrain.data.Data)
        """

        # Run GAP in parallel to predict energies and forces
        predictions = data.copy()
        predictions.name += '_pred'

        predictions.parallel_gap(self)

        self.plot_correlation(data, predictions, rel_energies=True)
        predictions.save(override=True)

        return predictions

    def train(self, data):
        """
        Train this GAP on some data

        :param data: (gaptrain.data.Data)
        """
        assert all(config.energy is not None for config in data)

        logger.info('Training a Gaussian Approximation potential on '
                    f'*{len(data)}* training data points')

        start_time = time()

        self.training_data = data
        self.training_data.save()

        # Run the training using a specified number of total cores
        os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)

        p = Popen(GTConfig.gap_fit_command + self.train_command(),
                  shell=False, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()

        delta_time = time() - start_time
        print(f'GAP training ran in {delta_time/60:.1f} m')

        if delta_time < 0.05 or b'SYSTEM ABORT' in err:
            raise GAPFailed(f'GAP train errored with:\n '
                            f'{err.decode()}\n'
                            f'{" ".join(self.train_command())}')
        return None

    def save_params(self):
        """Save the parameters used in this GAP"""
        return pickle.dump(self.params,
                           file=open(f'{self.name}.gap', 'wb'))

    def load_params(self):
        """Load the parameters used"""
        self.params = pickle.load(open(f'{self.name}.gap', 'rb'))
        return None

    def __init__(self, name, system):
        """A Gaussian Approximation Potential"""

        self.name = name
        self.params = Parameters(atom_symbols=set(system.atom_symbols()))
        self.training_data = None


class Parameters:

    def n_sparses(self, inc_soap=True, inc_two_body=True):
        """
        Return a list of all the n_sparse parameters used to generate a GAP

        :param inc_soap: (bool)
        :param inc_two_body: (bool)
        :return: (list(int))
        """
        ns = []

        if inc_soap:
            ns += [at_soap['n_sparse'] for at_soap in self.soap.values()]

        if inc_two_body:
            ns += [params['n_sparse'] for params in self.pairwise.values()]

        return ns

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


class GAPEnsemble:

    def n_gaps(self):
        return len(self.gaps)

    def predict_energy_error(self, *args):
        """
        Predict the standard deviation between predicted energies

        -----------------------------------------------------------------------
        :param args: (gaptrain.configurations.Configuration |
                      gaptrain.configurations.ConfigurationSet)

        :return: (float | list(float)) Error on each of the configurations
        """
        configs = []

        # Populate a list of all the configurations that need to be
        for arg in args:
            try:
                for config in arg:
                    configs.append(config)

            except TypeError:
                assert arg.__class__.__name__ == 'Configuration'
                configs.append(arg)

        assert len(configs) != 0

        start_time = time()
        results = np.empty(shape=(len(configs), self.n_gaps()), dtype=object)
        predictions = np.empty(shape=results.shape)

        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MLK_NUM_THREADS'] = '1'
        logger.info('Set OMP and MLK threads to 1')

        with Pool(processes=GTConfig.n_cores) as pool:

            # Apply the method to each configuration in this set
            for i, config in enumerate(configs):
                for j, gap in enumerate(self.gaps):
                    result = pool.apply_async(func=run_gap,
                                              args=(config, None, gap))
                    results[i, j] = result

            # Reset all the configurations in this set with updated energy
            # and forces (each with .true)
            for i in range(len(configs)):
                for j in range(self.n_gaps()):
                    predictions[i, j] = results[i, j].get(timeout=None).energy

        logger.info(f'Calculations done in {(time() - start_time):.1f} s')

        if len(configs) == 1:
            return np.std(predictions[0])

        print(predictions)

        return [np.std(predictions[i, :]) for i in range(len(configs))]

    def train(self, data):
        """
        Train the ensemble of GAPS

        :param data: (gaptrain.data.Data)
        """
        logger.info(f'Training an ensemble with a total of {len(data)} '
                    'configurations')

        for i, gap in enumerate(self.gaps):

            sub_sampled_data = data.copy()

            # Remove points randomly from the training data to give an n-th
            n_data = int(len(data)/self.n_gaps())

            if n_data == 0:
                raise RuntimeError('Insufficient configurations to sub-sample')

            if any(n_sparse > n_data for n_sparse in gap.params.n_sparses()):
                raise RuntimeError('Number of sub-sampled data must be greater'
                                   ' than or equal to the number of sparse '
                                   'points')

            sub_sampled_data.remove_random(remainder=n_data)
            sub_sampled_data.name += f'ss_{i}'

            gap.train(data=sub_sampled_data)

        return None

    def __init__(self, name, system, n=5):
        """
        Ensemble of Gaussian approximation potentials allowing for error
        estimates by sub-sampling

        :param name:
        :param system:
        """
        logger.info(f'Initialising a GAP ensemble with {int(n)} GAPs')

        self.gaps = [GAP(f'{name}_{i}', system) for i in range(int(n))]
