from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
from gaptrain.plotting import plot_ef_correlation
from gaptrain.exceptions import GAPFailed
from gaptrain.calculators import run_gap
from autode.atoms import elements
from subprocess import Popen, PIPE
from itertools import combinations_with_replacement
from multiprocessing import Pool
import numpy as np
import pickle
from time import time
import os


def atomic_number(symbol):
    """Elements in order indexed from 0 so the atomic number is the index+1"""
    return elements.index(symbol) + 1


def predict(gap, data):
    """
    Predict energies and forces for a set of data

    :param gap: (gaptrain.gap.GAP | gaptrain.gap.AdditiveGAP)
    :param data: (gaptrain.data.Data)
    """

    # Run GAP in parallel to predict energies and forces
    predictions = data.copy()
    predictions.name += '_pred'

    predictions.parallel_gap(gap)

    gap.plot_correlation(data, predictions, rel_energies=True)
    predictions.save(override=True)

    return predictions


class GAP:

    def ase_gap_potential_str(self):
        """Generate the quippy/ASE string to run the potential"""

        if not os.path.exists(f'{self.name}.xml'):
            raise IOError(f'GAP parameter file ({self.name}.xml) did not exist')

        return ('pot = quippy.Potential("IP GAP", \n'
                f'              param_filename="{self.name}.xml")')

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
                       'species_Z={{'
                       f'{atomic_number(symbol_a)}'
                       '}} '
                       f'Z={atomic_number(symbol_b)}: ')

            logger.info(f'2b term: {params}')

        for (symbol_a, symbol_b, symbol_c), angle in self.params.angle.items():
            logger.info(f'Adding angle (3b): {symbol_a}-{symbol_b}-{symbol_c}')

            params += ('angle_3b '
                       f'delta={angle["delta"]} '
                       'covariance_type=ard_se '
                       f'n_sparse={angle["n_sparse"]} '
                       f'cutoff={angle["cutoff"]} '
                       'theta_fac=1.0 '
                       'sparse_method=uniform '
                       'n_Z=1 '
                       'species_Z={{'
                       f'{atomic_number(symbol_b)}, {atomic_number(symbol_c)}' 
                       '}} '
                       f'Z={atomic_number(symbol_a)}: ')

        # Likewise with all the SOAPs to be added
        for symbol, soap in self.params.soap.items():
            logger.info(f'Adding SOAP:              {symbol}')
            other_atomic_ns = [atomic_number(s) for s in soap["other"]]
            logger.info(f'with neighbours           {soap["other"]}')

            params += ('soap sparse_method=cur_points '
                       f'n_sparse={soap["n_sparse"]} '
                       f'covariance_type=dot_product '
                       f'zeta=4 '
                       f'atom_sigma={soap["sigma_at"]} '
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
        return plot_ef_correlation(self.name, true_data, predicted_data,
                                   rel_energies=rel_energies)

    def predict(self, data):
        return predict(self, data)

    def train(self, data):
        """
        Train this GAP on some data

        :param data: (gaptrain.data.Data)
        """
        assert all(config.energy is not None for config in data)
        assert self.params is not None

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

    def __init__(self, name, system=None, default_params=True):
        """
        A Gaussian Approximation Potential

        :param name: (str)
        :param system: (gaptrain.systems.System | None)
        """

        self.name = name

        if system is not None and default_params:
            self.params = Parameters(atom_symbols=system.atom_symbols())

        else:
            self.params = Parameters(atom_symbols=[])
            logger.warning('Initialised a GAP with no parameters. '
                           'gap.train not available')

        self.training_data = None


class InterGAP(GAP):
    pass


class IntraGAP(GAP):

    def _set_mol_idxs(self, system):
        """Set the molecular indexes from a system as the most abundant
        identical molecules"""
        mols_and_num = {}
        for molecule in system.molecules:
            if str(molecule) not in mols_and_num:
                mols_and_num[str(molecule)] = 1
            else:
                mols_and_num[str(molecule)] += 1

        # Rudimentary sort..
        max_mol, max_num = None, 0
        for mol, num in mols_and_num.items():
            if num > max_num:
                max_mol = mol

        # Add the indexes of the most abundant molecules
        curr_n_atoms = 0

        for molecule in system.molecules:
            if str(molecule) == max_mol:
                idxs = range(curr_n_atoms, curr_n_atoms + molecule.n_atoms)
                self.mol_idxs.append(list(idxs))

            curr_n_atoms += molecule.n_atoms

        return None

    def __init__(self, name, system):
        """An intramolecular GAP, must be initialised with a system so the
        molecules are defined

        :param name: (str)
        :param system: (gt.system.System)
        """
        super().__init__(name, system)

        self.mol_idxs = []
        self._set_mol_idxs(system)
        print(self.mol_idxs)


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

    def _set_pairs(self, atom_symbols):
        """Set the two-body pair parameters"""

        for pair in combinations_with_replacement(set(atom_symbols), r=2):

            s_a, s_b = pair         # Atomic symbols of the pair
            if s_a == s_b and atom_symbols.count(s_a) == 1:
                logger.info(f'Only a single {s_b} atom not adding pairwise')
                continue

            self.pairwise[pair] = GTConfig.gap_default_2b_params

        return None

    def _set_soap(self, atom_symbols):
        """Set the SOAP parameters"""
        added_pairs = []

        for symbol in set(atom_symbols):

            if symbol == 'H':
                logger.warning('Not adding SOAP on H')
                continue

            params = GTConfig.gap_default_soap_params.copy()

            # Add all the atomic symbols that aren't this one, the neighbour
            # density for which also hasn't been added already
            params["other"] = [s for s in set(atom_symbols)
                               if s != symbol
                               and s+symbol not in added_pairs
                               and symbol+s not in added_pairs]

            for other_symbol in params["other"]:
                added_pairs.append(symbol+other_symbol)

            if len(params["other"]) == 0:
                logger.info(f'Not adding SOAP to {symbol} - should be covered')
                continue

            self.soap[symbol] = params

        return None

    def __init__(self, atom_symbols):
        """
        Parameters for a GAP potential

        :param atom_symbols: (list(str)) Atomic symbols used to to generate
        parameters for a GAP potential
        """

        self.general = GTConfig.gap_default_params

        self.pairwise = {}
        self.angle = {}
        self.soap = {}
        self._set_soap(atom_symbols)


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

        return [np.std(predictions[i, :]) for i in range(len(configs))]

    def sub_sampled_data(self, data, gap=None, random=True):
        """
        Select a portion of the data to train a GAP on as an nth of the full
        training data where n is the number of GAPs in this ensemble

        :param data: (gaptrain.data.Data)
        :param gap: (gaptrain.gap.GAP | None)
        :param random: (bool) Whether to take a random sample
        :return:
        """
        sub_sampled_data = data.copy()

        # Remove points randomly from the training data to give an n-th
        n_data = int(len(data) / self.n_gaps())

        if n_data == 0:
            raise RuntimeError('Insufficient configurations to sub-sample')

        if gap is not None:
            if any(n_sparse > n_data for n_sparse in gap.params.n_sparses()):
                raise RuntimeError('Number of sub-sampled data must be greater'
                                   ' than or equal to the number of sparse '
                                   'points')
        else:
            logger.warning('Cannot check that the number of data is larger'
                           'than the number of sparse points')

        if random:
            sub_sampled_data.remove_random(remainder=n_data)
        else:
            raise NotImplementedError

        return sub_sampled_data

    def train(self, data, sub_sample=True):
        """
        Train the ensemble of GAPS

        :param data: (gaptrain.data.Data)
        :param sub_sample: (bool) Should the data be sub sampled or the full
                           set used?
        """
        logger.info(f'Training an ensemble with a total of {len(data)} '
                    'configurations')

        for i, gap in enumerate(self.gaps):

            if sub_sample:
                training_data = self.sub_sampled_data(data, gap, random=True)
            else:
                training_data = data.copy()

            # Ensure that the data's name is unique, for saving etc.
            training_data.name += f's{i}'

            # Train the GAP
            gap.train(data=training_data)

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


class AdditiveGAP:
    """GAP where the energy is a sum of terms"""

    def ase_gap_potential_str(self):
        """Generate the quippy/ASE string to run the potential"""

        pt_str = ''
        for i in range(2):
            pt_str += (f'pot{i+1} = quippy.Potential("IP GAP", \n'
                       f'          param_filename="{self[i].name}.xml")\n')

            if not os.path.exists(f'{self[i].name}.xml'):
                raise IOError(f'GAP parameter file ({self[i].name}.xml) in '
                              f'additiive GAP did not exist')

        pt_str += f'pot = quippy.Potential("Sum", pot1=pot1, pot2=pot2)'

        return pt_str

    def predict(self, data):
        return predict(self, data)

    def plot_correlation(self, true_data, predicted_data, rel_energies=True):
        return plot_ef_correlation(f'{self._list[0].name}+{self._list[1].name}',
                                   true_data,
                                   predicted_data,
                                   rel_energies=rel_energies)

    def __getitem__(self, item):
        return self._list[item]

    def __init__(self, gap1, gap2):
        """
        Additive GAP

        :param gap1:
        :param gap2:
        """

        self.name = f'{gap1.name}_{gap2.name}'
        self._list = [gap1, gap2]


class IIGAP:
    """Inter+intra GAP where the inter is evaluated in a different box"""

    def ase_gap_potential_str(self,
                              calc_str='IICalculator(intra_gap, inter_gap)'):
        """Generate the quippy/ASE string to run the potential"""

        if not (os.path.exists(f'{self.inter.name}.xml')
                and os.path.exists(f'{self.intra.name}.xml')):
            raise IOError(f'GAP parameter files did not exist')

        # Custom calculator to calculate the intra component of the energy in
        # a larger box; in a file for neatness; first line is numpy import
        here = os.path.abspath(os.path.dirname(__file__))
        pt = open(os.path.join(here, 'iicalculator.py'), 'r').readlines()

        pt += [f'inter_gap = quippy.Potential("IP GAP", '
               f'param_filename="{self.inter.name}.xml")\n',
               f'intra_gap = quippy.Potential("IP GAP", '
               f'param_filename="{self.intra.name}.xml")\n',
               f'intra_gap.mol_idxs = {self.intra.mol_idxs}\n',
               f'pot = {calc_str}\n']

        return ''.join(pt)

    def __init__(self, *args):
        """
        Collective GAP comprised of inter and intra components

        :param args: (gt.gap.GAP)
        """

        self.inter = None
        self.intra = None

        for arg in args:
            if isinstance(arg, IntraGAP):
                self.intra = arg

            elif isinstance(arg, InterGAP):
                self.inter = arg

            else:
                raise ValueError('IIGAP must be initialised with only '
                                 'InterGAP or IntraGAP potentials')

        if self.inter is None or self.intra is None:
            raise AssertionError('Must have both an inter+intra GAP')


class SSGAP(IIGAP):

    def ase_gap_potential_str(self, calc_str=None):
        """Generate the quippy/ASE potential string with three calculators"""

        if calc_str is None:
            calc_str = 'SSCalculator(solute_gap, intra_gap, inter_gap)'

        pt = ('solute_gap = quippy.Potential("IP GAP", '
              f'     param_filename="{self.solute_intra.name}.xml")\n'
              f'solute_gap.mol_idxs = {self.solute_intra.mol_idxs}\n')

        pt += super().ase_gap_potential_str(calc_str=calc_str)
        return pt

    def __init__(self, solute_intra, solvent_intra, inter):
        """
        Solute-solvent (SS) Gaussian Approximation Potential comprised of a
        GAP for thr gas phase solute, the solvent and the remainder of the
        intermolecular interactions
        """
        super().__init__(solvent_intra, inter)

        self.solute_intra = solute_intra
        assert hasattr(self.solute_intra, 'mol_idxs')

        if len(self.solute_intra.mol_idxs) > 1:
            raise ValueError('More than one solvent not supported')
