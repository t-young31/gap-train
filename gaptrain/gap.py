from gaptrain.gtconfig import GTConfig
from gaptrain.log import logger
from gaptrain.plotting import plot_ef_correlation
from gaptrain.exceptions import GAPFailed
from gaptrain.calculators import run_gap
from ase.calculators.calculator import Calculator
from ase.atoms import Atoms as ASEAtoms
from gaptrain.ase_calculators import IntraCalculator, IICalculator
from autode.atoms import elements
from subprocess import Popen, PIPE
from itertools import combinations_with_replacement
from multiprocessing import Pool
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

    @property
    def xml_filename(self):
        return f'{self.name}.xml'

    def _check_xml_exists(self):
        """Raise an exception if the parameter file (.xml) doesn't exist"""
        if not os.path.exists(self.xml_filename):
            raise IOError(f'GAP parameter file ({self.xml_filename}) did not '
                          f'exist')

    def ase_calculator(self):
        """
        ASE Calculator instance to evaluate the energy using a GAP with
        parameter filename: self.xml_filename

        :return: (ase.Calculator)
        """
        try:
            import quippy
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Quippy was not installed. Try\n'
                                      'pip install quippy-ase')

        self._check_xml_exists()
        return quippy.potential.Potential("IP GAP",
                                          param_filename=self.xml_filename)

    def train_command(self):
        """Generate the teach_sparse function call for this system of atoms"""

        general = self.params.general
        params = ('default_sigma={'
                  f'{general["sigma_E"]:.6f} {general["sigma_F"]:.6f} 0.0 0.0'
                  '} ')

        params += 'e0_method=average gap={'

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
                       f'add_species=F '
                       f'n_Z=1 '
                       f'n_species={len(soap["other"])} '
                       'species_Z={{'
                       # Remove the brackets from the ends of the list
                       f'{str(other_atomic_ns)[1:-1]}'
                       '}} '
                       f'Z={atomic_number(symbol)} '
                       f'n_max={int(2*soap["order"])} '
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

        if all(len(params) == 0 for params in
               (self.params.soap, self.params.pairwise, self.params.angle)):
            raise AssertionError('Must have some GAP parameters!')

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
        logger.info(f'GAP training ran in {delta_time/60:.1f} m')

        if any((delta_time < 0.01,
                b'SYSTEM ABORT' in err,
                not os.path.exists(f'{self.name}.xml'))):

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
        :param system: (gt.systems.System
                        | gt.molecules.UniqueMolecule
                        | None)
        """

        # Default to removing the extension if it exists
        self.name = name if not name.endswith('.xml') else name[:-4]

        if system is not None and default_params:
            self.params = Parameters(atom_symbols=system.atom_symbols)

        else:
            self.params = Parameters(atom_symbols=[])
            logger.warning('Initialised a GAP with no parameters. '
                           'gap.train not available')

        self.training_data = None


class InterGAP(GAP):
    """'Intermolecular' GAP i.e. reminder after intra subtraction"""


class IntraGAP(GAP):

    def ase_calculator(self):
        """
        Generate the quippy/ASE string to run the potential

        :param mol_idxs: (list(list(int)) | None) Indexes of the atoms within
                         a configuration which form molecules that this
                         intra GAP can evaluate. e.g. for a configuration of
                         two water molecules:
                            mol_idxs = [[0, 1, 2], [3, 4, 5]]

                        If None then use the initial definition given by
                        the unique molecule

        :return: (ase.Calculator)
        """
        self._check_xml_exists()

        return IntraCalculator("IP GAP", self.xml_filename, mol_idxs=self.mol_idxs)

    def __init__(self, name, unique_molecule):
        """
        An intramolecular GAP defined for a unique molecule specific to a
        system

        :param name: (str)
        :param unique_molecule: (gt.molecules.UniqueMolecule)
        """
        super().__init__(name, unique_molecule)
        self.mol_idxs = unique_molecule.atom_idxs

        logger.info(f'Initialised an intra-GAP with molecular indexes:'
                    f' {self.mol_idxs}')


class IIGAP:
    """Inter+intra GAP where the inter is evaluated in a different box"""

    def train(self, data):
        """Train the inter-component of the GAP"""
        logger.info('Training the intermolecular component of the potential. '
                    'Expecting data that is free from intra energy and force')

        if not all(os.path.exists(gap.xml_filename) for gap in self.intra_gaps):
            raise RuntimeError('Intramolecular GAPs must be already trained')

        self.training_data = data
        return self.inter_gap.train(data)

    def _check_xmls_exists(self):
        """Raise an exception if the parameter file (.xml) doesn't exist"""
        if not os.path.exists(self.inter_gap.xml_filename):
            raise IOError(f'Intermolecular GAP parameter file must exist')

        for gap in self.intra_gaps:
            if not os.path.exists(gap.xml_filename):
                raise IOError(f'GAP parameter file ({gap.xml_filename}) did '
                              f'not exist')
        return None

    def ase_calculator(self):
        """Generate the quippy/ASE string to run the potential"""
        self._check_xmls_exists()

        return IICalculator("IP GAP",
                            xml_filename=self.inter_gap.xml_filename,
                            intra_calculators=[gap.ase_calculator()
                                               for gap in self.intra_gaps])

    def __init__(self, *args):
        """
        Collective GAP comprised of inter and intra components

        :param args: (gt.gap.IntraGAP | gt.gap.InterGAP)
        """
        self.training_data = None   # intermolecular training data
        self.inter_gap = None
        self.intra_gaps = []

        for arg in args:
            if isinstance(arg, IntraGAP):
                self.intra_gaps.append(arg)

            elif isinstance(arg, InterGAP):
                self.inter_gap = arg

            else:
                raise ValueError('IIGAP must be initialised with only '
                                 'InterGAP or IntraGAP potentials')

        if self.inter_gap is None or len(self.intra_gaps) == 0:
            raise AssertionError('Must have both an inter+intra GAP')

        self.name = f'II_{self.inter_gap.name}'


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
                               if s+symbol not in added_pairs
                               and symbol+s not in added_pairs]

            # If there are no other atoms of this type then remove the self
            # pair
            if atom_symbols.count(symbol) == 1:
                params["other"].remove(symbol)

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
