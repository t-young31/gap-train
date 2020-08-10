from ase.calculators.dftb import Dftb
from gaptrain.gtconfig import GTConfig
from gaptrain.utils import work_in_tmp_dir
from gaptrain.exceptions import MethodFailed
import os


class DFTB(Dftb):
    """
    DFTB+ installed from the binaries downloaded from:
    https://www.dftbplus.org/download/dftb-stable/

    sk-files from:
    http://www.dftb.org/parameters/download/3ob/3ob-3-1-cc/
    """

    def read_fermi_levels(self):
        """ASE calculator doesn't quite work..."""

        try:
            super().read_fermi_levels()
        except AssertionError:
            pass

        return None


@work_in_tmp_dir()
def run_gpaw(configuration, n_cores):
    """Run a periodic DFT calculation using GPAW"""

    """
    'dft = GPAW(mode=PW(400),',
              '      basis=\'dzp\',',
              f'     charge={self.charge},',
              '      xc=\'PBE\',',
              f'     txt=\'{output_filename}\')',
              'system.set_calculator(dft)',
              'system.get_potential_energy()',
              'system.get_forces()
    """

    raise NotImplementedError


@work_in_tmp_dir()
def run_gap(configuration, n_cores):
    raise NotImplementedError


@work_in_tmp_dir()
def run_dftb(configuration, n_cores):
    """Run periodic DFTB+ on this configuration"""

    # Environment variables required for ASE
    os.environ['DFTB_PREFIX'] = GTConfig.dftb_data
    os.environ['DFTB_COMMAND'] = GTConfig.dftb_exe
    os.environ['OMP_NUM_THREADS'] = str(n_cores)

    ase_atoms = configuration.ase_atoms()
    dftb = DFTB(atoms=ase_atoms,
                kpts=(1, 1, 1),
                Hamiltonian_Charge=configuration.charge)

    ase_atoms.set_calculator(dftb)
    try:
        configuration.energy.true = ase_atoms.get_potential_energy()
    except ValueError:
        raise MethodFailed('DFTB+ failed to generate an energy')

    configuration.forces.set_true(forces=ase_atoms.get_forces())

    # Return self to allow for multiprocessing
    return configuration
