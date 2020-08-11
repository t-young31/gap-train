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
def run_gpaw(configuration, n_cores=1):
    """Run a periodic DFT calculation using GPAW"""
    from gpaw import GPAW, PW

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['MLK_NUM_THREADS'] = str(n_cores)
    os.environ['GPAW_SETUP_PATH'] = GTConfig.gpaw_setup_path

    atoms = configuration.ase_atoms()

    dft = GPAW(mode=PW(400),
               basis='dzp',
               charge=configuration.charge,
               xc='PBE',
               txt=None)

    atoms.set_calculator(dft)
    configuration.energy.true = atoms.get_potential_energy()
    configuration.forces.set_true(forces=atoms.get_forces())

    return configuration


@work_in_tmp_dir()
def run_gap(configuration, n_cores):
    raise NotImplementedError


@work_in_tmp_dir()
def run_dftb(configuration, n_cores=1):
    """Run periodic DFTB+ on this configuration"""

    # Environment variables required for ASE
    env = os.environ.copy()
    env['DFTB_PREFIX'] = GTConfig.dftb_data
    env['DFTB_COMMAND'] = GTConfig.dftb_exe
    env['OMP_NUM_THREADS'] = str(n_cores)

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
