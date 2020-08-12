from ase.calculators.dftb import Dftb
from gaptrain.utils import work_in_tmp_dir
from gaptrain.exceptions import MethodFailed
from ase.optimize import BFGS


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
def run_gpaw(configuration, max_force):
    """Run a periodic DFT calculation using GPAW"""
    from gpaw import GPAW, PW

    ase_atoms = configuration.ase_atoms()

    dft = GPAW(mode=PW(400),
               basis='dzp',
               charge=configuration.charge,
               xc='PBE',
               txt=None)

    ase_atoms.set_calculator(dft)

    if max_force is not None:
        minimisation = BFGS(ase_atoms)
        minimisation.run(fmax=float(max_force))

    configuration.energy.true = ase_atoms.get_potential_energy()
    configuration.forces.set_true(forces=ase_atoms.get_forces())

    return configuration


@work_in_tmp_dir()
def run_gap(configuration, max_force):
    raise NotImplementedError


@work_in_tmp_dir()
def run_dftb(configuration, max_force):
    """Run periodic DFTB+ on this configuration"""

    ase_atoms = configuration.ase_atoms()
    dftb = DFTB(atoms=ase_atoms,
                kpts=(1, 1, 1),
                Hamiltonian_Charge=configuration.charge)

    ase_atoms.set_calculator(dftb)

    try:
        configuration.energy.true = ase_atoms.get_potential_energy()

        if max_force is not None:
            minimisation = BFGS(ase_atoms)
            minimisation.run(fmax=float(max_force))

    except ValueError:
        raise MethodFailed('DFTB+ failed to generate an energy')

    configuration.forces.set_true(forces=ase_atoms.get_forces())

    # Return self to allow for multiprocessing
    return configuration
