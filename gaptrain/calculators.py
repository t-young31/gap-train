import numpy as np
from ase.calculators.dftb import Dftb
from gaptrain.utils import work_in_tmp_dir
from gaptrain.log import logger
from gaptrain.exceptions import MethodFailed, GAPFailed
from gaptrain.gtconfig import GTConfig
from subprocess import Popen, PIPE
import os


def set_threads(n_cores):
    """Set the number of threads to use"""

    n_cores = GTConfig.n_cores if n_cores is None else n_cores
    logger.info(f'Using {n_cores} cores')

    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    os.environ['MLK_NUM_THREADS'] = str(n_cores)

    return None


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
def run_orca(configuration, max_force=None, n_cores=1):
    """
    Run an orca calculation

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None
    """
    from autode.species import Species
    from autode.calculation import Calculation
    from autode.methods import ORCA
    from autode.exceptions import CouldNotGetProperty
    from autode.wrappers.keywords import GradientKeywords

    assert max_force is None

    if not isinstance(GTConfig.orca_keywords, GradientKeywords):
        raise AssertionError('ORCA requires a set of autodE GradientKeywords')

    species = Species(name=configuration.name,
                      atoms=configuration.atoms,
                      charge=configuration.charge,
                      mult=configuration.mult)
    calc = Calculation(name='tmp',
                       molecule=species,
                       method=ORCA(),
                       keywords=GTConfig.orca_keywords,
                       n_cores=n_cores)
    calc.run()
    ha_to_ev = 27.2114
    try:
        configuration.forces = -ha_to_ev * calc.get_gradients()
    except CouldNotGetProperty:
        logger.error('Failed to set forces')

    configuration.energy = ha_to_ev * calc.get_energy()

    return configuration


@work_in_tmp_dir(kept_exts=['.traj'])
def run_gpaw(configuration, max_force):
    """Run a periodic DFT calculation using GPAW. Will set configuration.energy
    and configuration.forces as their DFT calculated values at the 400eV/PBE
    level of theory

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None
    """
    from gpaw import GPAW, PW
    from ase.optimize import BFGS

    if ('GPAW_SETUP_PATH' not in os.environ.keys()
            or os.environ['GPAW_SETUP_PATH'] == ''):
        raise AssertionError('$GPAW_SETUP_PATH needs to be set')

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
        set_configuration_atoms_from_ase(configuration, ase_atoms)

    configuration.energy = ase_atoms.get_potential_energy()
    configuration.forces = ase_atoms.get_forces()

    return configuration


def ase_gap_potential_str(gap):
    """
    Return a string appropriate for a GAP or an additive GAP used in an ASE
    script

    :param gap: (gaptrain.gap.GAP | gaptrain.gap.AdditiveGAP)
    :return: (str)
    """

    # Add the potential section from either a normal or additive GAP
    from gaptrain.gap import GAP, AdditiveGAP
    pt_section = ''

    if isinstance(gap, GAP):
        if not os.path.exists(f'{gap.name}.xml'):
            raise IOError(f'GAP parameter file ({gap.name}.xml) did not exist')

        pt_section += ('pot = quippy.Potential("IP GAP", \n'
                       f'              param_filename="{gap.name}.xml")')

    if isinstance(gap, AdditiveGAP):
        for i in range(2):
            pt_section += (f'pot{i+1} = quippy.Potential("IP GAP", \n'
                           f'          param_filename="{gap[i].name}.xml")\n')

            if not os.path.exists(f'{gap[i].name}.xml'):
                raise IOError(f'GAP parameter file ({gap[i].name}.xml) in '
                              f'additiive GAP did not exist')

        pt_section += f'pot = quippy.Potential("Sum", pot1=pot1, pot2=pot2)'

    return pt_section


@work_in_tmp_dir(kept_exts=['.traj'], copied_exts=['.xml'])
def run_gap(configuration, max_force, gap, traj_name=None):
    """
    Run a GAP calculation using quippy as the driver which is a wrapper around
    the F90 QUIP code used to evaluate forces and energies using a GAP

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None

    :param gap: (gaptrain.gap.GAP)
    :return:
    """
    configuration.save(filename='config.xyz')
    a, b, c = configuration.box.size

    # Energy minimisation section to the file
    min_section = ''

    if max_force is not None:
        if traj_name is not None:
            min_section = (f'traj = Trajectory(\'{traj_name}\', \'w\', '
                           f'                  system)\n'
                           'dyn = BFGS(system)\n'
                           'dyn.attach(traj.write, interval=1)\n'
                           f'dyn.run(fmax={float(max_force)})')
        else:
            min_section = ('dyn = BFGS(system)\n'
                           f'dyn.run(fmax={float(max_force)})')

    # Print a Python script to execute quippy - likely not installed in the
    # current interpreter..
    with open(f'gap.py', 'w') as quippy_script:
        print('import quippy',
              'import numpy as np',
              'from ase.io import read, write',
              'from ase.optimize import BFGS',
              'from ase.io.trajectory import Trajectory',
              'system = read("config.xyz")',
              f'system.cell = [{a}, {b}, {c}]',
              'system.pbc = True',
              'system.center()',
              f'{gap.ase_gap_potential_str()}',
              'system.set_calculator(pot)',
              f'{min_section}',
              'np.savetxt("energy.txt",\n'
              '           np.array([system.get_potential_energy()]))',
              'np.savetxt("forces.txt", system.get_forces())',
              f'write("config.xyz", system)',
              sep='\n', file=quippy_script)

    # Run the process
    subprocess = Popen(GTConfig.quippy_gap_command + ['gap.py'],
                       shell=False, stdout=PIPE, stderr=PIPE)
    subprocess.wait()

    # Grab the energy from the output after unsetting it
    try:
        configuration.load(filename='config.xyz')
        configuration.energy = np.loadtxt('energy.txt')

    except IOError:
        raise GAPFailed('Failed to calculate energy with the GAP')

    # Grab the final forces from the numpy array
    configuration.forces = np.loadtxt('forces.txt')

    return configuration


@work_in_tmp_dir(kept_exts=['.traj'])
def run_dftb(configuration, max_force, traj_name=None):
    """Run periodic DFTB+ on this configuration. Will set configuration.energy
    and configuration.forces as their calculated values at the TB-DFT level

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None

    :param traj_name: (str) or None
    """
    from ase.optimize import BFGS

    ase_atoms = configuration.ase_atoms()
    dftb = DFTB(atoms=ase_atoms,
                kpts=(1, 1, 1),
                Hamiltonian_Charge=configuration.charge)

    ase_atoms.set_calculator(dftb)

    try:
        configuration.energy = ase_atoms.get_potential_energy()

        if max_force is not None:
            minimisation = BFGS(ase_atoms, trajectory=traj_name)
            minimisation.run(fmax=float(max_force))
            configuration.n_opt_steps = minimisation.get_number_of_steps()
            set_configuration_atoms_from_ase(configuration, ase_atoms)

    except ValueError:
        raise MethodFailed('DFTB+ failed to generate an energy')

    configuration.forces = ase_atoms.get_forces()

    configuration.partial_charges = ase_atoms.get_charges()

    # Return self to allow for multiprocessing
    return configuration


def set_configuration_atoms_from_ase(config, ase_atoms):
    """
    Set the atom positions of a configuration given a set of ASE atoms

    :param config: (gaptrain.configurations.Configuration)
    :param ase_atoms: (ase.Atoms)
    """

    for i, coord in enumerate(ase_atoms.get_positions()):
        config.atoms[i].coord = coord

    return None
