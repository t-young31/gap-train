import numpy as np
from ase.calculators.dftb import Dftb
from gaptrain.utils import work_in_tmp_dir
from gaptrain.log import logger
from gaptrain.exceptions import MethodFailed, GAPFailed
from gaptrain.gtconfig import GTConfig
from subprocess import Popen, PIPE
import os

ha_to_ev = 27.2114
a0_to_ang = 0.52917829614246


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
def run_autode(configuration, max_force=None, method=None, n_cores=1, kwds=None):
    """
    Run an orca or xtb calculation

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None

    :param method: (autode.wrappers.base.ElectronicStructureMethod)

    :param n_cores: (int) Number of cores to use for the calculation

    :param kwds: (autode.wrappers.keywords.Keywords)
    """
    from autode.species import Species
    from autode.calculation import Calculation
    from autode.exceptions import CouldNotGetProperty
    
    if method.name == 'orca' and GTConfig.orca_keywords is None and kwds is None:
        raise ValueError("For ORCA training GTConfig.orca_keywords must be"
                         " set. or this function called with kwds e.g. "
                         "GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])")

    # optimisation is not implemented, needs a method to run
    assert max_force is None and method is not None

    species = Species(name=configuration.name,
                      atoms=configuration.atoms,
                      charge=configuration.charge,
                      mult=configuration.mult)

    # allow for an ORCA calculation to have non-default keywords
    if kwds is None and method.name == 'orca':
        kwds = GTConfig.orca_keywords
    if kwds is None:                      # Default to a gradient calculation
        kwds = method.keywords.grad

    calc = Calculation(name='tmp',
                       molecule=species,
                       method=method,
                       keywords=kwds,
                       n_cores=n_cores)
    calc.run()
    ha_to_ev = 27.2114
    try:
        configuration.forces = -ha_to_ev * calc.get_gradients()
    except CouldNotGetProperty:
        logger.error('Failed to set forces')

    configuration.energy = ha_to_ev * calc.get_energy()

    configuration.partial_charges = calc.get_atomic_charges()

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


@work_in_tmp_dir(kept_exts=['.traj'])
def run_cp2k(configuration, max_force):
    """Run periodic CP2K on this configuration. Will set configuration.energy
    and configuration.forces as their calculated values at the DFT level.

    --------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param max_force: (float) or None
    """
    assert max_force is None

    if 'CP2K_BASIS_FOLDER' not in os.environ:
        raise RuntimeError('Could not execute CP2K. Set the environment '
                           'variable for the directory containing basis sets '
                           '$CP2K_BASIS_FOLDER')

    if set([atom.label for atom in configuration.atoms]) != {'O', 'H'}:
        raise NotImplementedError('CP2K input files only built for O/H '
                                  'containing configurations')

    basis_dir = os.environ['CP2K_BASIS_FOLDER']
    if not basis_dir.endswith('/'):
        basis_dir += '/'

    configuration.save(filename='init.xyz')

    a, b, c = configuration.box.size
    with open('cp2k.inp', 'w') as inp_file:
        print("&GLOBAL",
              "  PROJECT name",
              "  RUN_TYPE ENERGY_FORCE",
              "  PRINT_LEVEL LOW",
              "&END GLOBAL",
              "",
              "&FORCE_EVAL",
              "  &DFT",
              f"    BASIS_SET_FILE_NAME {basis_dir}GTH_BASIS_SETS",
              f"    BASIS_SET_FILE_NAME {basis_dir}BASIS_ADMM",
              f"    POTENTIAL_FILE_NAME {basis_dir}POTENTIAL",
              "    &MGRID",
              "      CUTOFF 400",
              "    &END MGRID",
              "    &SCF",
              "      SCF_GUESS ATOMIC",
              "      MAX_SCF 20",
              "      EPS_SCF 5.0E-7",
              "      &OT",
              "        MINIMIZER DIIS",
              "        PRECONDITIONER FULL_ALL",
              "      &END OT",
              "      &OUTER_SCF",
              "        MAX_SCF 20",
              "        EPS_SCF 5.0E-7",
              "      &END OUTER_SCF",
              "    &END SCF",
              "    &QS",
              "      EPS_DEFAULT 1.0E-12",
              "      EPS_PGF_ORB 1.0E-14",
              "      EXTRAPOLATION_ORDER 5",
              "    &END QS",
              "    &XC # revPBE0-TC-D3",
              "      &XC_FUNCTIONAL",
              "        &PBE",
              "          PARAMETRIZATION REVPBE",
              "          SCALE_X 0.75",
              "          SCALE_C 1.0",
              "        &END",
              "      &END XC_FUNCTIONAL",
              "      &HF",
              "        FRACTION 0.25",
              "        &SCREENING",
              "          EPS_SCHWARZ 1.0E-6",
              "          SCREEN_ON_INITIAL_P FALSE",
              "        &END",
              "        &MEMORY",
              "          MAX_MEMORY 37000",
              "          EPS_STORAGE_SCALING 0.1",
              "        &END",
              "        &INTERACTION_POTENTIAL",
              "          POTENTIAL_TYPE TRUNCATED",
              "          CUTOFF_RADIUS 6.0",
              f"          T_C_G_DATA {basis_dir}t_c_g.dat",
              "        &END",
              "        &HF_INFO",
              "        &END HF_INFO",
              "      &END",
              "      &VDW_POTENTIAL",
              "         POTENTIAL_TYPE PAIR_POTENTIAL",
              "         &PAIR_POTENTIAL",
              "            TYPE DFTD3",
              "            R_CUTOFF 15",
              "            LONG_RANGE_CORRECTION TRUE",
              "            REFERENCE_FUNCTIONAL revPBE0",
              f"           PARAMETER_FILE_NAME {basis_dir}dftd3.dat",
              "         &END",
              "      &END",
              "      &XC_GRID",
              "        XC_DERIV SPLINE2",
              "      &END",
              "    &END XC",
              "    &AUXILIARY_DENSITY_MATRIX_METHOD",
              "      METHOD BASIS_PROJECTION",
              "      ADMM_PURIFICATION_METHOD MO_DIAG",
              "    &END AUXILIARY_DENSITY_MATRIX_METHOD",
              "  &END DFT",
              "  &SUBSYS",
              "    &TOPOLOGY",
              "      COORD_FILE_NAME init.xyz",
              "      COORD_FILE_FORMAT XYZ",
              "      CONN_FILE_FORMAT GENERATE",
              "    &END TOPOLOGY",
              "    &CELL",
              f"      ABC [angstrom] {a:.2f} {b:.2f} {c:.2f}",
              "    &END CELL",
              "    &KIND H",
              "      BASIS_SET TZV2P-GTH",
              "      BASIS_SET AUX_FIT cpFIT3",
              "      POTENTIAL GTH-PBE-q1",
              "    &END KIND",
              "    &KIND O",
              "      BASIS_SET TZV2P-GTH",
              "      BASIS_SET AUX_FIT cpFIT3",
              "      POTENTIAL GTH-PBE-q6",
              "    &END KIND",
              "  &END SUBSYS",
              "  &PRINT",
              "    &FORCES ON",
              "    &END FORCES",
              "  &END PRINT",
              "&END FORCE_EVAL",
              "",
              sep='\n', file=inp_file)

    # Run the calculation
    calc = Popen(GTConfig.cp2k_command + ['-o', 'cp2k.out',  'cp2k.inp'],
                 shell=False)
    calc.communicate()

    if not os.path.exists('cp2k.out'):
        raise RuntimeError('CP2K failed')

    set_energy_forces_cp2k_out(configuration, out_filename='cp2k.out')
    return configuration


def get_gp_var_quip_out(configuration, out_filename='quip.out'):
    """
    Given a QUIP output file extract the numpy array of atomic variances for
    each set of atoms in the output file

    :param configuration: (gt.Configuration)
    :param out_filename: (str)
    :return: (list(np.ndarray))
    """
    out_lines = [line for line in open(out_filename, 'r')
                 if line.startswith('AT')]

    # and grab the local GP variance per atom from the output
    first_line_idx = None
    for i, line in enumerate(out_lines):
        try:
            if int(line.split()[-1]) == len(configuration.atoms):
                first_line_idx = i
                break

        except ValueError:
            continue

    if first_line_idx is None:
        raise RuntimeError('Could not extract the first line')

    gp_vars = []
    n_atoms = len(configuration.atoms)
    for i, line in enumerate(out_lines[first_line_idx:][::n_atoms+4]):
        atom_vars = []

        # Go through each xyz section and grab the predicted atomic variance
        first_line = first_line_idx + i*(n_atoms+2) + 2
        for xyz_line in out_lines[first_line:first_line+n_atoms]:
            try:
                atom_var = float(xyz_line.split()[-4])
                atom_vars.append(atom_var)

            except (ValueError, IndexError):
                raise RuntimeError('Could not extract the atomic var')

        gp_vars.append(np.array(atom_vars))

    return gp_vars


def set_energy_forces_cp2k_out(configuration, out_filename='cp2k.out'):
    """
    Set the energy and forces of a configuration from a CP2K output file

    :param configuration: (gt.Configuration)
    :param out_filename: (str)
    """
    n_atoms = len(configuration.atoms)
    forces = []

    out_lines = open(out_filename, 'r').readlines()
    for i, line in enumerate(out_lines):
        """
        Total energy:                                      -17.23430883483457
        """
        if 'Total energy:' in line:
            # Convert from Ha to eV
            configuration.energy = ha_to_ev * float(line.split()[-1])

        # And grab the first set of atomic forces
        if 'ATOMIC FORCES' in line:
            logger.info('Found CP2K forces')
            """
            Format e.g.:
            
            ATOMIC FORCES in [a.u.]

            # Atom   Kind   Element          X              Y              Z
              1      1      O           0.02872261     0.00136975     0.02168759
              2      2      H          -0.00988376     0.02251862    -0.01740272
              3      2      H          -0.01791165    -0.02390685    -0.00393702
            """

            for f_line in out_lines[i+3:i+3+n_atoms]:
                fx, fy, fz = f_line.split()[3:]
                forces.append([float(fx), float(fy), float(fz)])
            break

    # Convert from atomic units to eV Å-1
    configuration.forces = np.array(forces) * (ha_to_ev / a0_to_ang)
    return None


def set_configuration_atoms_from_ase(config, ase_atoms):
    """
    Set the atom positions of a configuration given a set of ASE atoms

    :param config: (gt.Configuration)
    :param ase_atoms: (ase.Atoms)
    """

    for i, coord in enumerate(ase_atoms.get_positions()):
        config.atoms[i].coord = coord

    return None
