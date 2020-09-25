from gaptrain.trajectories import Trajectory
from gaptrain.calculators import DFTB
from gaptrain.utils import work_in_tmp_dir
from gaptrain.log import logger
from gaptrain.gtconfig import GTConfig
from subprocess import Popen, PIPE
import subprocess
import numpy as np
import os


def simulation_steps(dt, kwargs):
    """Calculate the number of simulation steps

    :param dt: (float) Timestep in fs
    :param kwargs: (dict)
    :return: (float)
    """
    if dt < 0.09 or dt > 5:
        logger.warning('Unexpectedly small or large timestep - is it in fs?')

    if 'ps' in kwargs:
        time_fs = 1E3 * kwargs['ps']

    elif 'fs' in kwargs:
        time_fs = kwargs['fs']

    elif 'ns' in kwargs:
        time_fs = 1E6 * kwargs['ns']

    else:
        raise ValueError('Simulation time not found')

    logger.info(f'Running {time_fs / dt:.0f} steps with a timestep of {dt} fs')
    # Run at least one step
    return max(int(time_fs / dt), 1)


def run_mmmd(system, config, temp, dt, interval, **kwargs):
    """
    Generate topology and input gro files.
    Run classical molecular mechanics MD on a system

    ---------------------------------------------------------------------------
    :param system: (gaptrain.MMSystem)

    :param config: (gaptrain.MMSystem.random)

    :param temp: (float) Temperature in K to use

    :param dt: (float) Timestep in fs

    :param interval: (int) Interval between printing the geometry

    :param kwargs: {fs, ps, ns} Simulation time in some units
    """

    os.environ['OMP_NUM_THREADS'] = str(GTConfig.n_cores)

    # Create topol.top and input.gro files
    system.generate_topology()
    config.print_gro_file(system=system)

    a, b, c = system.box.size

    if a <= 20 or b <= 20 or c <= 20:
        # GROMACS requires cutoff to be less than hal the smallest box length
        cutoff = (np.min(system.box.size) * 0.05 - 0.001)
    else:
        cutoff = 1.0

    # Create min.mdp parameters file
    with open('min.mdp', 'w') as min_file:

        print(f'{"integrator":<20}{"= steep"}',
              f'{"emtol":<20}{"= 1000.0"}',
              f'{"emstep":<20}{"= 0.01"}',
              f'{"nsteps":<20}{"= 50000"}',
              f'{"nstlist":<20}{"= 1"}',
              f'{"cutoff-scheme":<20}{"= Verlet"}',
              f'{"ns_type":<20}{"= grid"}',
              f'{"coulombtype":<20}{"= PME"}',
              f'{"rcoulomb":<20s}{"= "}{cutoff:.3f}',
              f'{"rvdw":<20s}{"= "}{cutoff:.3f}',
              f'{"pbc":<20s}{"= xyz"}', file=min_file, sep='\n')

    # Create nvt.mdp parameters file
    with open('nvt.mdp', 'w') as nvt_file:

        print(f'{"title":<25}{"= GAP-Train NVT parameter file"}',
              f'{"define":<25}{"= -DPOSRES"}',
              f'{"integrator":<25}{"= md"}',
              f'{"nsteps":<25}{"= "}{simulation_steps(dt, kwargs)}',
              f'{"dt":<25}{"= "}{dt / 1E3}',   # converts to picoseconds (ps)
              f'{"init_step":<25}{"= 0"}',
              f'{"comm-mode":<25}{"= Linear"}',
              f'{"nstxout":<25}{"= "}{interval}',
              f'{"nstvout":<25}{"= "}{interval}',
              f'{"nstenergy":<25}{"= "}{interval}',
              f'{"nstlog":<25}{"= "}{interval}',
              f'{"nstxout-compressed":<25}{"= "}{interval}',
              f'{"continuation":<25}{"= no"}',
              f'{"constraint_algorithm":<25}{"= lincs"}',
              f'{"constraints":<25}{"= h-bonds"}',
              f'{"lincs_iter":<25}{"= 1"}',
              f'{"lincs_order":<25}{"= 4"}',
              f'{"cutoff-scheme":<25}{"= Verlet"}',
              f'{"ns_type":<25}{"= grid"}',
              f'{"nstlist":<25}{"= 10"}',
              f'{"rcoulomb":<25}{"= "}{cutoff:.3f}',
              f'{"vdw-type":<25}{"= Cut-off"}',
              f'{"rvdw":<25}{"= "}{cutoff:.3f}',
              f'{"coulombtype":<25}{"= PME"}',
              f'{"pme_order":<25}{"= 4"}',
              f'{"fourierspacing":<25}{"= 0.12"}',
              f'{"tcoupl":<25}{"= V-rescale"}',
              f'{"tc-grps":<25}{"= system"}',
              f'{"tau_t":<25}{"= 0.1"}',
              f'{"ref_t":<25}{"= "}{temp}',
              f'{"pcoupl":<25}{"= no"}',
              f'{"pbc":<25}{"= xyz"}',
              f'{"DispCorr":<25}{"= EnerPres"}',
              f'{"gen_vel":<25}{"= yes"}',
              f'{"gen-temp":<25}{"= "}{temp}',
              f'{"gen-seed":<25}{"= -1"}', file=nvt_file, sep='\n')

    # Run gmx minimisation and nvt simulations
    grompp_em = Popen(['gmx', 'grompp', '-f', 'min.mdp', '-c', 'input.gro',
                       '-p', 'topol.top', '-o', 'em.tpr',
                       '-maxwarn', '5'], shell=False)
    grompp_em.wait()

    minimisation = Popen(['gmx', 'mdrun', '-deffnm', 'em'], shell=False)
    minimisation.wait()

    grompp_nvt = Popen(['gmx', 'grompp', '-f', 'nvt.mdp', '-c', 'em.gro',
                        '-p', 'topol.top', '-o', 'nvt.tpr',
                        '-maxwarn', '5'], shell=False)
    grompp_nvt.wait()

    nvt = Popen(['gmx', 'mdrun', '-deffnm', 'nvt'], shell=False)
    nvt.wait()

    echo = Popen(('echo', "System"), stdout=PIPE)
    subprocess.check_output(['gmx', 'trjconv', '-f', 'nvt.xtc', '-s', 'nvt.tpr'
                                , '-o', 'nvt_traj.gro'], stdin=echo.stdout)
    echo.wait()

    return Trajectory(filename='nvt_traj.gro', init_configuration=config)


def run_dftbmd(configuration, temp, dt, interval, **kwargs):
    """
    Run ab-initio molecular dynamics on a system. To run a 10 ps simulation
    with a timestep of 0.5 fs saving every 10th step at 300K

    run_dftbmd(config, temp=300, dt=0.5, interval=10, ps=10)

    ---------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param temp: (float) Temperature in K to use

    :param dt: (float) Timestep in fs

    :param interval: (int) Interval between printing the geometry

    :param kwargs: {fs, ps, ns} Simulation time in some units
    """
    logger.info('Running DFTB+ MD')
    ase_atoms = configuration.ase_atoms()

    dftb = DFTB(atoms=ase_atoms,
                kpts=(1, 1, 1),
                Hamiltonian_Charge=configuration.charge)
    ase_atoms.set_calculator(dftb)

    # Do a single point energy evaluation to make sure the calculation works..
    # also to generate the input file which can be modified
    try:
        ase_atoms.get_potential_energy()

    except ValueError:
        raise Exception('DFTB+ failed to calculate the first point')

    # Append to the generated input file
    with open('dftb_in.hsd', 'a') as input_file:

        print('Driver = VelocityVerlet{',
              f'  TimeStep [fs] = {dt}',
              '  Thermostat = NoseHoover {',
              f'    Temperature [Kelvin] = {temp}',
              '    CouplingStrength [cm^-1] = 3200',
              '  }',
              f'  Steps = {simulation_steps(dt, kwargs)}',
              '  MovedAtoms = 1:-1',
              f'  MDRestartFrequency = {interval}',
              '}', sep='\n', file=input_file)

    with open('dftb_md.out', 'w') as output_file:
        process = Popen([os.environ['DFTB_COMMAND']],
                        shell=False, stderr=PIPE, stdout=output_file)
        _, err = process.communicate()

    if len(err) > 0:
        logger.error(f'DFTB MD: {err.decode()}')

    return Trajectory('geo_end.xyz', init_configuration=configuration)


@work_in_tmp_dir(copied_exts=['.xml'])
def run_gapmd(configuration, gap, temp, dt, interval, **kwargs):
    """
    Run molecular dynamics on a system using a GAP to predict energies and
    forces

    ---------------------------------------------------------------------------
    :param configuration: (gaptrain.configurations.Configuration)

    :param temp: (float) Temperature in K to use

    :param dt: (float) Timestep in fs

    :param interval: (int) Interval between printing the geometry

    :param kwargs: {fs, ps, ns} Simulation time in some units
    """
    logger.info('Running GAP MD')
    configuration.save(filename='config.xyz')

    a, b, c = configuration.box.size
    n_steps = simulation_steps(dt, kwargs)

    n_cores = min(GTConfig.n_cores, 8)
    os.environ['OMP_NUM_THREADS'] = str(n_cores)
    logger.info(f'Using {n_cores} cores for GAP MD')

    if not os.path.exists(f'{gap.name}.xml'):
        raise IOError('GAP parameter file (.xml) did not exist')

    # Print a Python script to execute quippy and use ASE to drive the dynamics
    with open(f'gap.py', 'w') as quippy_script:
        print('import quippy',
              'import numpy as np',
              'from ase.io import read, write',
              'from ase.io.trajectory import Trajectory',
              'from ase.md.velocitydistribution import MaxwellBoltzmannDistribution',
              'from ase import units',
              'from ase.md.langevin import Langevin',
              'system = read("config.xyz")',
              f'system.cell = [{a}, {b}, {c}]',
              'system.pbc = True',
              'system.center()',
              'pot = quippy.Potential("IP GAP", \n'
              f'                      param_filename="{gap.name}.xml")',
              'system.set_calculator(pot)',
              f'MaxwellBoltzmannDistribution(system, {temp} * units.kB)',
              'traj = Trajectory("tmp.traj", \'w\', system)\n'
              f'dyn = Langevin(system, {dt:.1f} * units.fs, {temp} * units.kB, 0.02)',
              f'dyn.attach(traj.write, interval={interval})',
              f'dyn.run(steps={n_steps})',
              sep='\n', file=quippy_script)

    # Run the process
    quip_md = Popen(GTConfig.quippy_gap_command + ['gap.py'],
                    shell=False, stdout=PIPE, stderr=PIPE)
    _, err = quip_md.communicate()

    if len(err) > 0 and 'WARNING' not in err.decode():
        logger.error(f'GAP MD: {err.decode()}')

    traj = Trajectory('tmp.traj', init_configuration=configuration)

    return traj
