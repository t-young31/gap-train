from gaptrain.trajectories import Trajectory
from gaptrain.calculators import DFTB
from gaptrain.utils import work_in_tmp_dir
from gaptrain.log import logger
from gaptrain.gtconfig import GTConfig
from subprocess import Popen, PIPE
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


def run_mmmd(mmsystem, *kwargs):
    """Run classical molecular mechanics MD on a system"""
    raise NotImplementedError


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

    assert os.path.exists(f'{gap.name}.xml')

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
