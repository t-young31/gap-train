from gaptrain.trajectories import Trajectory
from gaptrain.calculators import DFTB
from gaptrain.log import logger
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
    """Run ab-initio molecular dynamics on a system"""
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


def run_gapmd(system, gap, *kwargs):
    """Run molecular dynamics on a system using a GAP potential"""
    raise NotImplementedError
