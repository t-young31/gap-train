import numpy as np
from ase.calculators.dftb import Dftb
from gaptrain.utils import work_in_tmp_dir
from gaptrain.exceptions import MethodFailed
from gaptrain.gtconfig import GTConfig
from ase.optimize import BFGS
from subprocess import Popen, PIPE


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

    configuration.energy = ase_atoms.get_potential_energy()
    configuration.forces = ase_atoms.get_forces()

    return configuration


@work_in_tmp_dir()
def run_gap(configuration, max_force, gap):
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

    # Print a Python script to execute quippy - likely not installed in the
    # current interpreter..
    with open(f'gap.py', 'w') as quippy_script:
        print('import quippy',
              'import numpy as np',
              'from ase.io import read, write',
              'from ase.optimize import BFGS',
              'from ase.io.trajectory import Trajectory',
              f'system = read("config.xyz")',
              f'system.cell = [{a}, {b}, {c}]',
              'system.pbc = True',
              'system.center()',
              f'pot = quippy.Potential("IP GAP", \n'
              f'                      param_filename="{gap.name}.xml")',
              'system.set_calculator(pot)',
              'print("energy=" system.get_potential_energy())',
              'np.savetxt("forces.txt", system.get_forces())',
              sep='\n', file=quippy_script)

    if max_force is not None:
        """
        Add something like:
        
          f'traj = Trajectory(\'out.traj\', \'w\', system)',
          'dyn = BFGS(system)',
          'dyn.attach(traj.write, interval=2)',
          f'dyn.run(steps={n_steps})',
          f'write(\'{opt_filename}\', system)',
        """
        raise NotImplementedError

    # Run the process
    subprocess = Popen(GTConfig.quippy_gap_command + ['gap.py'],
                       shell=False, stdout=PIPE, stderr=PIPE)
    out, err = subprocess.communicate()

    # Grab the energy from the output
    for line in out:
        if b'energy' in line:
            configuration.energy = float(line.decode().split()[-1])

    # Grab the final forces from the numpy array
    configuration.forces = np.loadtxt('forces.txt')

    return None


@work_in_tmp_dir()
def run_dftb(configuration, max_force):
    """Run periodic DFTB+ on this configuration"""

    ase_atoms = configuration.ase_atoms()
    dftb = DFTB(atoms=ase_atoms,
                kpts=(1, 1, 1),
                Hamiltonian_Charge=configuration.charge)

    ase_atoms.set_calculator(dftb)

    try:
        configuration.energy = ase_atoms.get_potential_energy()

        if max_force is not None:
            minimisation = BFGS(ase_atoms)
            minimisation.run(fmax=float(max_force))

    except ValueError:
        raise MethodFailed('DFTB+ failed to generate an energy')

    configuration.forces = ase_atoms.get_forces()

    # Return self to allow for multiprocessing
    return configuration
