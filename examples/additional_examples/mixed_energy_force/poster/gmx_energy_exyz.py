"""
Evaluate GROMACS energies on a set of configurations given as an extended
.xyz file
"""
import os
import gaptrain as gt
import numpy as np
from subprocess import Popen, DEVNULL

ang_to_nm = 0.1
kj_mol_to_ev = 0.01036427230133138


class GROMACSAtom:

    @property
    def x(self):
        return self.coord[0]

    @property
    def y(self):
        return self.coord[1]

    @property
    def z(self):
        return self.coord[2]

    def __init__(self, atom_type, x, y, z):

        self.atom_type = atom_type
        self.coord = [float(x), float(y), float(z)]


class GROMACSBox:

    @property
    def a(self):
        return self.size[0]

    @property
    def b(self):
        return self.size[1]

    @property
    def c(self):
        return self.size[2]

    def __init__(self, size):

        self.size = size   # nm


class GROMACSStructure:

    @property
    def n_atoms(self):
        return len(self.atoms)

    def _parse_file(self, gro_filename):

        for i, line in  enumerate(open(gro_filename, 'r')):
            if i < 2:
                continue

            if len(line.split()) == 6:
                self.resid_name, atom_type, _, x, y, z = line.split()
                self.atoms.append(GROMACSAtom(atom_type, x=x, y=y, z=z))

            if len(line.split()) == 3:
                self.box = GROMACSBox(size=[float(x) for x in line.split()])

    def __init__(self, gro_filename):

        self.resid_name = 'AA'
        self.atoms = []
        self.box = None

        self._parse_file(gro_filename)


def _check():
    if not os.path.exists(f'{mol_name}.top'):
        exit(f'Topology did not exist, Must have {mol_name}.top')

    if not os.path.exists(f'{mol_name}.gro'):
        exit(f'GROMACS structure did not exist, Must have {mol_name}.gro'
             f'to modify')

    if not os.path.exists('energy.mdp'):
        exit('Must have an energy.mdp file for the paramters')

    return None


def energy_from_log():
    """Get the total energy from the GROMACS log file"""

    if not os.path.exists('energy.log'):
        exit('energy.log did not exist - cannot obtain the energy')

    energy_filelines = open('energy.log', 'r').readlines()

    try:
        total_energy_idx = next(i for i, line in enumerate(energy_filelines)
                                if 'Total Energy' in line)

        return float(energy_filelines[total_energy_idx+1].split()[1])

    except StopIteration:
        exit('No total energy in energy.log')


def print_gro_file(filename, configuration, structure):

    if len(configuration.atoms) != structure.n_atoms:
        exit('Mismatched atom number between GROMACS structure and '
             'exyz')

    with open(filename, 'w') as gro_file:
        print("GROningen MAchine for Chemical Simulation",
              len(configuration.atoms),
              sep='\n', file=gro_file)

        for i, gmx_atom in enumerate(structure.atoms):
            atom = configuration.atoms[i]

            print(f'{structure.resid_name:>7s}'
                  f'{gmx_atom.atom_type:>8s}'
                  f'{str(i+1):>5s}'   # atoms indexed from 1
                  f'{atom.coord[0] * ang_to_nm:8.3f}'
                  f'{atom.coord[1] * ang_to_nm:8.3f}'
                  f'{atom.coord[2] * ang_to_nm:8.3f}',
                  file=gro_file)

        print(f'{structure.box.a:10.5f}'
              f'{structure.box.b:10.5f}'
              f'{structure.box.c:10.5f}', file=gro_file)

    return None


def gmx_energy(configuration):
    """Get the GROMACS energy for a configuration"""
    _check()
    print_gro_file('tmp.gro', configuration,
                   structure=GROMACSStructure(f'{mol_name}.gro'))

    gmx = Popen(['gmx', 'grompp',
                 '-f', 'energy.mdp',
                 '-c', 'tmp.gro',
                 '-p', f'{mol_name}.top',
                 '-o', 'energy.tpr'],
                stdout=DEVNULL, stderr=DEVNULL)
    gmx.wait()
    gmx = Popen(['gmx', 'mdrun', '-nt', '1', '-deffnm', 'energy'],
                stdout=DEVNULL, stderr=DEVNULL)
    gmx.wait()

    energy = energy_from_log()
    print(energy)
    return kj_mol_to_ev * energy


if __name__ == '__main__':

    traj = gt.Trajectory('gap_traj.xyz',
                         box=gt.Box([50, 50, 50]), charge=0, mult=1)
    mol_name = 'acoh'
    mapping = {}

    energies = [gmx_energy(frame) for frame in traj]
    np.savetxt(f'{mol_name}.txt', np.array(energies))

