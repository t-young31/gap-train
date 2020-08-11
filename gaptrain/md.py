import os


def run_mmmd():
    """Run classical molecular mechanics MD on a system"""
    os.popen('gmx grompp -f min.mdp -c input.gro -p topol.top -o em.tpr')
    os.popen('gmx mdrun -deffnm em')


def run_aimd(system, *kwargs):
    """Run ab-initio molecular dynamics on a system"""
    raise NotImplementedError


def run_gapmd(system, gap, *kwargs):
    """Run molecular dynamics on a system using a GAP potential"""
    raise NotImplementedError

