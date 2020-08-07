import os


def run_mmmd(mmsystem, *kwargs):
    """Run classical molecular mechanics MD on a system"""
    os.system(' gmx grompp -f min.mdp -c input.gro -p topol.top -o em.tpr')
    os.system('gmx mdrun -deffnm em') #change to subprocess?
    raise NotImplementedError


def run_aimd(system, *kwargs):
    """Run ab-initio molecular dynamics on a system"""
    raise NotImplementedError


def run_gapmd(system, gap, *kwargs):
    """Run molecular dynamics on a system using a GAP potential"""
    raise NotImplementedError

