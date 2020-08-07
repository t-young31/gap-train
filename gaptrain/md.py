

def run_mmmd(mmsystem, *kwargs):
    """Run classical molecular mechanics MD on a system"""
    raise NotImplementedError


def run_aimd(system, *kwargs):
    """Run ab-initio molecular dynamics on a system"""
    raise NotImplementedError


def run_gapmd(system, gap, *kwargs):
    """Run molecular dynamics on a system using a GAP potential"""
    raise NotImplementedError
