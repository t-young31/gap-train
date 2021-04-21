import gaptrain as gt
import autode as ade
from autode.wrappers.keywords import GradientKeywords, OptKeywords
gt.GTConfig.n_cores = 8


# TODO: remove
gt.GTConfig.quip_version_above_66c553f = False


gt.GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'D3BJ', 'EnGrad'])

# Generate an optimised cyclobutene molecule
mol = ade.Molecule(smiles='C1C=CC1', name='cyclobutene')
mol.optimise(method=ade.methods.ORCA(),
             keywords=OptKeywords(['PBE', 'def2-SVP', 'D3BJ', 'Looseopt']))
mol.print_xyz_file()

# Create a gap-train system and train
system = gt.System(box_size=[10, 10, 10])
system.add_molecules(gt.Molecule('cyclobutene.xyz'))

data, gap = gt.active.train(system,
                            method_name='orca',
                            temp=500,
                            max_time_active_fs=200,
                            validate=False,
                            bbond_energy={(0, 3): 4},  # Atoms indexed from 0..
                            max_active_iters=15)       # ..and the energy in eV
