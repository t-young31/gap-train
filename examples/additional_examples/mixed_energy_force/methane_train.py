import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 10

methane = gt.System(box_size=[50, 50, 50])
methane.add_molecules(gt.Molecule('methane.xyz'))

data, gap = gt.active.train(methane,
                            method_name='xtb',
                            temp=1000)
data.save(filename='xtb_active.xyz')

data = gt.Data('xtb_active.xyz')
# evaluate gradients
data.parallel_orca(keywords=GradientKeywords(['MP2', 'TightSCF', 'def2-TZVP', 'EnGrad']))
# and then single point energies
data.parallel_orca(keywords=GradientKeywords(['CCSD(T)', 'TightSCF', 'def2-TZVP']))

gap = gt.GAP(name='mixed', system=methane)
gap.train(data)

traj = gt.md.run_gapmd(configuration=methane.random(),
                       gap=gap,
                       temp=500,     # Kelvin
                       dt=0.5,       # fs
                       interval=1,   # frames
                       fs=500,
                       n_cores=4)

traj.save('traj.xyz')
