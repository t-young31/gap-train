import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 20

system = gt.System(box_size=[10, 10, 10])
system.add_molecules(gt.Molecule('AcOH.xyz'))

# data, gap = gt.active.train(system,
#                         method_name='xtb',
#                         temp=1000)
# data.save(filename='xtb_active.xyz')

data = gt.Data('xtb_active.xyz')
data.truncate(n=len(data)//2, method='cur')

# evaluate gradients
data.parallel_orca(keywords=GradientKeywords(['DLPNO-MP2', 'TightPNO', 'def2-TZVP/C', 'RIJCOSX', 'def2/J', 'TightSCF', 'def2-TZVP', 'EnGrad']))
# and then single point energies
data.parallel_orca(keywords=GradientKeywords(['DLPNO-CCSD(T)', 'TightPNO', 'def2-TZVP/C', 'RIJCOSX', 'def2/J', 'TightSCF', 'def2-TZVP']))
data.save(filename='xtb_active_ccsdt_e_mp2_f.xyz')

gap = gt.GAP(name='mixed', system=system)
gap.train(data)

traj = gt.md.run_gapmd(configuration=system.random(),
                       gap=gap,
                       temp=500,     # Kelvin
                       dt=0.5,       # fs
                       interval=1,   # frames
                       fs=500,
                       n_cores=4)

traj.save('traj.xyz')
