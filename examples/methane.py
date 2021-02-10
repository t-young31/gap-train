"""Train a GAP for a gas phase methane molecule with a PBE/def2-SVP ground
truth and compare the accuracy along a trajectory"""
import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 8

# Set the keywords to use for an ORCA energy and gradient calculation
gt.GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])

# Initialise a cubic box 10x10x10 Å containing a single methane molecule
methane = gt.System(box_size=[10, 10, 10])
methane.add_molecules(gt.Molecule('methane.xyz'))


# Train a GAP at 1000 K
data, gap = gt.active.train(methane,
                            method_name='orca',
                            temp=1000)

# Run 1 ps molecular dynamics using the GAP at 300 K using a 0.5 fs time-step.
# The initial configuration is methane located at a random position in the box
traj = gt.md.run_gapmd(configuration=methane.random(),
                       gap=gap,
                       temp=500,    # Kelvin
                       dt=0.5,      # fs
                       interval=1,  # frames
                       fs=50,
                       n_cores=4)

# save the trajectory with no energies
traj.save(filename='traj.xyz')

# create sets of data from the trajectory containing predicted & true energies
pred = gt.Data('traj.xyz')
pred.parallel_gap(gap=gap)

true = gt.Data('traj.xyz')
true.parallel_orca()

# and plot the energies over time ---------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.plot(np.linspace(0, 50, len(pred)),                  # 0 -> 50 fs
         pred.energies() - np.min(true.energies()),      # rel energies
         label='GAP', lw=2)

plt.plot(np.linspace(0, 50, len(true)),
         true.energies() - np.min(true.energies()),
         label='true', c='orange', lw=2)

# plot the region of 'chemical accuracy' 1 kcal mol-1 = 0.043 eV
plt.fill_between(np.linspace(0, 50, len(true)),
                 y1=true.energies() - np.min(true.energies()) - 0.043,
                 y2=true.energies() - np.min(true.energies()) + 0.043,
                 alpha=0.2, color='orange')

plt.xlabel('time / fs')
plt.ylabel('E / eV')
plt.xlim(0, 50)
plt.legend()
plt.savefig('energies_vs_time_methane.png', dpi=300)
