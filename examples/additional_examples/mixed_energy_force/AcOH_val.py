import gaptrain as gt
import numpy as np
import matplotlib.pyplot as plt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 20

# create sets of data from the trajectory containing predicted & true energies
pred = gt.Data('traj.xyz')
pred.remove_first(800)

true = pred.copy()
true.parallel_orca(keywords=GradientKeywords(['DLPNO-CCSD(T)', 'TightPNO', 'def2-TZVP/C', 'RIJCOSX', 'def2/J', 'TightSCF', 'def2-TZVP']))
true.save(filename='traj_ccsdt.xyz')

plt.plot(np.linspace(400, 500, len(pred)),                  # 0 -> 50 fs
         pred.energies() - np.min(true.energies()),      # rel energies
         label='GAP', lw=2)

plt.plot(np.linspace(400, 500, len(true)),
         true.energies() - np.min(true.energies()),
         label='true', c='green', lw=2)

# plot the region of 'chemical accuracy' 1 kcal mol-1 = 0.043 eV
plt.fill_between(np.linspace(400, 500, len(true)),
                 y1=true.energies() - np.min(true.energies()) - 0.043,
                 y2=true.energies() - np.min(true.energies()) + 0.043,
                 alpha=0.2, color='green')

plt.xlabel('time / fs')
plt.ylabel('E / eV')
plt.xlim(400, 500)
plt.legend()
plt.savefig('energies_vs_time_AcOH.pdf')
