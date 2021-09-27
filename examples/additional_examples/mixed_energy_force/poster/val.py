import gaptrain as gt
import numpy as np
import matplotlib.pyplot as plt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 20

ev_to_kcal = 23.060541945329334


# create sets of data from the trajectory containing predicted & true energies
pred = gt.Data('traj.xyz')
pred.remove_first(800)

true = gt.Data()
# true.parallel_orca(keywords=GradientKeywords(['DLPNO-CCSD(T)', 'TightPNO', 'def2-TZVP/C', 'RIJCOSX', 'def2/J', 'TightSCF', 'def2-TZVP']))
# true.save(filename='traj_ccsdt.xyz')
true.load('traj_ccsdt.xyz')
true_energies = ev_to_kcal * true.energies()

gmx_energies = ev_to_kcal * np.loadtxt('gmx_acoh_energies.txt')

plt.plot(np.linspace(400, 500, len(pred)),                  # 0 -> 50 fs
         ev_to_kcal * pred.energies() - np.min(true_energies),      # rel energies
         label='GAP', c='tab:blue', lw=2)

plt.plot(np.linspace(400, 500, len(true)),
         true_energies - np.min(true_energies),
         label='true', c='black', lw=3)

plt.plot(np.linspace(400, 500, len(gmx_energies)),
         gmx_energies - np.min(gmx_energies),
         label='MM', c='tab:orange', lw=2, alpha=0.3)

# plot the region of 'chemical accuracy' 1 kcal mol-1 = 0.043 eV
plt.fill_between(np.linspace(400, 500, len(true)),
                 y1=true_energies - np.min(true_energies) -1,# - 0.043,
                 y2=true_energies - np.min(true_energies) +1,# + 0.043,
                 alpha=0.2, color='black')

plt.xlabel('time / fs')
plt.ylabel('$E$ / kcal mol$^{-1}$')
plt.xlim(400, 500)
plt.legend()
plt.tight_layout()
plt.savefig('energies_vs_time_AcOH.pdf')
