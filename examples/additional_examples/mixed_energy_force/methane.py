import gaptrain as gt
import numpy as np
import matplotlib.pyplot as plt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 10
# plt.style.use('paper')


def generate_dftb_trajectory():
    """Generate some configurations to use as reference"""

    traj = gt.md.run_dftbmd(methane.configuration(),
                            temp=500,
                            dt=0.5,
                            interval=10,
                            ps=1)

    traj.save(filename='methane_dftb_md.xyz')
    return None


def generate_energies_forces(trajectory):
    """Calculate energies and forces for the frames at different levels of theory"""

    frames = trajectory.copy()
    frames.parallel_dftb()
    frames.save(filename='dftb_frames.xyz')
    np.savetxt('dftb_frames_F.txt', frames.force_components())

    frames = trajectory.copy()
    frames.parallel_xtb()
    frames.save(filename='xtb_frames.xyz')
    np.savetxt('xtb_frames.txt', frames.force_components())

    gt.GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])
    frames = trajectory.copy()
    frames.parallel_orca()
    frames.save(filename='pbe_frames.xyz')
    np.savetxt('pbe_frames.txt', frames.force_components())

    gt.GTConfig.orca_keywords = GradientKeywords(['PBE0', 'def2-SVP', 'EnGrad'])
    frames = trajectory.copy()
    frames.parallel_orca()
    frames.save(filename='pbe0_frames.xyz')
    np.savetxt('pbe0_frames.txt', frames.force_components())

    gt.GTConfig.orca_keywords = GradientKeywords(['MP2', 'def2-TZVP', 'EnGrad'])
    frames = trajectory.copy()
    frames.parallel_orca()
    frames.save(filename='mp2_frames.xyz')
    np.savetxt('mp2_frames.txt', frames.force_components())

    gt.GTConfig.orca_keywords = GradientKeywords(['CCSD(T)', 'def2-TZVP', 'EnGrad', 'NumGrad'])
    frames = trajectory.copy()
    frames.parallel_orca()
    frames.save(filename='ccsdt_frames.xyz')
    np.savetxt('ccsdt_frames_F.txt', frames.force_components())

    return None


def mad(array):
    """Mean absolute deviation"""
    return np.average(np.abs(array))


def plot_force_component_comparison(name, axis_idx):
    """Parity plot of some forces vs 'true' values"""

    ref_forces = np.loadtxt('ccsdt_frames_F.txt')
    forces = np.loadtxt(f'{name}_frames_F.txt')
    min_f, max_f = -5, 5

    plot_ax = ax[idx[0], idx[1]]
    plot_ax.scatter(ref_forces, forces, marker='o', lw=1,
                    edgecolors='blue',
                    alpha=0.4)

    plot_ax.plot([min_f, max_f], [min_f, max_f], c='k')
    plot_ax.annotate(f'MAD = {np.round(mad(ref_forces - forces), 2)} eV Å-1',
                     (0, -4))

    plot_ax.set_ylim(min_f, max_f)
    plot_ax.set_xlim(min_f, max_f)
    plot_ax.set_xlabel('$F_{CCSD(T)}$ / eV Å$^{-1}$')
    plot_ax.set_ylabel('$F_{'+name+'}$ / eV Å$^{-1}$')

    return None


if __name__ == '__main__':

    methane = gt.System(gt.Molecule('methane.xyz'),
                        box_size=[50, 50, 50])

    # generate_dftb_trajectory()
    # traj = gt.Trajectory('methane_dftb_md.xyz', charge=0, mult=1,
    #                      box=gt.Box([50, 50, 50]))
    # generate_energies_forces(traj)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
    fig.delaxes(ax[1, 2])
    for method_name, idx in zip(('dftb', 'xtb', 'pbe', 'pbe0', 'mp2'),
                                ([0, 0], [0, 1], [0, 2], [1, 0], [1, 1])):
        plot_force_component_comparison(name=method_name,
                                        axis_idx=idx)
    plt.tight_layout()
    plt.savefig(f'force_comparison_vs_ccsdt.pdf')
