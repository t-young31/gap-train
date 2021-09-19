import gaptrain as gt
import autode as ade
from autode.wrappers.keywords import GradientKeywords

gt.GTConfig.n_cores = 10


def find_ts():
    """Find the transition state for the [4+2] Diels-Alder reaction between
    ethene and butadiene"""
    ade.Config.n_cores = 8
    ade.Config.ORCA.keywords.set_functional('PBE')

    rxn = ade.Reaction('C=CC=C.C=C>>C1=CCCCC1', name='DA')
    rxn.locate_transition_state()
    rxn.ts.print_xyz_file(filename='ts.xyz')
    return None


# First find the transition state and save ts.xyz
find_ts()

# Set the keywords to use for an ORCA energy and gradient calculation as low-
# level PBE with a double zeta basis set
gt.GTConfig.orca_keywords = GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])

# Initialise a cubic box 10x10x10 Å containing a single methane molecule
da_ts = gt.System(box_size=[10, 10, 10])
da_ts.add_molecules(gt.Molecule('ts.xyz'))

# and train the GAP using a 0.1 eV threshold (~2 kcal mol-1), here the initial
# configuration from which dynamics is propagated from needs to be fixed to
# the TS (i.e. the first configuration) for good sampling around the TS
data, gap = gt.active.train(da_ts,
                            method_name='orca',
                            temp=500,
                            active_e_thresh=0.1,
                            max_time_active_fs=200,
                            fix_init_config=True)

# 'uplift' the configurations obtained at PBE/DZ to MP2/TZ
gt.GTConfig.orca_keywords = GradientKeywords(['RI-MP2', 'def2-TZVP', 'TightSCF',
                                              'AutoAux', 'NoFrozenCore', 'EnGrad'])
data.parallel_orca()

# and retrain the GAP with the new energies and forces on the PBE data
gap.train(data)

# Run a 400 fs molecular dynamics (NVT) using up-lifted GAP at 50 K using a 0.5
# fs time-step
traj = gt.md.run_gapmd(configuration=da_ts.random(),
                       gap=gap,
                       temp=50,      # Kelvin
                       dt=0.5,       # fs
                       interval=1,   # frames
                       fs=400,
                       n_cores=4)

traj.save('da_mp2_traj.xyz')
