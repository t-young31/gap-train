import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords

gt.GTConfig.n_cores = 10
gt.GTConfig.orca_keywords = GradientKeywords(['B3LYP', 'def2-SVP', 'EnGrad'])

# For non-periodic systems there's no need to define a box, but a System
# requires one
ts = gt.System(box_size=[10, 10, 10])
ts.add_molecules(gt.Molecule('ts1_prime.xyz'))

gap = gt.GAP(name='da_gap', system=ts, default_params={})
gap.params.soap['C'] = gt.GTConfig.gap_default_soap_params
gap.params.soap['C']['cutoff'] = 3.0
gap.params.soap['C']['other'] = ['H', 'C']

data, gap = gt.active.train(system=ts,
                            method_name='orca',
                            gap=gap,
                            max_time_active_fs=200,
                            temp=500,
                            active_e_thresh=3*0.043,      # 3 kcal mol-1
                            max_energy_threshold=5,
                            max_active_iters=50,
                            n_init_configs=10,
                            fix_init_config=True)
