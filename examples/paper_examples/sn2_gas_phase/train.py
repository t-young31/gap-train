import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords

gt.GTConfig.n_cores = 8

if __name__ == '__main__':

    gt.GTConfig.orca_keywords = GradientKeywords(['PBE', 'ma-def2-SVP', 'EnGrad'])

    # large box to ensure no self-interaction
    sn2_ts = gt.System(box_size=[20, 20, 20])
    sn2_ts.add_molecules(gt.Molecule('ts.xyz', charge=-1))

    gap = gt.GAP(name='sn2_gap', system=sn2_ts, default_params=False)
    gap.params.soap['C'] = gt.GTConfig.gap_default_soap_params
    gap.params.soap['C']['other'] = ['H', 'Cl']
    gap.params.soap['C']['cutoff'] = 6.0

    data, gap = gt.active.train(sn2_ts,
                                method_name='orca',
                                temp=500,
                                active_e_thresh=0.1,
                                max_time_active_fs=500,
                                fix_init_config=True)

    # 'uplift' the configurations obtained at PBE/DZ to MP2/TZ
    gt.GTConfig.orca_keywords = GradientKeywords(['DLPNO-CCSD(T)', 'ma-def2-TZVPP',
                                                  'NumGrad', 'AutoAux', 'EnGrad'])
    data.parallel_orca()
    gap.train(data)

    # Run a sample set of dynamics from the TS
    traj = gt.md.run_gapmd(configuration=sn2_ts.configuration(),
                           gap=gap,
                           temp=50,  # Kelvin
                           dt=0.5,  # fs
                           interval=1,  # frames
                           fs=400,
                           n_cores=4)

    traj.save('sn2_example_traj.xyz')
