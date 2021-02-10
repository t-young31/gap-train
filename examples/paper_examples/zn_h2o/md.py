import gaptrain as gt
gt.GTConfig.n_cores = 4


if __name__ == '__main__':

    system = gt.System(box_size=[12.42, 12.42, 12.42])
    system.add_molecules(gt.Molecule('znh2o6.xyz', charge=2))
    system.add_solvent('h2o', n=58)

    intra_h2o_gap = gt.gap.SolventIntraGAP(name='water_intra_gap',
                                           system=system)

    intra_znh2o6_gap = gt.gap.SoluteIntraGAP(name='intra_znh2o6',
                                             system=system,
                                             molecule=gt.Molecule(
                                                 'znh2o6.xyz'))
    inter_gap = gt.InterGAP(name='inter', system=system, default_params=False)

    # Run 30 ps of dynamics from an equilibrated point
    traj = gt.md.run_gapmd(configuration=gt.Data('eqm_final_frame.xyz')[0],
                       gap=gt.gap.SSGAP(solute_intra=intra_znh2o6_gap,
                                        solvent_intra=intra_h2o_gap,
                                        inter=inter_gap),
                       temp=300,
                       dt=0.5,
                       interval=5,
                       ps=30,
                       n_cores=4)

    traj.save(filename=f'zn_h2o_traj')
