import gaptrain as gt

system = gt.System(box_size=[10, 10, 10])
system.add_molecules(gt.Molecule('methane.xyz'))
system.add_solvent('h2o', n=20)

solv_gap = gt.gap.SolventIntraGAP(name=f'intra_h2o', system=system)
solute_gap = gt.gap.SoluteIntraGAP(name=f'intra_CH4',
                                   system=system,
                                   molecule=gt.Molecule('methane.xyz'))
inter_gap = gt.InterGAP(name='inter', system=system)


traj = gt.md.run_gapmd(configuration=system.random(min_dist_threshold=1.7),
                       gap=gt.gap.SSGAP(solute_intra=solute_gap,
                                        solvent_intra=solv_gap,
                                        inter=inter_gap),
                       temp=300,
                       dt=0.5,
                       interval=5,
                       ps=10,
                       n_cores=4)

traj.save(filename='traj.xyz')
