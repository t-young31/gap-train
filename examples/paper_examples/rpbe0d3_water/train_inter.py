import gaptrain as gt
from gaptrain.solvents import get_solvent
gt.GTConfig.n_cores = 10

h2o = gt.System(box_size=[8, 8, 8])
h2o.add_solvent('h2o', n=17)

# Now create an intra GAP that has the molecule indexes
intra_gap = gt.gap.IntraGAP(name='monomer_2b_3b',
                            system=h2o,
                            molecule=get_solvent('h2o'))

inter_gap = gt.InterGAP(name=f'inter_h2o',
                        system=h2o)
inter_gap.params.soap['O']['cutoff'] = 4.0

# And finally train the inter component of the energy
inter_data, gap = gt.active.train(h2o,
                                  method_name='cp2k',
                                  temp=300,
                                  max_energy_threshold=5,
                                  gap=gt.IIGAP(intra_gap, inter_gap),
                                  active_e_thresh=0.17)

# Run some sample MD
traj = gt.md.run_gapmd(configuration=h2o.random(min_dist_threshold=1.7),
                       gap=gap,
                       temp=300,
                       dt=0.5,
                       interval=5,
                       ps=1,
                       n_cores=4)

traj.save(filename='traj.xyz')
