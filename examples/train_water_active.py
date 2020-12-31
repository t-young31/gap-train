import gaptrain as gt
gt.GTConfig.n_cores = 4


h2o = gt.System(box_size=[10, 10, 10])
h2o.add_solvent('h2o', n=1)

# data, gap = gt.active.train(h2o, method_name='dftb', validate=False)

water = gt.System(box_size=[10, 10, 10])
water.add_solvent('h2o', n=10)

intra_gap = gt.IntraGAP(name='active_gap_h2o', system=water)
inter_gap = gt.InterGAP(name='inter', system=water)

# data, gap = gt.active.train(water,
#                             method_name='dftb',
#                             validate=False,
#                             gap=gt.IIGAP(intra_gap, inter_gap))


traj = gt.md.run_gapmd(configuration=water.random(min_dist_threshold=2.0),
                       gap=gt.IIGAP(intra_gap, inter_gap),
                       temp=300,
                       dt=0.5,
                       interval=5,
                       fs=500,
                       n_cores=4)

traj[0].save(filename='init.xyz')
traj.save(filename='tmp.xyz')
