import gaptrain as gt
gt.GTConfig.n_cores = 4

# Initialise a box of water
h2o = gt.System(box_size=[10, 10, 10])
h2o.add_solvent('h2o', n=20)

# Train a intra+inter GAP and generate parameter files
# Note: train functions return a tuple of the data used to train and the
# resulting GAP
data, gap = gt.active.train_ii(h2o, method_name='dftb')

# Run molecular dynamics using the GAP
traj = gt.md.run_gapmd(configuration=h2o.random(min_dist_threshold=1.7),
                       gap=gap,
                       temp=300,
                       dt=0.5,
                       interval=5,
                       ps=1,
                       n_cores=4)

traj.save(filename='traj.xyz')
