from gaptrain import *
from gaptrain.md import run_dftbmd
GTConfig.n_cores = 8

h2o = Molecule(xyz_filename='h2o.xyz')

zn_h2o = System(Ion('Zn', charge=2),
                box_size=[12, 12, 12])
zn_h2o.add_molecules(h2o, n=52)

# Generate a random configuration and minimise
conf = zn_h2o.random(on_grid=True, min_dist_threshold=1.2)
conf.run_dftb(max_force=10)

# Generate a trajectory from a 10ps MD simulation
trajectory = run_dftbmd(conf,
                        ps=20,
                        dt=0.5,
                        temp=300,
                        interval=1)

# Remove the first 5 ps of the trajectory
n_frames = int(5 * 1000 / (5 * 0.5))
trajectory.remove_first(n=n_frames)

# Add the equilibrated portion of the trajectory to the data
validation_data = Data(name='Zn_DFTBMD_data')
validation_data += trajectory

# Select 100 random points from the trajectory
validation_data.remove_random(remainder=100)
validation_data.async_dftb()

# Save the ground truth DFTB+ energy and forces
validation_data.save_true()
