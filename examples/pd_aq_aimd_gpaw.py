from gaptrain import *
GTConfig.n_cores = 24

# Generate a Pd(II)(aq) system in a 12 A^3 box with 52 water molecules
h2o = Molecule(xyz_filename='h2o.xyz')
pd_h2o = System(Ion('Pd', charge=2),
                box_size=[12.0, 12.0, 12.0])
pd_h2o.add_molecules(h2o, n=52)

# Initialise a configuration set
data = Data(name='AIMD_validation_GPAW')

# Load the AIMD trajectory with 7858 frames that are saved every 5, using
# a 0.5 fs timestep from a relaxed but not equilibrated NVT simulation
data.load(filename='PD_WATER-pos-1.xyz', system=pd_h2o)

# Remove the first 5 ps for equilibration then leave only 100 based on
# a random selection
n_frames = int(5 * 1000 / (5 * 0.5))

data.remove_first(n=n_frames)
data.remove_random(remainder=100)

# Run GPAW on all the data
data.parallel_gpaw()

# Save the ground truth DFTB+ coordinates, energy and forces
data.save()
