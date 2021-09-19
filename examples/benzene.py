"""Train a GAP for a gas phase benzene molecule with a DFTB ground truth"""
import gaptrain as gt
gt.GTConfig.n_cores = 10

# Initialise a cubic box 10x10x10 Å containing a single benzene molecule
benzene = gt.System(box_size=[10, 10, 10])
benzene.add_molecules(gt.Molecule('benzene.xyz'))

# Train a GAP at 1000 K using DFTB as the ground truth method
data, gap = gt.active.train(benzene,
                            method_name='dftb',
                            validate=False,
                            temp=1000)

# Run 1 ps molecular dynamics using the GAP at 300 K using a 0.5 fs time-step.
# The initial configuration is methane located at a random position in the box
traj = gt.md.run_gapmd(configuration=benzene.random(),
                       gap=gap,
                       temp=300,           # Kelvin
                       dt=0.5,             # fs
                       interval=5,         # frames
                       ps=1,
                       n_cores=4)

# and save the trajectory for visualisation in Avogadro, VMD, PyMol etc.
traj.save(filename='traj.xyz')
