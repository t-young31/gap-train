import gaptrain as gt
gt.GTConfig.n_cores = 10

# Initialise a box of water and methane
system = gt.System(box_size=[10, 10, 10])
system.add_molecules(gt.Molecule('methane.xyz'))
system.add_solvent('h2o', n=20)

# Train a solute-solvent GAP and generate parameter files
data, gap = gt.active.train_ii(system, method_name='dftb')
