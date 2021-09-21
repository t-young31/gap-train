"""Run some molecular dynamics with a solute-solvent GAP"""
import gaptrain as gt
gt.GTConfig.n_cores = 10

# Initialise a box of water and methane
system = gt.System(box_size=[10, 10, 10])
system.add_molecules(gt.Molecule('methane.xyz'))
system.add_solvent('h2o', n=20)

# Load the GAP parameter files
solute_gap = gt.IntraGAP(name=f'intra_CH4',
                         unique_molecule=system.unique_molecules[0])
solv_gap = gt.IntraGAP(name=f'intra_h2o',
                       unique_molecule=system.unique_molecules[1])

inter_gap = gt.InterGAP(name='inter', system=system)

# Run molecular dynamics using the composite GAP
traj = gt.md.run_gapmd(configuration=system.random(min_dist_threshold=1.7),
                       gap=gt.IIGAP(inter_gap, solv_gap, solute_gap),
                       temp=300,
                       dt=0.5,
                       interval=5,
                       ps=1,
                       n_cores=4)

traj.save(filename='traj.xyz')
