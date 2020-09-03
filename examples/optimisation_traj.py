import gaptrain as gt
from gaptrain.calculators import run_dftb


if __name__ == '__main__':

    system = gt.System(box_size=[10, 10, 10])
    system.add_molecules(gt.Molecule('h2o.xyz'), n=10)

    # Run a quick DFTB+ optimisation on the water molecule
    config = system.random()
    run_dftb(config, max_force=0.1, traj_name='test.traj')

    traj = gt.Trajectory('test.traj',
                         init_configuration=config)

    traj.save()
