import gaptrain as gt
gt.GTConfig.n_cores = 4

if __name__ == '__main__':

    # Initilise a box of containing benzene in the gas phase
    benzene = gt.System(box_size=[10, 10, 10])
    benzene.add_molecules(gt.Molecule('C6H6.xyz'))

    # Train a GAP
    data, gap = gt.active.train(benzene,
                                method_name='dftb',
                                validate=False,
                                temp=1000)

    # Run molecular dynamics using the GAP
    traj = gt.md.run_gapmd(configuration=benzene.random(),
                           gap=gap,
                           temp=300,
                           dt=0.5,
                           interval=5,
                           ps=1,
                           n_cores=4)

    traj.save(filename='traj.xyz')
