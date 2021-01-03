import gaptrain as gt
gt.GTConfig.n_cores = 10

if __name__ == '__main__':

    system = gt.System(box_size=[8, 8, 8])
    system.add_molecules(gt.Molecule('methane.xyz'))
    system.add_solvent('h2o', n=20)

    data, gap = gt.active.train_ss(system, method_name='dftb')
