"""
Train a GAP for a single molecule in the gas phase, here benzene
"""
import gaptrain as gt
gt.GTConfig.n_cores = 4


benzene = gt.System(box_size=[10, 10, 10])
benzene.add_molecules(gt.Molecule('C6H6.xyz'))

data, gap = gt.active.train(benzene,
                            method_name='dftb',
                            validate=False,
                            temp=1000)
