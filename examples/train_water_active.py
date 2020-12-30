import gaptrain as gt
gt.GTConfig.n_cores = 4


h2o = gt.System(box_size=[10, 10, 10])
h2o.add_solvent('h2o', n=1)

data, gap = gt.active.train(h2o, method_name='dftb', validate=True)
