import sys
import gaptrain as gt
import autode as ade
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 10

idx = int(sys.argv[2])

mol = ade.Molecule(smiles=sys.argv[1])
mol.optimise(method=ade.methods.XTB())
mol.print_xyz_file()

system = gt.System(box_size=[50, 50, 50])
system.add_molecules(gt.Molecule(f'{mol.name}.xyz'))

for _ in range(3):
    data, gap = gt.active.train(system,
                                method_name='xtb',
                                max_active_iters=100,
                                temp=500)

    tau = gt.Tau(configs=[system.configuration() for _ in range(3)],
                         e_lower=0.04336,
                         temp=300)
    tau.calculate(gap=gap, method_name='xtb')

    with open(f'{idx}_tau.txt', 'a') as tau_file:
        print(sum(c.n_evals for c in data), len(data), str(tau), file=tau_file)
