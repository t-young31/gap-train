import sys
import gaptrain as gt
import autode as ade
gt.GTConfig.n_cores = 10

n_carbons = int(sys.argv[1])

mol = ade.Molecule(smiles='C'*n_carbons)
mol.optimise(method=ade.methods.XTB())
mol.print_xyz_file(filename=f'C{n_carbons}.xyz')

system = gt.System(box_size=[50, 50, 50])
system.add_molecules(gt.Molecule(f'C{n_carbons}.xyz'))

data, gap = gt.active.train(system,
                            method_name='xtb',
                            max_active_iters=100,
                            temp=500)

tau = gt.Tau(configs=[system.configuration() for _ in range(3)],
             e_lower=0.04336,  # 1 kcal mol-1
             temp=300)
tau.calculate(gap=gap, method_name='xtb')

with open(f'C{n_carbons}_tau.txt', 'a') as tau_file:
    print(sum(c.n_evals for c in data), len(data), str(tau), file=tau_file)

data.save(filename=f'C{n_carbons}_active.xyz')
