import gaptrain as gt
from autode.wrappers.keywords import GradientKeywords
gt.GTConfig.n_cores = 10
gt.GTConfig.orca_keywords = GradientKeywords(['MP2', 'TightSCF', 'def2-TZVP', 'EnGrad'])


def mp2_all_configs():
    """Evaluate MP2 energy and forces on all XTB active configurations"""

    data = gt.Data('xtb_active.xyz')
    data.parallel_orca()
    data.save(filename='xtb_active_mp2.xyz')

    return


if __name__ == '__main__':

    system = gt.System(box_size=[10, 10, 10])
    system.add_molecules(gt.Molecule('MeOH.xyz'))

    tau_file = open('tau.txt', 'w')
    print('n \t τ_acc', file=tau_file)

    for n in (190, 160, 130, 100, 70, 40):
        trunc_data = gt.Data('xtb_active_mp2.xyz')
        trunc_data.truncate(n=n, method='cur')

        gap = gt.GAP(name=f'mp2_{n}_configs', system=system)
        gap.train(trunc_data)

        tau = gt.Tau(configs=[system.configuration() for _ in range(3)],
                     e_lower=0.04336,
                     temp=500)
        tau.calculate(gap=gap, method_name='orca')
        print(n, str(tau), file=tau_file)

    tau_file.close()
