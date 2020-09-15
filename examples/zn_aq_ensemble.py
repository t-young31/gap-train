import gaptrain as gt
import matplotlib.pyplot as plt
gt.GTConfig.n_cores = 8

if __name__ == '__main__':

    zn_aq = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_aq.add_molecules(gt.Molecule('h2o.xyz'), n=52)

    vd1 = gt.Data()
    vd1.load('Zn_DFTBMD_data.xyz', system=zn_aq)

    ensemble = gt.GAPEnsemble(name='zn_test',
                              system=zn_aq,
                              num=5)
    ensemble.train(vd1)

    errors = []
    for config in vd1[::5]:
        error = ensemble.predict_energy_error(config)
        errors.append(error)

    plt.plot(list(range(len(errors))), errors,
             marker='o')
    plt.savefig('predicted_errors.png', dpi=400)
