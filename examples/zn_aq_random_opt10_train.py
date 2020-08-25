import gaptrain as gt
from gaptrain.exceptions import GAPFailed
gt.GTConfig.n_cores = 8


def add_n_configs(training_data, n):

    configs = gt.ConfigurationSet()
    for _ in range(n):
        configs += zn_h2o.random(min_dist_threshold=1.4)

    # Compute energies and forces and add to the training data
    configs.parallel_dftb(max_force=10)
    training_data += configs

    # Try to train the GAP
    try:
        gap.train(training_data)
    except GAPFailed:
        exit()
        return None

    # Predict on the validation data
    predictions = gap.predict(validation)
    val_errs.append(gt.RMSE(validation, predictions))

    # and the training data
    predictions = gap.predict(training_data)
    train_errs.append(gt.RMSE(training_data, predictions))

    return None


if __name__ == '__main__':

    zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

    # Load the validation data
    validation = gt.Data(name='Zn_DFTBMD_data')
    validation.load(filename='Zn_DFTBMD_data.xyz',
                    system=zn_h2o)

    # Set up the training data and the GAP
    training_data = gt.Data(name='Zn_random_opt10_train')
    gap = gt.GAP(name='Zn_random_opt10_train', system=zn_h2o)

    train_errs, val_errs = [], []

    for i in range(50):
        try:
            add_n_configs(training_data, n=10)

        except GAPFailed:
            break

    with open('out.txt', 'w') as out_file:
        print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(|F|_val)',
              file=out_file)

        for train, val in zip(train_errs, val_errs):
            print(f'{train.energy},{train.force},{val.energy},{val.force}',
                  file=out_file)

    # Minimum energy from DFTB+ MD run
    training_data.histogram(name='training_data', ref_energy=-5877.02169783)
