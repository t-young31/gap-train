import gaptrain as gt
from gaptrain.exceptions import GAPFailed
gt.GTConfig.n_cores = 8


def train(n_configs=500):

    training_data = gt.Data(name=f'Zn_random_train_{proportion_kept}')
    gap = gt.GAP(name=f'Zn_random_train_{proportion_kept}',
                 system=zn_h2o)

    configs = gt.ConfigurationSet()
    for _ in range(int(n_configs / proportion_kept)):
        configs += zn_h2o.random(min_dist_threshold=1.4)

    configs.truncate(n=n_configs, method='CUR')

    # Compute energies and forces and add to the training data
    configs.parallel_dftb()
    training_data += configs

    try:
        gap.train(training_data)
    except GAPFailed:
        return None

    # Predict on the validation data
    predictions = gap.predict(validation)
    val = gt.RMSE(validation, predictions)

    # and the training data
    predictions = gap.predict(training_data)
    train_err = gt.RMSE(training_data, predictions)

    print(f'{train_err.energy},{train_err.force},{val.energy},{val.force}',
          file=out_file)

    return None


if __name__ == '__main__':

    # Aqueous Zn(II) system
    zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

    # Load the validation data
    validation = gt.Data(name='Zn_DFTBMD_data')
    validation.load(system=zn_h2o)

    e_thresh = 40
    proportion_kept = 0.1

    with open('out.txt', 'w') as out_file:
        print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(F_val)',
              file=out_file)

        # Several iterations the accuracy depends on the random configurations
        # added during the training
        for i in range(10):
            train()
