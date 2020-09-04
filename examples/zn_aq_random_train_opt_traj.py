import gaptrain as gt
from gaptrain.calculators import run_dftb
from gaptrain.exceptions import GAPFailed
import os
gt.GTConfig.n_cores = 8


def train(n_configs=5):

    training_data = gt.Data(name=f'Zn_random_train_{n_points}_{1}')
    gap = gt.GAP(name=f'Zn_random_train_{n_points}_{1}',
                 system=zn_h2o)

    configs = gt.ConfigurationSet()
    for _ in range(n_configs):
        configs += zn_h2o.random(min_dist_threshold=1.4)

    all_configs = gt.ConfigurationSet()
    for config in configs:

        run_dftb(config, max_force=10, traj_name='tmp.traj')
        traj = gt.Trajectory('tmp.traj', init_configuration=config)
        os.remove('tmp.traj')

        stride = int(len(traj)/n_points)

        # Take the final most points from the list using a stride at least 1
        all_configs += traj[::-max(stride, 1)]

    # Truncate back down to 500 configurations
    all_configs.remove_random(remainder=n_configs)
    training_data += all_configs

    try:
        gap.train(training_data)
    except GAPFailed:
        return False

    # Predict on the validation data
    predictions = gap.predict(validation)
    val = gt.RMSE(validation, predictions)

    # and the training data
    predictions = gap.predict(training_data)
    train_err = gt.RMSE(training_data, predictions)

    print(f'{train_err.energy},{train_err.force},{val.energy},{val.force}',
          file=out_file)

    return True


if __name__ == '__main__':

    # Aqueous Zn(II) system
    zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

    # Load the validation data
    validation = gt.Data(name='Zn_DFTBMD_data')
    validation.load(system=zn_h2o)

    n_points = 1

    with open('out.txt', 'w') as out_file:
        print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(F_val)',
              file=out_file)

        # Several iterations the accuracy depends on the random configurations
        # added during the training
        n_trained = 0

        while n_trained < 5:
            success = train()

            if success:
                n_trained += 1
