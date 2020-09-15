import gaptrain as gt
from gaptrain.calculators import run_gap
from gaptrain.exceptions import GAPFailed
from multiprocessing import Pool
import os
gt.GTConfig.n_cores = 8


def train(init_gap, n_configs=500, ref_energy = -5877.02169783):

    training_data = gt.Data(name=f'Zn_random_train_{n_trained}')
    gap = gt.GAP(name=f'Zn_random_train_{n_trained}',
                 system=zn_h2o)

    # Generate n_configs Configurations which have energy lower than the
    # threshold
    configs = gt.ConfigurationSet()
    while len(configs) < n_configs:

        # Parallel compute using a single core over some random configurations
        any_energy_configs = gt.ConfigurationSet()
        for _ in range(gt.GTConfig.n_cores):
            any_energy_configs += zn_h2o.random(min_dist_threshold=1.4)

        any_energy_configs.parallel_dftb()

        # Add those with an energy lower than the threshold
        for config in any_energy_configs:
            if config.energy - ref_energy < e_thresh:
                configs.add(config)

    # From low-ish energy configurations generate a larger set of
    # configurations including minimisation trajectories
    all_configs = gt.ConfigurationSet()

    for config in configs:
        config.add_perturbation(sigma=0.05)

    results = []
    with Pool(processes=gt.GTConfig.n_cores) as pool:

        # Apply the method to each configuration in this set
        for i, config in enumerate(configs):
            result = pool.apply_async(func=run_gap,
                                      args=(config, max_f, init_gap, f'tmp{i}.traj'))
            results.append(result)

        # Reset all the configurations in this set with updated energy
        # and forces (each with .true)
        for i, result in enumerate(results):
            configs[i] = result.get(timeout=None)
            traj = gt.Trajectory(f'tmp{i}.traj', init_configuration=configs[i])
            os.remove(f'tmp{i}.traj')

            stride = int(len(traj)/n_points)

            # Take the final most points from the list using a stride > 1
            all_configs += traj[::-max(stride, 1)]

    # Truncate back down to 500 configurations randomly
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


def train_init_gap(n_configs=500):

    training_data = gt.Data(name=f'Zn_random_train_init')
    gap = gt.GAP(name=f'Zn_random_train_init',
                 system=zn_h2o)

    # Add some random and perturbed Configurations to the training data
    for _ in range(n_configs):

        config = zn_h2o.random(min_dist_threshold=1.4)
        config.add_perturbation(sigma=0.05)
        training_data += config

    # Calculate energies and forces
    training_data.parallel_dftb()

    try:
        gap.train(training_data)

    # Call recursively until training is successful
    except GAPFailed:
        return train_init_gap(n_configs)

    return gap


if __name__ == '__main__':

    # Aqueous Zn(II) system
    zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

    # Load the validation data
    validation = gt.Data(name='Zn_DFTBMD_data')
    validation.load(system=zn_h2o)

    e_thresh = 40
    n_points = 10
    max_f = 10

    init_gap = train_init_gap(n_configs=500)

    with open('out.txt', 'w') as out_file:
        print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(F_val)',
              file=out_file)

        # Several iterations the accuracy depends on the random configurations
        # added during the training
        n_trained = 0

        while n_trained < 5:
            success = train(init_gap)

            if success:
                n_trained += 1
