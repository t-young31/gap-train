import gaptrain as gt
from gaptrain.exceptions import GAPFailed
gt.GTConfig.n_cores = 8

zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

# Load the validation data
validation = gt.Data(name='Zn_DFTBMD_data')
validation.load(system=zn_h2o)


# Minimum energy from DFTB+ MD run
ref_energy = -5877.02169783
e_thresh = 70

out_file = open(f'out_{e_thresh}.txt', 'w')


print(f'Energy threshold = {e_thresh}\n RMSE(E_train), RMSE(|F|_train),'
      f' RMSE(E_val), RMSE(|F|_val), # calculations',
      file=out_file)

for i in range(5):

    training_data = gt.Data(name=f'Zn_random_train_{e_thresh}_{i}')
    gap = gt.GAP(name=f'Zn_random_train_{e_thresh}_{i}',
                 system=zn_h2o)
    n_calcs = 0

    while len(training_data) < 500:
        configs = gt.ConfigurationSet()
        for _ in range(gt.GTConfig.n_cores):
            configs += zn_h2o.random(min_dist_threshold=1.4)

        # Compute energies and forces and add to the training data
        configs.parallel_dftb()
        n_calcs += gt.GTConfig.n_cores

        # Add configurations that are below the threshold maximum energy
        # and also there are no more than 500 configurations in the training
        for config in configs:
            if (config.energy - ref_energy < e_thresh
                    and len(training_data) < 500):

                training_data += config

        print(len(training_data), n_calcs)

    try:
        gap.train(training_data)
    except GAPFailed:
        pass

    # Predict on the validation data
    predictions = gap.predict(validation)
    val = gt.RMSE(validation, predictions)

    # and the training data
    predictions = gap.predict(training_data)
    train = gt.RMSE(training_data, predictions)

    print(f'{train.energy},{train.force},{val.energy},{val.force},{n_calcs}',
          file=out_file)

    # Temp histogram to check that the threshold is working
    training_data.histogram(name='training_data', ref_energy=ref_energy)

out_file.close()
