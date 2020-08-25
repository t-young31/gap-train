import gaptrain as gt
from gaptrain.exceptions import GAPFailed
gt.GTConfig.n_cores = 12

zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
zn_h2o.add_molecules(gt.Molecule(xyz_filename='h2o.xyz'), n=52)

# Load the validation data
validation = gt.Data(name='Zn_DFTBMD_data')
validation.load(filename='Zn_DFTBMD_data.xyz',
                system=zn_h2o)

training_data = gt.Data(name='Zn_random_train')
gap = gt.GAP(name='Zn_random_train', system=zn_h2o)

train_errs, val_errs = [], []

out_file = open('out.txt', 'w')
print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(|F|_val)', file=out_file)

for i in range(10):

    configs = gt.ConfigurationSet()
    for _ in range(50):
        configs += zn_h2o.random(min_dist_threshold=1.4)

    # Compute energies and forces and add to the training data
    configs.parallel_dftb()
    training_data += configs

    # Try to train the GAP
    try:
        gap.train(training_data)
    except GAPFailed:
        break

    # Predict on the validation data
    predictions = gap.predict(validation)
    val = gt.RMSE(validation, predictions)

    # and the training data
    predictions = gap.predict(training_data)
    train = gt.RMSE(training_data, predictions)

    print(f'{train.energy},{train.force},{val.energy},{val.force}', file=out_file)

out_file.close()

# Minimum energy from DFTB+ MD run
training_data.histogram(name='training_data', ref_energy=-5877.02169783)
