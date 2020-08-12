from gaptrain import *
import matplotlib.pyplot as plt


zn_h2o = System(Ion('Zn', charge=2),
                box_size=[12, 12, 12])
zn_h2o.add_molecules(Molecule(xyz_filename='h2o.xyz'), n=52)

# Load the validation data
validation_data = Data(name='Zn_DFTBMD_data')
validation_data.load(system=zn_h2o)

# Check the energy and force range of the data
# validation_data.histogram()

training_data = Data(name='Zn_random_train')
gap = GAP(name='Zn_random_train', system=zn_h2o)

rmses = []

for i in range(10):

    configs = ConfigurationSet()
    for _ in range(20):

        config = zn_h2o.random(min_dist_threshold=1.2)
        configs += config

    # Compute energies and forces and add to the training data
    configs.async_dftb()
    training_data += configs

    # Train the GAP
    gap.train(training_data)
    gap.predict(validation_data)

    rmses.append((validation_data.energy_rmse(),
                  validation_data.force_rmse()))
