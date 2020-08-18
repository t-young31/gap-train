from gaptrain import *
from gaptrain.loss import RMSE
import matplotlib.pyplot as plt
GTConfig.n_cores = 8

zn_h2o = System(Ion('Zn', charge=2),
                box_size=[12, 12, 12])
zn_h2o.add_molecules(Molecule(xyz_filename='h2o.xyz'), n=52)

# Load the validation data
validation = Data(name='Zn_DFTBMD_data')
validation.load(system=zn_h2o)

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
    configs.parallel_dftb()
    training_data += configs

    # Train the GAP
    gap.train(training_data)
    predictions = gap.predict(validation)

    rmses.append(RMSE(validation, predictions))
    print(rmses[-1])

# Plot the learning curve..
plt.plot(list(range(len(rmses))),
         [rmse.energy for rmse in rmses])
