from gaptrain import *
GTConfig.n_cores = 4

# Create a system of Na(I) in a 10 Å^3 box with 40 water molecules
system = System(Ion('Na', charge=1),
                box_size=[10, 10, 10])

system.add_molecules(Molecule('h2o.xyz'), n=10)

# Initialise the training and test data
training_data = Data(name='random_training')
test_data = Data(name='test')

# Generate 10 random configurations
configs = ConfigurationSet()
for _ in range(10):
    configs += system.random(min_dist_threshold=2.0)

# DFT in parallel
configs.async_dftb()

# Add to the training data
training_data += configs

# Initialise and train a GAP
gap = GAP(name='random_gap',
          system=system)
gap.train(training_data)

# Predict on the training and test data
exit()
gap.predict(training_data)
gap.predict(test_data)
