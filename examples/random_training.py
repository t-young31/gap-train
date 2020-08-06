from gaptrain import *

# Create a system of Pd(II) in a 10 Å^3 box with 40 water molecules
system = System(Ion('Pd', charge=2),
                box_size=[10, 10, 10],
                charge=2)

system.add_molecules(Molecule('h2o.xyz'), n=40)

# Initialise the training and test data
training_data = Data()
test_data = Data()

# Generate 10 random configurations
configs = ConfigurationSet()
for _ in range(10):
    system.randomise()
    configs += system.configuration()

# DFT in parallel
configs.run_gpaw()

# Add to the training data
training_data += configs

# Initialise and train a GAP
gap = GAP()
gap.train(training_data)

# Predict on the training and test data
gap.predict(training_data)
gap.predict(test_data)

# Plot the results
gap.plot_energy_correlation()
gap.plot_force_correlation()
gap.plot_energy_overlap()
