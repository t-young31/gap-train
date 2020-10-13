from gaptrain.loss import RMSE
from gaptrain.configurations import ConfigurationSet
from gaptrain.systems import System
import numpy as np

system = System(box_size=[10, 10, 10])
system.add_solvent('h2o', n=3)


def test_rmse():

    configs = ConfigurationSet()
    for _ in range(2):
        configs += system.random()

    configs[0].energy = 1
    configs[1].energy = 2

    true_configs = configs.copy()
    true_configs[0].energy = 1.1
    true_configs[1].energy = 1.8

    rmse = RMSE(configs, true_configs)

    expected = np.sqrt(((1-1.1)**2 + (2-1.8)**2)/2.0)
    assert np.abs(rmse.energy - expected) < 1E-6
