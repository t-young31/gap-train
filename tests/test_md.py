from gaptrain.systems import MMSystem
from gaptrain.md import run_mmmd
import os


def test_run_mmmd():

    water_box = MMSystem(box_size=[30, 30, 30])
    water_box.add_solvent('h2o', n=100)

    config = water_box.random()

    if 'GT_GMX' not in os.environ or not os.environ['GT_GMX'] == 'True':
        return

    run_mmmd(water_box, config, temp=300, dt=1, interval=1000, fs=100)
