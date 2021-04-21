import gaptrain as gt
import numpy as np
from autode.atoms import Atom
from copy import deepcopy
gt.GTConfig.n_cores = 8


def get_h2o(r1, r2, r3):
    """
            O
      r1  /   \  r2
         H     H
           r3
    """
    theta = np.arccos(max((r2 ** 2 + r1 ** 2 - r3 ** 2) / (2 * r2 * r1), -1))
    h_b = Atom('H', x=r2, y=0.0, z=0.0)
    # Rotate in the z axis
    h_b.rotate(axis=np.array([0.0, 0.0, 1.0]), theta=theta)

    config = gt.Configuration(charge=0, mult=1, box=gt.Box([10, 10, 10]))
    config.set_atoms(atoms=[Atom('O'), Atom('H', x=r1), h_b])
    return config


def grid_configs(n_to_cube):
    """
    :param n_to_cube: (int) Generate n^3 configurations
    :return: (gt.Configuration)
    """

    configs = gt.Data(name=f'grid_{n_to_cube}-cubed')

    # Also add the minimum energy strucutre
    minimum = get_h2o(r1=1.0, r2=1.0, r3=1.5)
    minimum.run_gpaw(max_force=0.01, n_cores=4)
    configs += minimum

    for r1 in np.linspace(0.8, 1.5, n_to_cube):
        for r2 in np.linspace(0.8, 1.5, n_to_cube):
            for r3 in np.linspace(1.0, 2.5, n_to_cube):
                h2o = get_h2o(r1, r2, r3)
                configs += h2o

    return configs


if __name__ == '__main__':
    water_monomer = gt.System(box_size=[10, 10, 10])
    water_monomer.add_solvent('h2o', n=1)

    # Load the grid configurations and evaluate at PBE/400eV
    grid_configs = grid_configs(n_to_cube=8)
    grid_configs.parallel_gpaw()

    gap = gt.GAP(name=f'water_intra_gap', system=water_monomer,
                 default_params=False)

    gap.params.pairwise[('O', 'H')] = deepcopy(gt.GTConfig.gap_default_2b_params)
    gap.params.pairwise[('O', 'H')]['cutoff'] = 3.0

    gap.params.pairwise[('H', 'H')] = deepcopy(gt.GTConfig.gap_default_2b_params)
    gap.params.pairwise[('H', 'H')]['cutoff'] = 3.0

    gap.params.angle[('H', 'O', 'H')] = deepcopy(gt.GTConfig.gap_default_2b_params)
    gap.params.angle[('H', 'O', 'H')]['cutoff'] = 3.0

    gap.train(grid_configs)
