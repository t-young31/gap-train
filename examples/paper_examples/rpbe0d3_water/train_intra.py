import gaptrain as gt
import numpy as np
from autode.atoms import Atom
gt.GTConfig.n_cores = 36


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
    minimum = gt.Configuration('h2o_min_revPBE0.xyz', box=gt.Box([8, 8, 8]))
    assert minimum.atoms is not None
    configs += minimum

    for r1 in np.linspace(0.8, 1.5, n_to_cube):
        for r2 in np.linspace(0.8, 1.5, n_to_cube):
            for r3 in np.linspace(1.0, 2.5, n_to_cube):
                h2o = get_h2o(r1, r2, r3)
                configs += h2o

    configs.parallel_cp2k()
    return configs


if __name__ == '__main__':

    water_monomer = gt.System(box_size=[8, 8, 8])
    water_monomer.add_solvent('h2o', n=1)

    gap = gt.GAP(name=f'monomer_2b_3b',
                 system=water_monomer,
                 default_params=None)

    # Should only have O-H
    gap.params.pairwise[('O', 'H')] = gt.GTConfig.gap_default_2b_params.copy()
    gap.params.pairwise[('O', 'H')]['cutoff'] = 3.0

    gap.params.angle[('H', 'O', 'H')] = gt.GTConfig.gap_default_2b_params.copy()
    gap.params.angle[('H', 'O', 'H')]['cutoff'] = 3.0

    train_data = grid_configs(n_to_cube=7)
    gap.train(train_data)
