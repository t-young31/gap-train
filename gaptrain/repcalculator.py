import numpy as np
from ase.calculators.calculator import Calculator
from scipy.spatial import distance_matrix


def repulsion(atoms, ref_dist_mat):
    """
    Calculate the repulsive energy with a purely repulsive potential of the form

    V_rep(r) = SUM_i>j exp(- r/b + a)

    Tested with systems of 100 atoms and the average evaluation time is <1ms,
    so reasonably fast

    :param atoms: (ase.Atoms)
    :param ref_dist_mat: (np.ndarray) shape = (N_atoms, N_atoms) with e.g.
                         sum of VdW radii for non-bonded atoms and bond lengths
                         for bonded nuclei
    """
    coords = atoms.get_positions()
    triu_idxs = np.triu_indices(n=len(atoms), k=1)
    dist_mat = distance_matrix(coords, coords)

    # see https://pubs.acs.org/doi/10.1021/acs.jcim.0c00519 for params
    b_mat = 0.083214 * ref_dist_mat - 0.003768
    a_mat = 11.576415 * (0.175541 * ref_dist_mat + 0.316642)
    exponents = -(dist_mat / b_mat) + a_mat

    # Initially parameterised in kcal mol-1, so convert to eV
    energy_mat = 0.043 * np.exp(exponents)

    # Only sum the unique atom pairs, which is the upper triangular
    # portion of  the matrix
    energy = np.sum(energy_mat[triu_idxs])

    derivative = np.zeros_like(coords)
    diff = np.ufunc.outer(np.subtract, coords.flatten(), coords.flatten())
    x_diff = diff[::3, ::3]
    y_diff = diff[1::3, 1::3]
    z_diff = diff[2::3, 2::3]

    coeff_mat = np.ones_like(dist_mat) / b_mat
    # Zero the diagonal where i = j, and thus should have no contribution
    np.fill_diagonal(coeff_mat, val=0)

    # Fill the diagonal of the distance matrix with ones to prevent
    # dividing by zero
    np.fill_diagonal(dist_mat, val=1)

    # x, y, z components respectively
    derivative[:, 0] = -np.sum((energy_mat * coeff_mat * (x_diff / dist_mat)),
                               axis=1)
    derivative[:, 1] = -np.sum((energy_mat * coeff_mat * (y_diff / dist_mat)),
                               axis=1)
    derivative[:, 2] = -np.sum((energy_mat * coeff_mat * (z_diff / dist_mat)),
                               axis=1)

    return energy, -derivative


class DRepCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=None,
                  system_changes=None,
                  **kwargs):
        """New calculate function used to get energies and forces"""

        rep_energy, rep_force = repulsion(atoms, self.ref_dist_mat)
        atoms.set_calculator(self.calc)

        # Add the energies and forces
        self.results["energy"] = (atoms.get_potential_energy() +
                                  rep_energy)
        self.results["free_energy"] = self.results["energy"]

        self.results["forces"] = (atoms.get_forces() +
                                  rep_force)
        return None

    def __init__(self, calc, ref_dist_mat, **kwargs):
        """
        Combination of a GAP calculator (quippy.Potential) and a classical
        repulsive FF
        """
        Calculator.__init__(self, restart=None, ignore_bad_restart_file=False,
                            label=None, atoms=None, **kwargs)

        self.ref_dist_mat = ref_dist_mat
        self.atoms = None
        self.name = "inter_intra"
        self.parameters = {}

        self.calc = calc
