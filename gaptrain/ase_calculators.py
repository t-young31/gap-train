import numpy as np
from ase.calculators.calculator import Calculator
from ase import Atoms as ASEAtoms


def expanded_atoms(atoms:            ASEAtoms,
                   expansion_factor: float,
                   mol_idxs:         np.ndarray):
    """
    Generate atoms expanded by a factor intermolecularly e.g.

                                 ________________
      ___________               /               /
     /     X   /       -->     /         X     /
    /  X     /                /              /
    ---------                /     X        /
                            ----------------

    will only apply shifts to the centre of mass for molecules defined in
    mol_idxs.

    :param atoms:  ASE atoms
    :param expansion_factor: (float)
    :param mol_idxs: (np.ndarray) Matrix of atom indexes for each molecule

    :returns: ASE atoms
    """
    ex_atoms = ASEAtoms(numbers=atoms.numbers)

    # Expand the box
    ex_atoms.set_cell(expansion_factor * atoms.cell)

    # Get the current coordinates and indexes of the atoms to shift',
    coords = atoms.get_positions()

    vec = np.average(coords[mol_idxs], axis=1)
    vecs = np.zeros_like(coords)
    vecs[mol_idxs.flatten(), :] = np.repeat(vec,
                                            repeats=mol_idxs.shape[1],
                                            axis=0)
    frac_com = vecs / np.diagonal(atoms.cell)

    # Shift from the current position to the new approximate
    # fractional center of mass
    coords += (frac_com * np.diagonal(ex_atoms.cell) - vecs)

    ex_atoms.set_positions(coords)
    ex_atoms.wrap()

    return ex_atoms


def _quippy_calc(xml_filename: str):
    """

    :param xml_filename: (str)
    :return:
    """

    if not xml_filename.endswith('.xml'):
        raise ValueError(f'xml filename must end with .xml.'
                         f' Had: {xml_filename}')

    try:
        import quippy
        return quippy.potential.Potential("IP GAP",
                                          param_filename=xml_filename)

    except ModuleNotFoundError:
        raise ModuleNotFoundError('Quippy was not installed. Try\n'
                                  'pip install quippy-ase')


class IntraCalculator(Calculator):

    implemented_properties = ["energy", "forces", "free_energy"]

    def expanded_subset(self, atoms):
        """Atoms expanded by their intramolecular COM then the QUIP
        calculator set"""

        _atoms = expanded_atoms(atoms,
                                expansion_factor=self.expansion_factor,
                                mol_idxs=self.mol_idxs)

        atoms_subset = ASEAtoms(numbers=_atoms.numbers[self.flat_mol_idxs],
                                positions=_atoms.positions[self.flat_mol_idxs, :],
                                pbc=True,
                                cell=_atoms.cell)

        atoms_subset.set_calculator(self.quip_calculator)
        return atoms_subset

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=None,
                  **kwargs):
        """Calculate energies and forces"""

        _atoms = self.expanded_subset(atoms)

        # Add the total energy of all the intra components
        self.results["energy"] = _atoms.get_potential_energy()

        # And a subset of the forces
        forces = np.zeros(shape=(len(atoms), 3))
        forces[self.flat_mol_idxs, :] += _atoms.get_forces()
        self.results["forces"] = forces
        return None

    def __init__(self,
                 name,
                 xml_filename,
                 mol_idxs,
                 expansion_factor=10,
                 **kwargs):
        """
        ASE calculator for the calculation of intramolecular energies and
        forces for all molecules in a system, defined by a set of mol_idxs

        :param name:
        :param xml_filename:
        :param mol_idxs:
        :param expansion_factor:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.name = name
        self.expansion_factor = expansion_factor
        self.quip_calculator = _quippy_calc(xml_filename)

        # Will throw a ValueError if the list of lists is not 'homogeneous'
        # and not correctly assigned to the same molecule
        self.mol_idxs = np.array(mol_idxs,  dtype=int)
        self.flat_mol_idxs = self.mol_idxs.flatten()


class IICalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def calculate(self,
                  atoms=None,
                  properties=None,
                  system_changes=None,
                  **kwargs):
        """
        Calculate the total energy and forces using the inter +
        """
        inter_atoms = ASEAtoms(numbers=atoms.numbers,
                               positions=atoms.positions,
                               pbc=True,
                               cell=atoms.cell)
        inter_atoms.set_calculator(self.inter_quip_calculator)

        self.results["energy"] = inter_atoms.get_potential_energy()
        forces = inter_atoms.get_forces()

        # Add intramolecular components in turn
        for intra_calc in self.intra_calculators:
            intra_atoms = intra_calc.expanded_subset(atoms)

            self.results["energy"] += intra_atoms.get_potential_energy()
            forces[intra_calc.flat_mol_idxs, :] += intra_atoms.get_forces()

        self.results["forces"] = forces
        return None

    def __init__(self,
                 name:              str,
                 xml_filename:      str,
                 intra_calculators: list,
                 **kwargs):
        """
        Inter+intra molecular ASE calculator using several intra calculators
        that apply to specific atoms indexes while the intermolecular
        component applies to everything in the defined box size

        :param name:
        :param xml_filename:
        :param intra_calculators:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.name = name
        self.intra_calculators = intra_calculators
        self.inter_quip_calculator = _quippy_calc(xml_filename)
