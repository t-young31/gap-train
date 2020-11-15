import numpy as np
from ase.calculators.calculator import Calculator
from ase import Atoms


class IICalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def expanded_atoms(self, atoms):
        """Generate atoms expanded by a factor intermolecularly"""
        ex_atoms = atoms.copy()

        # Expand the box
        ex_atoms.set_cell(self.expansion_factor * atoms.cell)

        # Get the current coordinates and indexes of the atoms to shift',
        coords = ex_atoms.get_positions()
        mol_idxs = np.array(self.intra.mol_idxs, dtype=int)

        for atom_idxs in mol_idxs:
            vec = np.average(coords[atom_idxs], axis=0)
            frac_com = vec / np.diagonal(atoms.cell)

            # Shift from the current position to the new approximate
            # fractional center of mass
            coords[atom_idxs] += (frac_com * np.diagonal(ex_atoms.cell) - vec)

        ex_atoms.set_positions(coords)
        ex_atoms.wrap()

        return ex_atoms

    def calculate(self, atoms=None, properties=None,
                  system_changes=None,
                  **kwargs):
        """New calculate function used to get energies and forces"""
        # Calculator.calculate(self, atoms, properties, system_changes)

        intra_atoms = self.expanded_atoms(atoms)
        intra_atoms.set_calculator(self.intra)

        inter_atoms = atoms.copy()
        inter_atoms.set_calculator(self.inter)

        # Add the energies and forces
        self.results["energy"] = (inter_atoms.get_potential_energy() +
                                  intra_atoms.get_potential_energy())
        self.results["free_energy"] = self.results["energy"]

        self.results["forces"] = (inter_atoms.get_forces() +
                                  intra_atoms.get_forces())
        return None

    def __init__(self, intra, inter, expansion_factor=10, **kwargs):
        """
        Combination of two ASE calculators used to evaluate intramolecular and
        intermolecular contributions separately. The intramolecular term is
        evaluated by expanding all the intermolecular distances uniformly by
        :param intra: (ase.Calculator)
        :param inter: (ase.Calculator)
        """
        Calculator.__init__(self, restart=None, ignore_bad_restart_file=False,
                            label=None, atoms=None, **kwargs)

        self.intra = intra
        assert hasattr(intra, "mol_idxs")

        self.inter = inter

        self.atoms = None
        self.name = "inter_intra"
        self.parameters = {}

        self.expansion_factor = expansion_factor


class SSCalculator(IICalculator):

    def calculate(self, atoms=None, properties=None,
                  system_changes=None,
                  **kwargs):

        solute_atoms = Atoms(numbers=[atoms.numbers[i] for i in self.solute_idxs],
                             positions=atoms.positions[self.solute_idxs],
                             pbc=True,
                             cell=atoms.cell)

        solvent_atoms = Atoms(numbers=[atoms.numbers[i] for i in self.solvent_idxs],
                              positions=atoms.positions[self.solvent_idxs],
                              pbc=True,
                              cell=atoms.cell)

        solute_atoms.set_calculator(self.solute_intra)

        solv_intra_atoms = self.expanded_atoms(solvent_atoms)
        solv_intra_atoms.set_calculator(self.intra)

        inter_atoms = atoms.copy()
        inter_atoms.set_calculator(self.inter)

        # Add the energies and forces
        self.results["energy"] = (inter_atoms.get_potential_energy() +
                                  solv_intra_atoms.get_potential_energy() +
                                  solute_atoms.get_potential_energy())
        self.results["free_energy"] = self.results["energy"]

        forces = inter_atoms.get_forces()
        forces[self.solute_idxs] += solute_atoms.get_forces()
        forces[self.solvent_idxs] += solv_intra_atoms.get_forces()

        self.results["forces"] = forces
        return None

    def __init__(self, solute_intra, solvent_intra, inter,
                 expansion_factor=10, **kwargs):
        IICalculator.__init__(self,
                              intra=solvent_intra,
                              inter=inter,
                              expansion_factor=expansion_factor,
                              **kwargs)
        """
        Combination of three ASE calculators to calculate the total energy and
        forces for a solute solvent system where the solute is described 
        with one GAP ASE calculator as are the intramolecular modes of the 
        solvent, the remaining interactions are calcd. with a separate GAP
        """

        self.solute_intra = solute_intra
        assert hasattr(solute_intra, "mol_idxs")

        self.solute_idxs = np.array(self.solute_intra.mol_idxs[0], dtype=int)

        self.solvent_idxs = []
        for idxs in solvent_intra.mol_idxs:
            self.solvent_idxs += list(idxs)
        self.solvent_idxs = np.array(self.solvent_idxs, dtype=int)
