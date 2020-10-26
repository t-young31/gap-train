from ase.calculators.calculator import Calculator
from ase.io import write
import numpy as np


class IICalculator(Calculator):

    implemented_properties = ['energy', 'forces']

    def expanded_atoms(self, atoms):
        """Generate atoms expanded by a factor intermolecularly"""
        ex_atoms = atoms.copy()

        # Expand the box
        ex_atoms.set_cell(self.expansion_factor * atoms.cell)

        # Get the current coordinates and indexes of the atoms to shift
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
        """
        New calculate function used to get energies and forces

        :param atoms: (ase.Atoms)
        """
        properties = ['energy', 'forces']

        self.inter.calculate(atoms, properties, system_changes, **kwargs)

        self.intra.calculate(self.expanded_atoms(atoms),
                             properties,
                             system_changes,
                             **kwargs)

        # Add the energies and forces
        self.results['energy'] = (self.inter.results['energy']
                                  + self.intra.results['energy'])
        self.results['free_energy'] = self.results['energy']

        self.results['forces'] = (self.inter.results['forces']
                                  + self.intra.results['forces'])

        return None

    def __init__(self, intra, inter, expansion_factor=10):
        """
        Combination of two ASE calculators used to evaluate intramolecular and
        intermolecular contributions separately. The intramolecular term is
        evaluated by expanding all the intermolecular distances uniformly by

        :param intra: (ase.Calculator)
        :param inter: (ase.Calculator)
        """
        super().__init__()

        self.intra = intra
        assert hasattr(intra, 'mol_idxs')
        self.inter = inter

        self.expansion_factor = expansion_factor


if __name__ == '__main__':

    import gaptrain as gt
    from gaptrain.calculators import DFTB

    l = 5
    tw = gt.System(box_size=[l, l, l])
    tw.add_solvent('h2o', n=3)
    config = tw.random()

    ase_atoms = config.ase_atoms()
    inter_dftb = DFTB(atoms=ase_atoms,
                      kpts=(1, 1, 1),
                      Hamiltonian_Charge=config.charge)
    intra_dftb = DFTB(atoms=ase_atoms,
                      kpts=(1, 1, 1),
                      Hamiltonian_Charge=config.charge)
    intra_gap = gt.IntraGAP(system=tw, name='tmp')
    intra_dftb.mol_idxs = intra_gap.mol_idxs

    iicalc = IICalculator(intra_dftb, inter_dftb)
    ase_atoms.set_calculator(iicalc)
    from ase.optimize import BFGS

    minimisation = BFGS(ase_atoms)
    minimisation.run(fmax=1)
    write('tmp.xyz', ase_atoms)
