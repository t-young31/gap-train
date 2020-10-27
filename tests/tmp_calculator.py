from ase.calculators.calculator import Calculator
from ase.io import write
import numpy as np




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
