import os
import gaptrain as gt
import autode as ade
from gaptrain.iicalculator import IICalculator
from ase.io import write


def test_box_expansion():

    h2o = gt.System(box_size=[7, 7, 7])
    h2o.add_solvent('h2o', n=5)
    ase_atoms = h2o.random().ase_atoms()

    intra = gt.GAP('intra')
    intra.mol_idxs = [[3*j + i for i in range(3)] for j in range(5)]
    calc = IICalculator(intra=intra,
                        inter=gt.GAP('iter'))

    new_atoms = calc.expanded_atoms(atoms=ase_atoms)

    from timeit import repeat
    print('Expansion runs in ',
          min(repeat(lambda: calc.expanded_atoms(atoms=ase_atoms),
                     number=100))/100,
          's')

    write('tmp.xyz', new_atoms)

    ade_mol = ade.Molecule('tmp.xyz')
    assert ade_mol.get_distance(0, 1) < 2           # Å
    assert ade_mol.get_distance(0, 3) > 5           # Å
    os.remove('tmp.xyz')
