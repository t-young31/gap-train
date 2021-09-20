import os
import gaptrain as gt
import autode as ade
import numpy as np
from gaptrain.ase_calculators import expanded_atoms
from ase.io import write


def test_box_expansion():

    h2o = gt.System(box_size=[7, 7, 7])
    h2o.add_solvent('h2o', n=5)
    ase_atoms = h2o.random().ase_atoms()

    mol_idxs = np.array([[3*j + i for i in range(3)] for j in range(5)])
    new_atoms = expanded_atoms(atoms=ase_atoms,
                               mol_idxs=mol_idxs,
                               expansion_factor=10)
    print(new_atoms.positions)

    from timeit import repeat
    print('Expansion runs in ',
          min(repeat(lambda: expanded_atoms(atoms=ase_atoms,
                                            mol_idxs=mol_idxs,
                                            expansion_factor=10),
                     number=100))/100*1000,
          'ms')

    write('tmp.xyz', new_atoms)

    ade_mol = ade.Molecule('tmp.xyz')
    assert ade_mol.distance(0, 1) < 2           # Å
    assert ade_mol.distance(0, 3) > 5           # Å
    os.remove('tmp.xyz')
