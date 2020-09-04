from gaptrain.systems import MMSystem, System
from gaptrain.md import run_mmmd


def test_run_mmmd():

    water_box = MMSystem(box_size=[20, 20, 20])
    water_box.add_solvent('h2o', n=50)

    MMSystem.generate_topology(water_box)

    for molecule in water_box.molecules:
        molecule.set_mm_atom_types()
    config = System.random(water_box)
    config.wrap()
    config.print_gro_file(system=water_box)

    run_mmmd()
