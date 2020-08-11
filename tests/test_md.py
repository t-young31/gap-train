from gaptrain.systems import MMSystem
from gaptrain.md import run_mmmd
from gaptrain.configurations import Configuration


def test_run_mmmd():

    water_box = MMSystem(box_size=[10, 10, 10])
    water_box.add_solvent('h2o', n=100)
    MMSystem.generate_topology(water_box)

    for molecule in water_box.molecules:
        molecule.set_mm_atom_types()

    config = Configuration(water_box)
    config.print_gro_file(system=water_box)

    run_mmmd()
