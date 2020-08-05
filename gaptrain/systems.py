from gaptrain.molecules import Molecule
from gaptrain.log import logger


class System:

    def __len__(self):
        return len(self.molecules)

    def __add__(self, other):
        """Add another list or molecule to the system"""

        if type(other) is list or type(other) is tuple:
            assert all(isinstance(m, Molecule) for m in other)
            self.molecules += list(other)

        elif isinstance(other, Molecule):
            self.molecules.append(other)

        elif isinstance(other, System):
            assert other.charge == self.charge
            assert other.mult == self.mult

            self.molecules += other.molecules

        else:
            raise Exception(f'Cannot add {other} to system')

        return self

    def add_water(self):
        """Add water to the system to generate a ~1 g cm-3 density"""
        raise NotImplementedError

    def add_molecules(self, molecule, n=1):
        """Add a number of the same molecule to the system"""
        self.molecules += [molecule for _ in range(n)]
        return None

    def density(self):
        """Calculate the density of the system"""
        raise NotImplementedError

    def __init__(self, *args, box_size, charge, spin_multiplicity=1):
        """
        System containing a set of molecules.

        e.g. pd_1water = (Pd, water, box_size=[10, 10, 10], charge=2)
        for a system containing a Pd(II) ion and one water in a 10 Å^3 box

        ----------------------------------------------------------------------
        :param args: (gaptrain.molecules.Molecule) Molecules that comprise
                     the system

        :param box_size: (list(float)) Dimensions of the box that the molecules
                        occupy. e.g. [10, 10, 10] for a 10 Å cubic box.

        :param charge: (int) Total Charge on the system e.g. 0 for a water box
                       or 2 for a Pd(II)(aq) system

        :param spin_multiplicity: (int) Spin multiplicity on the whole system
                                  2S + 1 where S is the number of unpaired
                                  electrons
        """
        self.molecules = list(args)
        print(self.molecules)

        assert len(box_size) == 3
        self.box_size = [float(k) for k in box_size]

        self.charge = int(charge)
        self.mult = int(spin_multiplicity)

        logger.info(f'Initalised a system\n'
                    f'Number of molecules = {len(self.molecules)}\n'
                    f'Charge              = {self.charge} e\n'
                    f'Spin multiplicity   = {self.mult}')


class MMSystem(System):

    def generate_topology(self):
        """Generate a GROMACS topology for this system"""
        assert all(m.itp_filename is not None for m in self.molecules)

        raise NotImplementedError

    def __init__(self, *args, box_size, charge, spin_multiplicity):
        """System that can be simulated with molecular mechanics"""

        super().__init__(*args,
                         box_size=box_size,
                         charge=charge,
                         spin_multiplicity=spin_multiplicity)
