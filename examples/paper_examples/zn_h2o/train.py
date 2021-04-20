import os
import gaptrain as gt
gt.GTConfig.n_cores = 20


def train_intra_zn():
    gap = gt.GAP(name='intra_znh2o6', system=znh2o6, default_params=False)
    gap.params.soap['O'] = gt.GTConfig.gap_default_soap_params
    gap.params.soap['O']['cutoff'] = 3.0
    gap.params.soap['O']['other'] = ['Zn', 'H', 'O']

    _, _ = gt.active.train(znh2o6,
                           gap=gap,
                           method_name='gpaw',
                           validate=True,
                           temp=1000,
                           tau_max=1000,
                           active_e_thresh=0.1,
                           n_configs_iter=20)
    return None


def train_inter():
    intra_h2o_gap = gt.gap.SolventIntraGAP(name='water_intra_gap',
                                           system=system)

    intra_methane_gap = gt.gap.SoluteIntraGAP(name='intra_znh2o6',
                                              system=system,
                                              molecule=gt.Molecule(
                                                  'znh2o6.xyz'))

    inter_gap = gt.InterGAP(name='inter', system=system, default_params=False)
    inter_gap.params.soap['O'] = gt.GTConfig.gap_default_soap_params
    inter_gap.params.soap['O']['cutoff'] = 4.0
    inter_gap.params.soap['O']['other'] = ['Zn', 'H', 'O']

    gt.active.train(system,
                    method_name='gpaw',
                    validate=False,
                    max_time_active_fs=5000,
                    active_e_thresh=0.1+0.04*20,
                    gap=gt.gap.SSGAP(solute_intra=intra_methane_gap,
                                     solvent_intra=intra_h2o_gap,
                                     inter=inter_gap),
                    max_energy_threshold=5,
                    n_configs_iter=20)
    return None


if __name__ == '__main__':

    if not os.path.exists('water_intra_gap.xml'):
        exit('Intramolecular GAP for water did not exist. Please generate it '
             'with e.g. train_water_h2o.py')

    znh2o6 = gt.System(box_size=[10, 10, 10])
    znh2o6.add_molecules(gt.Molecule('znh2o6.xyz', charge=2))
    train_intra_zn()

    # Generate a [Zn(H2O)6](aq) system
    system = gt.System(box_size=[10, 10, 10])
    system.add_molecules(gt.Molecule('znh2o6.xyz', charge=2))
    system.add_solvent('h2o', n=20)
    train_inter()
