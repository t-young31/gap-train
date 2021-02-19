import gaptrain as gt
gt.GTConfig.n_cores = 10


def train_h2o():
    """Train the intramolecular PES of H2O"""
    _, _ = gt.active.train(h2o,
                           gap=gt.GAP(name='intra_h2o', system=h2o),
                           method_name='gpaw',
                           validate=False,
                           temp=1000)
    return None


def train_sn2():
    """Train the intramolecular PES of a Cl+CH3Cl reaction"""
    _, _ = gt.active.train(sn2,
                           gap=gt.GAP(name='intra_sn2', system=sn2),
                           active_e_thresh=0.1,
                           max_time_active_fs=500,
                           fix_init_config=True,
                           method_name='gpaw',
                           validate=False,
                           temp=500)
    return None


def train_inter():
    """Train the intermolecular components between H2O and the reaction"""
    _, _ = gt.active.train(system,
                           method_name='gpaw',
                           validate=False,
                           fix_init_config=True,
                           active_e_thresh=0.5,
                           max_time_active_fs=500,
                           gap=gt.gap.SSGAP(solute_intra=intra_sn2_gap,
                                            solvent_intra=intra_h2o_gap,
                                            inter=inter_gap))
    return None


if __name__ == '__main__':

    h2o = gt.System(box_size=[10, 10, 10])
    h2o.add_solvent('h2o', n=1)
    train_h2o()

    sn2 = gt.System(box_size=[10, 10, 10])
    sn2.add_molecules(gt.Molecule('ts.xyz', charge=-1))
    train_sn2()

    system = gt.System(box_size=[10, 10, 10])
    system.add_molecules(gt.Molecule('ts.xyz', charge=-1))
    system.add_solvent('h2o', n=20)

    # Generate 'intra' GAPs for the solvent (water) and solute (SN2 reaction)
    intra_h2o_gap = gt.gap.SolventIntraGAP(name='intra_h2o', system=system)
    intra_sn2_gap = gt.gap.SoluteIntraGAP(name='intra_sn2',
                                          system=system,
                                          molecule=gt.Molecule('ts.xyz', charge=-1))

    # Use a custom GAP to allow for the longer interaction range of Cl-
    inter_gap = gt.InterGAP(name='inter', system=system, default_params=False)
    inter_gap.params.soap['O'] = gt.GTConfig.gap_default_soap_params
    inter_gap.params.soap['O']['other'] = ['O', 'H', 'Cl', 'C']

    inter_gap.params.soap['Cl'] = gt.GTConfig.gap_default_soap_params
    inter_gap.params.soap['Cl']['cutoff'] = 4.5
    inter_gap.params.soap['Cl']['other'] = ['O', 'H', 'Cl', 'C']

    train_inter()

    # run some dynamics
    traj = gt.md.run_gapmd(configuration=system.random(min_dist_threshold=1.8),
                           gap=gt.gap.SSGAP(solute_intra=intra_sn2_gap,
                                            solvent_intra=intra_h2o_gap,
                                            inter=inter_gap),
                           temp=300,
                           dt=0.5,
                           interval=5,
                           fs=500,
                           n_cores=4)

    traj.save(filename='water_sn2_traj.xyz')
