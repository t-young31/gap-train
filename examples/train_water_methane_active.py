import gaptrain as gt
gt.GTConfig.n_cores = 4


def train_h2o():
    h2o_gap = gt.GAP(name='intra_h2o', system=h2o)
    _, _ = gt.active.train(h2o,
                           gap=h2o_gap,
                           method_name='dftb',
                           validate=False,
                           temp=1000)
    return None


def train_methane():
    methane_gap = gt.GAP(name='intra_methane', system=methane)
    _, _ = gt.active.train(methane,
                           gap=methane_gap,
                           method_name='dftb',
                           validate=False,
                           temp=1000)
    return None


def train_inter():
    init_configs = gt.Data()
    init_configs.load('init_configs.xyz')

    data, gap = gt.active.train(system,
                                init_configs=init_configs,
                                method_name='dftb',

                                validate=False,
                                gap=gt.gap.SSGAP(
                                    solute_intra=intra_methane_gap,
                                    solvent_intra=intra_h2o_gap,
                                    inter=inter_gap))
    return None


if __name__ == '__main__':

    h2o = gt.System(box_size=[10, 10, 10])
    h2o.add_solvent('h2o', n=1)
    # train_h2o()

    methane = gt.System(box_size=[10, 10, 10])
    methane.add_molecules(gt.Molecule('methane.xyz'))
    # train_methane()

    system = gt.System(box_size=[10, 10, 10])
    system.add_molecules(gt.Molecule('methane.xyz'))
    system.add_solvent('h2o', n=10)

    intra_h2o_gap = gt.gap.SolventIntraGAP(name='intra_h2o', system=system)
    intra_methane_gap = gt.gap.SoluteIntraGAP(name='intra_methane',
                                              system=system,
                                              molecule=gt.Molecule('methane.xyz'))
    inter_gap = gt.InterGAP(name='inter', system=system)
    # train_inter()

    traj = gt.md.run_gapmd(configuration=system.random(min_dist_threshold=2.0),
                           gap=gt.gap.SSGAP(solute_intra=intra_methane_gap,
                                            solvent_intra=intra_h2o_gap,
                                            inter=inter_gap),
                           temp=300,
                           dt=0.5,
                           interval=5,
                           fs=500,
                           n_cores=4)

    traj[0].save(filename='init.xyz')
    traj.save(filename='tmp.xyz')
