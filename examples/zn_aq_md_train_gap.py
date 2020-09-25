import gaptrain as gt
from gaptrain.md import run_mmmd, gro2xyz
gt.GTConfig.n_cores = 24


def predict(gap, training_data):
    """Predict energies and forces on the training and validation data"""

    predictions = gap.predict(validation)
    val = gt.RMSE(training_data, predictions)

    # and the training data
    predictions = gap.predict(training_data)
    train_err = gt.RMSE(training_data, predictions)

    print(f'{train_err.energy},{train_err.force}, {val.energy}, {val.force}',
          file=out_file)

    return None


def train(init_gap, curr_training_data, init_config,
          n_configs=50,
          iteration=0,
          max_iters=10000):
    """Train a GAP by adding training data using MD from an initial config"""

    gap = gt.GAP(name=f'Zn_md_train_{n_trained}_{iteration}',
                 system=zn_h2o)

    all_configs = gt.ConfigurationSet

    # Times in femtoseconds for the MD simulation
    for time in [5, 10, 20, 40, 80, 160, 320, 640, 1280]:

        try:
            gap_md_traj = gt.md.run_gapmd(init_config, gap=init_gap,
                                          temp=300,
                                          interval=5,
                                          dt=0.5,
                                          fs=time)
            # Calculate energy and forces on these GAP MD snapshots
            gap_md_traj.parallel_dftb()

        except:
            print("Something broke :(")
            break

    # Truncate back down to n_configs configurations with CUR
    all_configs.truncate(n=n_configs, method='cur')
    curr_training_data += all_configs

    gap.train(curr_training_data)
    predict(gap, training_data=curr_training_data)

    return gap


if __name__ == "__main__":

    # Aqueous Zn(II) sysstem
    zn_h2o = gt.System(gt.Ion('Zn', charge=2), box_size=[12, 12, 12])
    zn_h2o.add_solvent('h2o', n=52)

    # Load the validation data
    validation = gt.Data(name='Zn_DFTBMD_data')
    validation.load(system=zn_h2o)

    ref_energy = -5877.02169783  # Where did this come from?
    f_thresh = 10

    # Pretrained GAP on 100 configurations from MD # is this correct?
    config = zn_h2o.random()
    run_mmmd(zn_h2o, config, temp=300, dt=1, interval=100, ps=10)
    gro2xyz('nvt_traj.gro', zn_h2o)

    init_gap = gt.GAP(name=f'nvt_traj.xyz', system=zn_h2o)

    # Several iterations the accuracy depends on the random configurations
    # added during the training
    n_trained = 0

    while n_trained < 5:

        out_file = open(f'out_{n_trained}.txt', 'w')
        print('RMSE(E_train), RMSE(|F|_train), RMSE(E_val), RMSE(F_val)',
              file=out_file)

        training_data = gt.Data(name=f'Zn_random_train_{n_trained}')

        n_trained += 1

        out_file.close()
