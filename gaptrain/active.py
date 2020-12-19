import os
import gaptrain as gt
import numpy as np
import gaptrain.exceptions as ex
from gaptrain.utils import unique_name
from autode.atoms import get_vdw_radius
from multiprocessing import Pool
from gaptrain.log import logger


def calc_error(frame, gap, method_name):
    """Calculate the error between the ground truth and the GAP prediction"""
    getattr(frame, f'run_{method_name.lower()}')(n_cores=1)

    pred = frame.copy()
    pred.run_gap(gap=gap, n_cores=1)

    error = np.abs(pred.energy - frame.energy)
    logger.info(f'|E_GAP - E_0| = {np.round(error, 3)} eV')
    return error


def get_active_config(config, gap, temp, e_thresh, max_time_fs,
                      method_name='dftb', curr_time_fs=0, n_calls=0):
    """
    Given a configuration run MD with a GAP until the absolute error between
    the predicted and true values is above a threshold

    --------------------------------------------------------------------------
    :param config: (gt.Configuration)
    :param gap: (gt.GAP)
    :param e_thresh: (float)
    :param temp: (float) Temperature to propagate GAP-MD
    :param max_time_fs: (float)
    :param method_name: (str)
    :param curr_time_fs: (float)
    :param n_calls: (int) Number of times this function has been called

    :return: (gt.Configuration)
    """
    if float(temp) < 0:
        raise ValueError('Cannot run MD with a negative temperature')

    if float(e_thresh) < 0:
        raise ValueError(f'Error threshold {e_thresh} must be postiive (eV)')

    implemented_ref_methods = ['dftb', 'gpaw', 'orca']
    if method_name not in implemented_ref_methods:
        raise ValueError(f'Unknown reference method ({method_name}) '
                         f'implemented methods are {implemented_ref_methods}')

    if method_name == 'orca' and gt.GTConfig.orca_keywords is None:
        raise ValueError("For ORCA training GTConfig.orca_keywords must be "
                         "set. e.g. "
                         "GradientKeywords(['PBE', 'def2-SVP', 'EnGrad'])")

    md_time_fs = 2 + n_calls**3
    gap_traj = gt.md.run_gapmd(config,
                               gap=gap,
                               temp=float(temp),
                               dt=0.5,
                               interval=4,
                               fs=md_time_fs,
                               n_cores=1)
    curr_time_fs += md_time_fs

    # Evaluate the error on the final frame
    error = calc_error(frame=gap_traj[-1], gap=gap, method_name=method_name)

    if error > 100 * e_thresh:
        logger.error('Huge error: 100x threshold, returning the first frame')
        getattr(gap_traj[0], f'run_{method_name}')(n_cores=1)
        return gap_traj[0]

    if error > 10 * e_thresh:
        logger.warning('Error 10 x threshold! Taking the last frame less than '
                       '10x the threshold')
        # Stride through only 10 frames to prevent very slow backtracking
        for frame in reversed(gap_traj[::max(1, len(gap_traj)//10)]):

            error = calc_error(frame, gap=gap, method_name=method_name)
            if e_thresh < error < 10 * e_thresh:
                return frame

    if error > e_thresh:
        return gap_traj[-1]

    if curr_time_fs > max_time_fs:
        logger.info(f'Reached the maximum time {max_time_fs} fs, returning '
                    f'None')
        return None

    # If the prediction is within the threshold then call this function again
    return get_active_config(config, gap, temp, e_thresh, max_time_fs,
                             curr_time_fs=curr_time_fs,
                             method_name=method_name,
                             n_calls=n_calls+1)


def get_active_configs(config, gap, method_name,
                       max_time_fs=1000, n_configs=10, temp=300, e_thresh=0.1):
    """
    Generate n_configs using on-the-fly active learning parallelised over
    GTConfig.n_cores

    --------------------------------------------------------------------------
    :param config: (gt.Configuration) Initial configuration to propagate from

    :param gap: (gt.gap.GAP) GAP to run MD with

    :param method_name: (str) Name of the method to use as the ground truth

    :param max_time_fs: (float) Maximum propagation time in the active learning
                        loop. Default = 1 ps

    :param n_configs: (int) Number of configurations to generate

    :param temp: (float) Temperature in K to run the intermediate MD with

    :param e_thresh: (float) Energy threshold in eV above which the MD frame
                     is returned by the active learning function i.e
                     E_t < |E_GAP - E_true|

    :return:(gt.ConfigurationSet)
    """
    if int(n_configs) < int(gt.GTConfig.n_cores):
        raise NotImplementedError('Active learning is only implemented using '
                                  'one core for each process. Please use '
                                  'n_configs >= gt.GTConfig.n_cores')
    results = []
    configs = gt.Data()

    logger.info(f'Using {gt.GTConfig.n_cores} processes')
    with Pool(processes=int(gt.GTConfig.n_cores)) as pool:

        for _ in range(n_configs):
            result = pool.apply_async(func=get_active_config,
                                      args=(config, gap, temp, e_thresh,
                                            max_time_fs, method_name))
            results.append(result)

        for result in results:
            try:
                config = result.get(timeout=None)
                if config is not None and config.energy is not None:
                    configs.add(config)
            # Lots of different exceptions can be raised when trying to
            # generate an active config, continue regardless..
            except:
                continue

    return configs


def get_init_configs(system, init_configs=None, n=10, method_name=None):
    """Generate a set of initial configurations to use for active learning"""

    if init_configs is not None:

        if all(cfg.energy is not None for cfg in init_configs):
            logger.info(f'Initialised with {len(init_configs)} configurations '
                        f'all with defined energy')
            return init_configs

    # Initial configurations are not defined, so make some - will use random
    # with the largest maximum distance between molecules possible
    max_vdw = max(get_vdw_radius(symbol) for symbol in system.atom_symbols())
    ideal_dist = 2*max_vdw - 0.5    # Desired minimum distance in Å

    # Reduce the distance until there is a probability at least 0.1 that a
    # random configuration can be generated with that distance threshold
    p_acc, dist = 0, ideal_dist+0.2

    while p_acc < 0.1:
        n_generated_configs = 0
        dist -= 0.2                 # Reduce the minimum distance requirement

        for _ in range(10):
            try:
                _ = system.random(min_dist_threshold=dist)
                n_generated_configs += 1

            except ex.RandomiseFailed:
                continue

        p_acc = n_generated_configs / 10

    init_configs = gt.Data(name='init_configs')
    # Finally generate the initial configurations
    while len(init_configs) < n:
        try:
            init_configs += system.random(min_dist_threshold=dist)
        except ex.RandomiseFailed:
            continue

    if method_name is None:
        logger.warning('Have no method - not evaluating energies')
        return init_configs

    # And run the desired method in parallel across them
    method = getattr(init_configs, f'parallel_{method_name.lower()}')
    method()

    init_configs.save()
    return init_configs


def train(system,
          method_name,
          gap=None,
          max_time_active_fs=1000,
          n_configs_iter=10,
          temp=300,
          active_e_thresh=None,
          max_energy_threshold=None,
          validate=False,
          tau=None,
          val_interval=None,
          max_active_iters=50,
          n_init_configs=10,
          init_configs=None):
    """
    Train a system using active learning, by propagating dynamics using ML
    driven molecular dynamics (MD) and adding configurations where the error
    is above a threshold. Loop looks something like

    Generate configurations -> train a GAP -> run GAP-MD -> frames with error
                                   ^                               |
                                   |________ calc true  ___________


    Active learning will loop until either (1) the iteration > max_active_iters
    or (2) no configurations are found to add or (3) if calculated τ = max(τ)
    where the loop will break out

    --------------------------------------------------------------------------
    :param system: (gt.system.System)

    :param method_name: (str) Name of a method to use as the ground truth e.g.
                        dftb, orca, gpaw

    :param gap: (gt.gap.GAP) GAP to train with the active learnt data, if
                None then one will be initialised by placing SOAPs on each
                heavy atom and defining the 'other' atom types included in the
                neighbour density by their proximity. Distance cutoffs default
                to 3.5 Å for all atoms

    :param max_time_active_fs: (float) Maximum propagation time in the active
                               learning loop. Default = 1 ps

    :param n_configs_iter: (int) Number of configurations to generate per
                           active learning cycle

    :param temp: (float) Temperature in K to propagate active learning at -
                 higher is better for stability but requires more training

    :param active_e_thresh: (float) Threshold in eV (E_t) above which a
                            configuration is added to the potential. If None
                            then will use 1 kcal mol-1 molecule-1

    :param max_energy_threshold: (float) Maximum relative energy threshold for
                                 configurations to be added to the training
                                 data

    :param validate: (bool) Whether or not to validate the potential during
                     the training. Will, by default run a τ calculation with
                     an interval max_time_active_fs / 100, so that a maximum of
                     50 calculations are run and a maximum time of
                     max(τ) = 5 x max_time_active_fs

    :param tau: (gt.loss.Tau) A instance of the τ error metric, unused if no
                validation is performed. Otherwise

    :param val_interval: (int) Interval in the active training loop at which to
                         run the validation. Defaults to max_active_iters // 10
                         if validation is requested

    :param max_active_iters: (int) Maximum number of active learning
                             iterations to perform. Will break if we hit the
                             early stopping criteria

    :param n_init_configs: (int) Number of initial configurations to generate,
                           will be ignored if init_configs is not None

    :param init_configs: (gt.ConfigurationSet) A set of configurations from
                         which to start the active learning from

    :return:
    """
    if len(system.molecules) > 1:
        # TODO training for arbitrary systems
        raise NotImplementedError

    init_configs = get_init_configs(init_configs=init_configs,
                                    n=n_init_configs,
                                    method_name=method_name,
                                    system=system)

    # Initial configuration must have energies
    assert all(cfg.energy is not None for cfg in init_configs)

    if gap is None:
        gap = gt.GAP(name=unique_name('active_gap'), system=system)

    # Initialise a τ metric with default parameters
    if validate and tau is None:
        tau = gt.loss.Tau(configs=get_init_configs(system, n=5))

    # Default to validating 10 times through the training
    if validate and val_interval is None:
        val_interval = max_active_iters // 10

    # Initialise training data
    train_data = gt.Data(name=gap.name)
    train_data += init_configs

    # and train an initial GAP
    gap.train(init_configs)

    if active_e_thresh is None:
        #                 1 kcal mol-1 molecule-1
        active_e_thresh = 0.043363 * len(system.molecules)

    tau_file = open(f'{gap.name}_tau.txt', 'w')
    for iteration in range(max_active_iters):
        configs = get_active_configs(system.random(),
                                     gap=gap,
                                     method_name=method_name,
                                     n_configs=n_configs_iter,
                                     temp=temp,
                                     e_thresh=active_e_thresh,
                                     max_time_fs=max_time_active_fs)
        if len(configs) == 0:
            logger.info('No configs to add. Active learning = DONE')
            break

        train_data += configs

        # If required remove high-lying energy configuration from the data
        if max_energy_threshold is not None:
            train_data.remove_above_e(max_energy_threshold)

        # Retrain on these new data
        gap.train(train_data)

        # Print the accuracy
        if validate and iteration % val_interval == 0:
            tau.calculate(gap=gap, method_name=method_name)
            print(iteration, tau.value, sep='\t', file=tau_file)

            if np.abs(tau.value - tau.max_value) < 1:
                logger.info('Reached the maximum tau. Active learning = Done')
                break

    return train_data, gap


def train_ii(system,
             method_name,
             intra_gap,
             inter_gap=None,
             max_time_active_fs=1000,
             n_configs_iter=10,
             temp=300,
             active_e_thresh=None,
             max_energy_threshold=None,
             validate=False,
             tau=None,
             val_interval=None,
             max_active_iters=50,
             n_init_configs=10,
             init_inter_configs=None):
    """Train an intra+inter molecular GAP

    :param intra_gap: Intramolecular GAP - *must* be trained
    """

    if not all(mol == system.molecules[0] for mol in system.molecules):
        raise ValueError('An intra+intermolecular GAP requires all the same '
                         'molecules in the system e.g. H2O(aq)')

    if not os.path.exists(f'{intra_gap.name}.xml'):
        raise RuntimeError('An intramolecular GAP must be trained prior to an '
                           'intra+intermolecular GAP')





