import gaptrain as gt
import numpy as np
import gaptrain.exceptions as ex
from subprocess import Popen, PIPE
from autode.atoms import Atom
from gaptrain.utils import unique_name, work_in_tmp_dir
from gaptrain.calculators import get_gp_var_quip_out
from multiprocessing import Pool
from gaptrain.log import logger


def calc_error(frame, gap, method_name):
    """Calculate the error between the ground truth and the GAP prediction"""
    frame.single_point(method_name=method_name,  n_cores=1)

    pred = frame.copy()
    pred.run_gap(gap=gap, n_cores=1)

    error = np.abs(pred.energy - frame.energy)
    logger.info(f'|E_GAP - E_0| = {np.round(error, 3)} eV')
    return error


def remove_intra(configs, gap):
    """
    Remove the intramolecular energy and forces from a set of configurations
    using a GAP

    :param configs: (gt.ConfigurationSet)
    :param gap: (gt.GAP)
    """

    if isinstance(gap, gt.gap.IIGAP):
        logger.info('Removing the intramolecular energy and forces..')
        intra_configs = configs.copy()

        for intra_gap in gap.intra_gaps:
            intra_configs.parallel_gap(gap=intra_gap)

            for config, i_config in zip(configs, intra_configs):
                config.energy -= i_config.energy
                config.forces -= i_config.forces

    return configs


def get_active_config_diff(config, gap, temp, e_thresh, max_time_fs,
                           ref_method_name='dftb', curr_time_fs=0, n_calls=0,
                           extra_time_fs=0, **kwargs):
    """
    Given a configuration run MD with a GAP until the absolute error between
    the predicted and true values is above a threshold

    --------------------------------------------------------------------------
    :param config: (gt.Configuration)

    :param gap: (gt.GAP)

    :param e_thresh: (float) Threshold energy error (eV) above which the
                     configuration is returned

    :param temp: (float) Temperature to propagate GAP-MD

    :param max_time_fs: (float)

    :param ref_method_name: (str)

    :param curr_time_fs: (float)

    :param n_calls: (int) Number of times this function has been called

    :param extra_time_fs: (float) Some extra time to run initially e.g. as the
                          GAP is already likely to get to e.g. 100 fs, so run
                          that initially and don't run ground truth evaluations

    :return: (gt.Configuration)
    """
    if float(temp) < 0:
        raise ValueError('Cannot run MD with a negative temperature')

    if float(e_thresh) < 0:
        raise ValueError(f'Error threshold {e_thresh} must be positive (eV)')

    if extra_time_fs > 0:
        logger.info(f'Running an extra {extra_time_fs:.1f} fs of MD before '
                    f'calculating an error')

    md_time_fs = 2 + n_calls**3 + float(extra_time_fs)
    gap_traj = gt.md.run_gapmd(config,
                               gap=gap,
                               temp=float(temp),
                               dt=0.5,
                               interval=4,
                               fs=md_time_fs,
                               n_cores=1,
                               **kwargs)

    # Actual initial time, given this function can be called multiple times
    for frame in gap_traj:
        frame.t0 = curr_time_fs + extra_time_fs

    # Evaluate the error on the final frame
    error = calc_error(frame=gap_traj[-1], gap=gap, method_name=ref_method_name)

    # And the number of ground truth evaluations for this configuration
    n_evals = n_calls + 1

    if error > 100 * e_thresh:
        logger.error('Huge error: 100x threshold, returning the first frame')
        gap_traj[0].single_point(method_name=ref_method_name, n_cores=1)
        gap_traj[0].n_evals = n_evals + 1
        return gap_traj[0]

    if error > 10 * e_thresh:
        logger.warning('Error 10 x threshold! Taking the last frame less than '
                       '10x the threshold')
        # Stride through only 10 frames to prevent very slow backtracking
        for frame in reversed(gap_traj[::max(1, len(gap_traj)//10)]):
            error = calc_error(frame, gap=gap, method_name=ref_method_name)
            n_evals += 1

            if e_thresh < error < 10 * e_thresh:
                frame.n_evals = n_evals
                return frame

    if error > e_thresh:
        gap_traj[-1].n_evals = n_evals
        return gap_traj[-1]

    if curr_time_fs + md_time_fs > max_time_fs:
        logger.info(f'Reached the maximum time {max_time_fs} fs, returning '
                    f'None')
        return None

    # Increment t_0 to the new time
    curr_time_fs += md_time_fs

    # If the prediction is within the threshold then call this function again
    return get_active_config_diff(config, gap, temp, e_thresh, max_time_fs,
                                  curr_time_fs=curr_time_fs,
                                  ref_method_name=ref_method_name,
                                  n_calls=n_calls+1,
                                  **kwargs)


def get_active_config_qbc(config, gap, temp, std_e_thresh, max_time_fs, **kwargs):
    """
    Generate an 'active' configuration, i.e. a configuration to be added to the
    training set by active learning, using a query-by-committee model, where
    the prediction between different models (standard deviation) exceeds a
    threshold (std_e_thresh)

    ------------------------------------------------------------------------
    :param config: (gt.Configuration)

    :param gap: (gt.GAPEnsemble) An 'ensemble' of GAPs trained on the same/
                similar data to make predictions with

    :param temp: (float) Temperature for the GAP-MD

    :param std_e_thresh: (float) Threshold for the maximum standard deviation
                         between the GAP predictions (on the whole system),
                         above which a frame is added

    :param max_time_fs: (float)

    :return: (gt.Configuration)
    """
    n_iters, curr_time = 0, 0.0

    while curr_time < max_time_fs:

        gap_traj = gt.md.run_gapmd(config,
                                   gap=gap.gaps[0],
                                   temp=float(temp),
                                   dt=0.5,
                                   interval=4,
                                   fs=2 + n_iters**3,
                                   n_cores=1,
                                   **kwargs)

        for frame in gap_traj[::max(1, len(gap_traj)//10)]:
            pred_es = []
            # Calculated predicted energies in serial for all the gaps
            for single_gap in gap.gaps:
                frame.run_gap(gap=single_gap, n_cores=1)
                pred_es.append(frame.energy)

            # and return the frame if the standard deviation in the predictions
            # is larger than a threshold
            std_e = np.std(np.array(pred_es))
            if std_e > std_e_thresh:
                return frame

            logger.info(f'σ(t={curr_time:.1f}) = {std_e:.6f}')

        n_iters += 1
        curr_time += 2 + n_iters**3

    return None


@work_in_tmp_dir(kept_exts=[], copied_exts=['.xml'])
def get_active_config_gp_var(config, gap, temp, var_e_thresh,
                             max_time_fs, **kwargs):
    """
    Generate an active configuration by calculating the predicted variance
    on a configuration using the trained gap

    :param config: (gt.Configuration) Initial configuration to propagate GAP-MD
                   from

    :param gap: (gt.GAP)

    :param temp: (float) Temperature for the GAP-MD

    :param var_e_thresh: (float) Threshold for the maximum atomic variance in
                         the GAP predicted energy

    :param max_time_fs: (float)

    :return: (gt.Configuration)
    """
    # Needs a single gap to calculate the variance simply, if this is a II or
    # then assume the intra is well trained and use inter prediction
    gap_name = gap.inter_gap.name if hasattr(gap, 'inter') else gap.name

    def run_quip():
        """Use QUIP on a set of configurations to predict the variance"""
        with open('quip.out', 'w') as tmp_out_file:
            subprocess = Popen(gt.GTConfig.quip_command +
                               ['calc_args=local_gap_variance', 'E=T', 'F=T',
                                'atoms_filename=tmp_traj.xyz',
                                f'param_filename={gap_name}.xml'],
                               shell=False, stdout=tmp_out_file, stderr=PIPE)
            subprocess.wait()

        return None

    n_iters, curr_time = 0, 0.0
    while curr_time < max_time_fs:

        gap_traj = gt.md.run_gapmd(config,
                                   gap=gap,
                                   temp=float(temp),
                                   dt=0.5,
                                   interval=max(1, (2 + n_iters**3)//20),
                                   fs=2 + n_iters**3,
                                   n_cores=1,
                                   **kwargs)

        curr_time += 2 + n_iters**3

        gap_traj.save('tmp_traj.xyz')
        run_quip()
        gp_vars = get_gp_var_quip_out(configuration=config)

        # Enumerate all the frames and return one that has a variance above the
        # threshold
        for frame, var in zip(gap_traj, gp_vars):

            logger.info(f'var(GAP pred, t={curr_time})={np.max(var):.5f} eV')
            if np.max(var) > var_e_thresh:
                return frame

        n_iters += 1

    return None


def get_active_configs(config, gap, ref_method_name, method='diff',
                       max_time_fs=1000, n_configs=10, temp=300, e_thresh=0.1,
                       min_time_fs=0, **kwargs):
    """
    Generate n_configs using on-the-fly active learning parallelised over
    GTConfig.n_cores

    --------------------------------------------------------------------------
    :param config: (gt.Configuration | gt.ConfigurationSet) Initial
                    configuration(s) to propagate from

    :param gap: (gt.gap.GAP) GAP to run MD with

    :param ref_method_name: (str) Name of the method to use as the ground truth

    :param method: (str) Name of the strategy used to generate new configurations

    :param max_time_fs: (float) Maximum propagation time in the active learning
                        loop. Default = 1 ps

    :param n_configs: (int) Number of configurations to generate

    :param temp: (float) Temperature in K to run the intermediate MD with

    :param e_thresh: (float) Energy threshold in eV above which the MD frame
                     is returned by the active learning function i.e
                     E_t < |E_GAP - E_true|  method='diff'

    :param min_time_fs: (float) Minimum propagation time in the active learning
                        loop. If non-zero then will run this amount of time
                        initially then look for a configuration with a
                        |E_0 - E_GAP| > e_thresh

    :param kwargs: Additional keyword arguments passed to the GAP MD function

    :return:(gt.ConfigurationSet)
    """
    if int(n_configs) < int(gt.GTConfig.n_cores):
        raise NotImplementedError('Active learning is only implemented using '
                                  'one core for each process. Please use '
                                  'n_configs >= gt.GTConfig.n_cores')
    results = []
    configs = gt.Data()
    logger.info('Searching for "active" configurations with a threshold of '
                f'{e_thresh:.6f} eV')

    if method.lower() == 'diff':
        function = get_active_config_diff
        args = [gap, temp, e_thresh, max_time_fs, ref_method_name,
                0, 0, min_time_fs]

    elif method.lower() == 'gp_var':
        function = get_active_config_gp_var
        args = [gap, temp, e_thresh, max_time_fs]

    else:
        raise ValueError('Unsupported active method')

    logger.info(f'Using {gt.GTConfig.n_cores} processes')
    with Pool(processes=int(gt.GTConfig.n_cores)) as pool:

        for i in range(n_configs):

            # Prepend the arguments with the initial configuration
            if isinstance(config, gt.ConfigurationSet):
                init_config = config[i].copy()
            else:
                init_config = config.copy()

            result = pool.apply_async(func=function,
                                      args=[init_config] + args,
                                      kwds=kwargs)
            results.append(result)

        for result in results:

            try:
                config = result.get(timeout=None)
                if config is not None and config.energy is not None:
                    configs.add(config)

            # Lots of different exceptions can be raised when trying to
            # generate an active config, continue regardless..
            except Exception as err:
                logger.error(f'Raised an exception in calculating the energy\n'
                             f'{err}')
                continue

    if method.lower() != 'diff':
        logger.info('Running reference calculations on configurations '
                    f'generated by {method}')
        configs.single_point(method_name=ref_method_name)

        # Set the number of ground truth function calls for each iteration
        for config in configs:
            config.n_evals = 1

    return configs


def get_init_configs(system, init_configs=None, n=10, method_name=None):
    """Generate a set of initial configurations to use for active learning"""

    if init_configs is not None:

        if all(cfg.energy is not None for cfg in init_configs):
            logger.info(f'Initialised with {len(init_configs)} configurations '
                        f'all with defined energy')
            return init_configs
        else:
            init_configs.single_point(method_name=method_name)
            return init_configs

    # Initial configurations are not defined, so make some - will use random
    # with the largest maximum distance between molecules possible
    max_vdw = max(Atom(symbol).vdw_radius for symbol in system.atom_symbols)
    ideal_dist = 2*max_vdw - 0.5    # Desired minimum distance in Å

    # Reduce the distance until there is a probability at least 0.1 that a
    # random configuration can be generated with that distance threshold
    p_acc, dist = 0, ideal_dist+0.2

    while p_acc < 0.1:
        n_generated_configs = 0
        dist -= 0.2                 # Reduce the minimum distance requirement

        for _ in range(10):
            try:
                _ = system.random(min_dist_threshold=dist,
                                  max_attempts=2000)
                n_generated_configs += 1

            except ex.RandomiseFailed:
                continue

        p_acc = n_generated_configs / 10
        logger.info(f'Generated configurations with p={p_acc:.2f} with a '
                    f'minimum distance of {dist:.2f}')

    init_configs = gt.Data(name='init_configs')
    # Finally generate the initial configurations
    while len(init_configs) < n:
        try:
            init_configs += system.random(min_dist_threshold=dist,
                                          with_intra=True)
        except ex.RandomiseFailed:
            continue
    logger.info(f'Added {len(init_configs)} configurations with min dist = '
                f'{dist:.3f} Å')

    if method_name is None:
        logger.warning('Have no method - not evaluating energies')
        return init_configs

    # And run the desired method in parallel across them
    method = getattr(init_configs, f'parallel_{method_name.lower()}')
    method()

    init_configs.save()
    return init_configs


def train(system: gt.System,
          method_name: str,
          gap=None,
          max_time_active_fs=1000,
          min_time_active_fs=0,
          n_configs_iter=10,
          temp=300,
          active_e_thresh=None,
          active_method='diff',
          max_energy_threshold=None,
          validate=False,
          tau=None,
          tau_max=None,
          val_interval=None,
          max_active_iters=50,
          n_init_configs=10,
          init_configs=None,
          remove_intra_init_configs=True,
          fix_init_config=False,
          bbond_energy=None,
          fbond_energy=None,
          init_active_temp=None):
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

    :param min_time_active_fs: (float) Minimum propagation time for an
                               active learnt configuration. Will be updated
                               so the error is only calculated where the GAP
                               is unlikely to be accurate

    :param n_configs_iter: (int) Number of configurations to generate per
                           active learning cycle

    :param temp: (float) Temperature in K to propagate active learning at -
                 higher is better for stability but requires more training


    :param active_method: (str) Method used to generate active learnt
                          configurations. One of ['diff', 'gp_var']

    :param active_e_thresh: (float) Threshold in eV (E_t) above which a
                            configuration is added to the potential. If None
                            then will use 1 kcal mol-1 molecule-1

                            1. active_method='diff': |E_0 - E_GAP| > E_t

                            2. active_method='qbc': σ(E_GAP1, E_GAP2...) > E_t

                            3. active_method='gp_var': σ^2_GAP(predicted) > E_t

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

    :param tau_max: (float | None) Maximum τ_acc in fs if float, will break out
                    of the active learning loop if this value is reached. If
                    None then won't break out

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

    :param remove_intra_init_configs: (bool) Whether the intramolecular
                                      component of the energy/force needs to
                                      be removed prior to training with
                                      init_configs. only applies for IIGAP
                                      and init_configs != None

    :param fix_init_config: (bool) Always start from the same initial
                            configuration for the active learning loop, if
                            False then the minimum energy structure is used.
                            Useful for TS learning, where dynamics should be
                            propagated from a saddle point not the minimum

    :param bbond_energy: (dict | None) Additional energy to add to a breaking
                         bond. e.g. bbond_energy={(0, 1), 0.1} Adds 0.1 eV
                         to the 'bond' between atoms 0 and 1 as velocities
                        shared between the atoms in the breaking bond direction

    :param fbond_energy: (dict | None) As bbond_energy but in the direction to
                         form a bond


    :param init_active_temp: (float | None) Initial temperature for velocities
                             in the 'active' MD search for configurations

    :return: (gt.Data, gt.GAP)
    """
    init_configs = get_init_configs(init_configs=init_configs,
                                    n=n_init_configs,
                                    method_name=method_name,
                                    system=system)

    # Remove the intra-molecular energy if an intra+inter (II) GAP is being
    # trained
    if remove_intra_init_configs:
        remove_intra(init_configs, gap=gap)

    # Initial configuration must have energies
    assert all(cfg.energy is not None for cfg in init_configs)

    if gap is None:
        gap = gt.GAP(name=unique_name('active_gap'), system=system)

    # Initialise a τ metric with default parameters
    if validate and tau is None:
        # 1 ps default maximum tau
        tau = gt.loss.Tau(configs=get_init_configs(system, n=5),
                          e_lower=0.043363 * len(system.molecules),
                          max_fs=tau_max if tau_max is not None else 1000)

    # Default to validating 10 times through the training
    if validate and val_interval is None:
        val_interval = max(max_active_iters // 10, 1)

    # Initialise training data
    train_data = gt.Data(name=gap.name)
    train_data += init_configs

    # and train an initial GAP
    gap.train(init_configs)

    if active_e_thresh is None:
        if active_method.lower() == 'diff':
            #                 1 kcal mol-1 molecule-1
            active_e_thresh = 0.043363 * len(system.molecules)

        if active_method.lower() == 'qbc':
            # optimised on a small box of water. std dev. for total energy
            active_e_thresh = 1E-6 * len(system.molecules)

        if active_method.lower() == 'gp_var':
            # Threshold for maximum per-atom GP variance (eV atom^-1)
            active_e_thresh = 5E-5

    # Initialise the validation output file
    if validate:
        tau_file = open(f'{gap.name}_tau.txt', 'w')
        print('Iteration    n_evals      τ_acc / fs', file=tau_file)

    # Run the active learning loop, running iterative GAP-MD
    for iteration in range(max_active_iters):

        # Set the configuration(s) from which GAP-MD will be run
        min_idx = int(np.argmin(train_data.energies()))
        init_config = init_configs if fix_init_config else train_data[min_idx]

        configs = get_active_configs(init_config,
                                     gap=gap,
                                     ref_method_name=method_name,
                                     method=str(active_method),
                                     n_configs=n_configs_iter,
                                     temp=temp,
                                     e_thresh=active_e_thresh,
                                     max_time_fs=max_time_active_fs,
                                     min_time_fs=min_time_active_fs,
                                     bbond_energy=bbond_energy,
                                     fbond_energy=fbond_energy,
                                     init_temp=init_active_temp)

        # Active learning finds no configurations,,
        if len(configs) == 0:
            # Calculate the final tau if we're running with validation
            if validate:
                tau.calculate(gap=gap, method_name=method_name)
                print(iteration, tau.value, sep='\t\t\t', file=tau_file)

            logger.info('No configs to add. Active learning = DONE')
            break

        min_time_active_fs = min(config.t0 for config in configs)
        logger.info(f'All active configurations reached t = '
                    f'{min_time_active_fs} fs before an error exceeded the '
                    f'threshold of {active_e_thresh:.3f} eV')

        # Only training the intermolecular component in a I+I GAP
        if isinstance(gap, gt.IIGAP):
            remove_intra(configs, gap=gap)

        train_data += configs

        # If required remove high-lying energy configuration from the data
        if max_energy_threshold is not None:
            train_data.remove_above_e(max_energy_threshold)

        # Retrain on these new data
        gap.train(train_data)

        # Print the accuracy
        if validate and iteration % val_interval == 0:

            tau.calculate(gap=gap, method_name=method_name)
            print(f'{iteration:<13g}'
                  f'{sum(config.n_evals for config in train_data):<13g}'
                  f'{tau.value}', sep='\t', file=tau_file)

            if np.abs(tau.value - tau.max_time) < 1:
                logger.info('Reached the maximum tau. Active learning = DONE')
                break

    return train_data, gap


def train_ii(system, method_name, intra_temp=1000, inter_temp=300, **kwargs):
    """
    Train an intra+intermolecular from just a system

    ---------------------------------------------------------------------------
    :param system: (gt.System)

    :param method_name: (str) e.g dftb

    :param intra_temp: (float) Temperature to run the intramolecular training

    :param inter_temp: (float) Temperature to run the intermolecular training
    """
    if system.n_unique_molecules < 1:
        raise ValueError('Must have at least one molecule to train GAP for')

    if 'temp' in kwargs:
        raise ValueError('Ambiguous specification, please specify: intra_temp '
                         'and inter_temp')

    all_data, intra_gaps = [], []

    for unq_mol in system.unique_molecules:
        # Create a system of just the monomer to train the intra-molecular
        # component of the system
        intra_system = gt.System(unq_mol.molecule,
                                 box_size=system.box.size)

        # and train the intra component using a bespoke GAP
        gap = gt.GAP(name=f'intra_{unq_mol.molecule.name}',
                     system=intra_system)

        intra_data, intra_gap = train(intra_system,
                                      method_name=method_name,
                                      gap=gap,
                                      temp=intra_temp,
                                      **kwargs)
        all_data.append(intra_data)
        intra_gaps.append(gt.IntraGAP(name=f'intra_{unq_mol.molecule.name}',
                                      unique_molecule=unq_mol))

    inter_gap = gt.InterGAP(name=f'inter', system=system)

    # And finally train the inter component of the energy
    inter_data, gap = gt.active.train(system,
                                      method_name=method_name,
                                      temp=inter_temp,
                                      gap=gt.IIGAP(inter_gap, *intra_gaps),
                                      **kwargs)
    all_data.append(inter_data)

    return tuple(all_data), gap
