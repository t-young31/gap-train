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
        intra_configs.parallel_gap(gap=gap.intra)

        for config, i_config in zip(configs, intra_configs):
            config.energy -= i_config.energy
            config.forces -= i_config.forces

        # If there is also a solute in the system then remove the energy
        # associated with it
        if isinstance(gap, gt.gap.SSGAP):
            solute_configs = configs.copy()
            solute_configs.parallel_gap(gap=gap.solute_intra)

            for config, s_config in zip(configs, solute_configs):
                config.energy -= s_config.energy
                config.forces -= s_config.forces

    return configs


def get_active_config_true(config, gap, temp, e_thresh, max_time_fs,
                           ref_method_name='dftb', curr_time_fs=0, n_calls=0,
                           extra_time_fs=0):
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
                               n_cores=1)

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
    return get_active_config_true(config, gap, temp, e_thresh, max_time_fs,
                                  curr_time_fs=curr_time_fs,
                                  ref_method_name=ref_method_name,
                                  n_calls=n_calls+1)


def get_active_config_qbc(config, gap, temp, std_e_thresh, max_time_fs):
    """
    Generate an 'active' configuration i.e. a configuration to be added to the
    training set by active learning using a query-by-committee model where
    above some standard deviation in the prediction between different models
    exceeds a threshold (std_e_thresh)

    ------------------------------------------------------------------------
    :param config: (gt.Configuration)
    :param gap: (gt.GAPEnsemble)
    :param temp: (float)
    :param std_e_thresh: (float) eV
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
                                   n_cores=1)

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
                frame.t0 = 0
                return frame

            logger.info(f'σ(t={curr_time:.1f}) = {std_e:.6f}')

        n_iters += 1
        curr_time += 2 + n_iters**3

    return gap_traj[-1]


def get_active_configs(config, gap, ref_method_name, method='true',
                       max_time_fs=1000, n_configs=10, temp=300, e_thresh=0.1,
                       min_time_fs=0, std_e_thresh=2E-5):
    """
    Generate n_configs using on-the-fly active learning parallelised over
    GTConfig.n_cores

    --------------------------------------------------------------------------
    :param config: (gt.Configuration) Initial configuration to propagate from

    :param gap: (gt.gap.GAP) GAP to run MD with

    :param ref_method_name: (str) Name of the method to use as the ground truth

    :param method: (str) Name of the strategy used to generate new configurations

    :param max_time_fs: (float) Maximum propagation time in the active learning
                        loop. Default = 1 ps

    :param n_configs: (int) Number of configurations to generate

    :param temp: (float) Temperature in K to run the intermediate MD with

    :param e_thresh: (float) Energy threshold in eV above which the MD frame
                     is returned by the active learning function i.e
                     E_t < |E_GAP - E_true|  method=='true'

    :param min_time_fs: (float) Minimum propagation time in the active learning
                        loop. If non-zero then will run this amount of time
                        initially then look for a configuration with a
                        |E_0 - E_GAP| > e_thresh

    :param std_e_thresh: (float) Standard deviation in the energy above which a
                         configuration is added if method=='qbc'

    :return:(gt.ConfigurationSet)
    """
    if int(n_configs) < int(gt.GTConfig.n_cores):
        raise NotImplementedError('Active learning is only implemented using '
                                  'one core for each process. Please use '
                                  'n_configs >= gt.GTConfig.n_cores')
    results = []
    configs = gt.Data()

    if method.lower() == 'true':
        function = get_active_config_true
        args = (config, gap, temp, e_thresh, max_time_fs, ref_method_name,
                0, 0, min_time_fs)

    elif method.lower() == 'qbc':
        function = get_active_config_qbc
        # Train a few GAPs on the same data
        gap = gt.gap.GAPEnsemble(name=f'{gap.name}_ensemble', gap=gap)
        gap.train()

        args = (config, gap, temp, std_e_thresh, max_time_fs)

    else:
        raise ValueError('Unsupported active method')

    logger.info(f'Using {gt.GTConfig.n_cores} processes')
    with Pool(processes=int(gt.GTConfig.n_cores)) as pool:

        for _ in range(n_configs):
            result = pool.apply_async(func=function, args=args)
            results.append(result)

        for result in results:
            try:
                config = result.get(timeout=None)
                if config is not None and config.energy is not None:
                    configs.add(config)

            # Lots of different exceptions can be raised when trying to
            # generate an active config, continue regardless..
            except:
                logger.error('Raised an exception in calculating the energy')
                continue

    if method.lower() == 'qbc':
        logger.info('Running reference calculations on configurations '
                    'generated by query-by-commitiee')
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


def train(system,
          method_name,
          gap=None,
          max_time_active_fs=1000,
          min_time_active_fs=0,
          n_configs_iter=10,
          temp=300,
          active_e_thresh=None,
          active_method='true',
          qbc_std_e_thresh=2E-5,
          max_energy_threshold=None,
          validate=False,
          tau=None,
          tau_max=None,
          val_interval=None,
          max_active_iters=50,
          n_init_configs=10,
          init_configs=None,
          remove_intra_init_configs=True,
          fix_init_config=False):
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

    :param active_e_thresh: (float) Threshold in eV (E_t) above which a
                            configuration is added to the potential. If None
                            then will use 1 kcal mol-1 molecule-1

    :param active_method: (str) Method used to generate active learnt
                          configurations. One of ['true', 'qbc']

    :param qbc_std_e_thresh: (float) Standard deviation in energies for an
                             'active' configuration to be added

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
    :return:
    """
    init_configs = get_init_configs(init_configs=init_configs,
                                    n=n_init_configs,
                                    method_name=method_name,
                                    system=system)

    # Remove the intra-molecular energy if an intra+inter (II) GAP is being
    # trained
    do_remove_intra = isinstance(gap, gt.IIGAP)
    if do_remove_intra and remove_intra_init_configs:
        remove_intra(init_configs, gap=gap)

    # Initial configuration must have energies
    assert all(cfg.energy is not None for cfg in init_configs)

    if gap is None:
        gap = gt.GAP(name=unique_name('active_gap'), system=system)

    # Initialise a τ metric with default parameters
    if validate and tau is None:
        tau = gt.loss.Tau(configs=get_init_configs(system, n=5))

    # Default to validating 10 times through the training
    if validate and val_interval is None:
        val_interval = max(max_active_iters // 10, 1)

    # Initialise training data
    train_data = gt.Data(name=gap.name)
    train_data += init_configs

    # and train an initial GAP
    gap.train(init_configs)

    if active_e_thresh is None:
        #                 1 kcal mol-1 molecule-1
        active_e_thresh = 0.043363 * len(system.molecules)

    # Initialise the validation output file
    if validate:
        tau_file = open(f'{gap.name}_tau.txt', 'w')
        print('Iteration    n_evals      τ_acc / fs', file=tau_file)

    # Run the active learning loop, running iterative GAP-MD
    for iteration in range(max_active_iters):

        # Set the configuration from which GAP-MD will be run
        min_idx = int(np.argmin(train_data.energies()))
        init_config = train_data[0] if fix_init_config else train_data[min_idx]

        configs = get_active_configs(init_config,
                                     gap=gap,
                                     ref_method_name=method_name,
                                     method=str(active_method),
                                     std_e_thresh=qbc_std_e_thresh,
                                     n_configs=n_configs_iter,
                                     temp=temp,
                                     e_thresh=active_e_thresh,
                                     max_time_fs=max_time_active_fs,
                                     min_time_fs=min_time_active_fs)

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

        if do_remove_intra:
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

            if tau_max is not None and np.abs(tau.value - tau_max) < 1:
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

    if system.n_unique_molecules > 1:
        raise ValueError('Can only train an inter+intra for a single bulk '
                         'molecular species')

    if system.n_unique_molecules < 1:
        raise ValueError('Must have at least one molecule to train GAP for')

    if 'temp' in kwargs:
        raise ValueError('Ambiguous specification, please specify: intra_temp '
                         'and inter_temp')

    # Create a system of just the monomer to train the intra-molecular
    # component of the system
    molecule = system.molecules[0]
    intra_system = gt.System(box_size=system.box.size)
    intra_system.add_molecules(molecule)

    # and train the intra component using a bespoke GAP
    gap = gt.GAP(name=f'intra_{molecule.name}', system=intra_system)
    intra_data, _ = train(intra_system,
                          method_name=method_name,
                          gap=gap,
                          temp=intra_temp,
                          **kwargs)

    if len(intra_data) == 0:
        raise RuntimeError('Failed to train the intra-system')

    # Now create an intra GAP that has the molecule indexes
    intra_gap = gt.gap.IntraGAP(name=f'intra_{molecule.name}',
                                system=system,
                                molecule=molecule)

    inter_gap = gt.InterGAP(name=f'inter_{molecule.name}',
                            system=system)

    # And finally train the inter component of the energy
    inter_data, gap = gt.active.train(system,
                                      method_name=method_name,
                                      temp=inter_temp,
                                      gap=gt.IIGAP(intra_gap, inter_gap),
                                      **kwargs)

    return (intra_data, inter_data), gap


def train_ss(system, method_name, intra_temp=1000, inter_temp=300, **kwargs):
    """
    Train an intra+intermolecular from just a system

    ---------------------------------------------------------------------------
    :param system: (gt.System)

    :param method_name: (str) e.g dftb

    :param intra_temp: (float) Temperature to run the intramolecular training

    :param inter_temp: (float) Temperature to run the intermolecular training
    """
    if system.n_unique_molecules != 2:
        raise ValueError('Can only train an solute-solvent GAP for a system '
                         'with two molecules, the solute and the solvent')

    # Find the least, and most abundant molecules in the system, as the solute
    # and solvent respectively
    names = [mol.name for mol in system.molecules]
    nm1, nm2 = tuple(set(names))
    solute_name, solv_name = (nm1, nm2) if names.count(nm1) == 1 else (nm2, nm1)

    solute = [mol for mol in system.molecules if mol.name == solute_name][0]
    solv = [mol for mol in system.molecules if mol.name == solv_name][0]

    data = []   # List of training data for all the components in the system

    # Train the intramolecular components of the potential for the solute and
    # the solvent
    for molecule in (solute, solv):
        # Create a system with only one molecule
        intra_system = gt.System(box_size=system.box.size)
        intra_system.add_molecules(molecule)

        # and train..
        logger.info(f'Training intramolecular component of {molecule.name}')
        mol_data, _ = gt.active.train(intra_system,
                                      gap=gt.GAP(name=f'intra_{molecule.name}',
                                                 system=intra_system),
                                      method_name=method_name,
                                      validate=False,
                                      temp=intra_temp,
                                      **kwargs)
        data.append(mol_data)

    # Recreate the GAPs with the full system (so they have the
    solv_gap = gt.gap.SolventIntraGAP(name=f'intra_{solv.name}', system=system)
    solute_gap = gt.gap.SoluteIntraGAP(name=f'intra_{solute.name}',
                                       system=system, molecule=solute)
    inter_gap = gt.InterGAP(name='inter', system=system)

    # and finally train the intermolecular part of the potential
    inter_data, gap = gt.active.train(system,
                                      method_name=method_name,
                                      gap=gt.gap.SSGAP(solute_intra=solute_gap,
                                                       solvent_intra=solv_gap,
                                                       inter=inter_gap),
                                      temp=inter_temp,
                                      **kwargs)
    data.append(inter_data)

    return tuple(data), gap
