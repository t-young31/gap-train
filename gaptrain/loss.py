import gaptrain as gt
import numpy as np
from gaptrain.log import logger


class DeltaError:
    """A error based on difference between two sets of configurations, one
    (probably) predicted with GAP and one ground truth"""

    def __str__(self):
        return (f'Loss: energy = {self.energy:.4f} eV, '
                f'force  = {self.force:.4f} eV Å-1')

    def function(self, deltas):
        """Function that transforms [∆v1, ∆v2, ..] into a float e.g. MSE"""
        raise NotImplementedError

    def loss(self, configs_a, configs_b, attr):
        """
        Compute the root mean squared loss between two sets of configurations

        -----------------------------------------------------------------------
        :param configs_a: (gaptrain.configurations.ConfigurationSet)

        :param configs_b: (gaptrain.configurations.ConfigurationSet)

        :param attr: (str) Attribute of a configuration to calculate use in
                           the loss function e.g. energy or froca

        :return: (gaptrain.loss.RMSE)
        """
        assert len(configs_a) == len(configs_b)

        deltas = []

        for (ca, cb) in zip(configs_a, configs_b):
            val_a, val_b = getattr(ca, attr), getattr(cb, attr)

            if val_a is None or val_b is None:
                logger.warning(f'Cannot calculate loss for {attr} at least '
                               f'one value was None')
                return None

            # Append the difference between the floats
            deltas.append(val_a - val_b)

        return self.function(np.array(deltas))

    def __init__(self, configs_a, configs_b):
        """
        A loss on energies (eV) and forces (eV Å-1). Force loss is computed
        element-wise

        :param configs_a: (gaptrain.configurations.ConfigurationSet)
        :param configs_b: (gaptrain.configurations.ConfigurationSet)
        """
        logger.info(f'Calculating loss between {len(configs_a)} configurations')

        self.energy = self.loss(configs_a, configs_b, attr='energy')
        self.force = self.loss(configs_a, configs_b, attr='forces')


class RMSE(DeltaError):
    """Root mean squared error"""

    def function(self, deltas):
        return np.sqrt(np.mean(np.square(deltas)))


class MSE(DeltaError):
    """Mean squared error"""

    def function(self, deltas):
        return np.mean(np.square(deltas))


class Tau:

    def __str__(self):
        err = np.round(self.error, 2) if self.error is not None else '??'
        return f'τ_acc = {np.round(self.value, 2)} ± {err} fs'

    def _calculate_single(self, init_config, gap, method_name):
        """
        Calculate a single τ_acc from one configuration

        :param init_config: (gt.Configuration)
        :param gap: (gt.GAP)
        :param method_name: (str) Ground truth method e.g. dftb, orca, gpaw
        """

        cuml_error, curr_time = 0, 0

        block_time = self.interval_time * gt.GTConfig.n_cores
        step_interval = self.interval_time // self.dt

        while curr_time < self.max_time:

            traj = gt.md.run_gapmd(init_config,
                                   gap=gap,
                                   temp=self.temp,
                                   dt=self.dt,
                                   interval=step_interval,
                                   fs=block_time,
                                   n_cores=min(gt.GTConfig.n_cores, 4))

            # Only evaluate the energy
            try:
                traj.single_point(method_name=method_name)
            except ValueError:
                logger.warning('Failed to calculate single point energies with'
                               f' {method_name}. τ_acc will be underestimated '
                               f'by <{block_time}')
                return curr_time

            pred = traj.copy()
            pred.parallel_gap(gap=gap)

            logger.info('      ___ |E_true - E_GAP|/eV ___')
            logger.info(f' t/fs      err      cumul(err)')

            for j in range(len(traj)):
                e_error = np.abs(traj[j].energy - pred[j].energy)

                # Add any error above the allowed threshold
                cuml_error += max(e_error - self.e_l, 0)
                curr_time += self.dt * step_interval
                logger.info(f'{curr_time:5.0f}     '
                            f'{e_error:6.4f}     '
                            f'{cuml_error:6.4f}')

                if cuml_error > self.e_t:
                    return curr_time

            init_config = traj[-1]

        logger.info(f'Reached max(τ_acc) = {self.max_time} fs')
        return self.max_time

    def calculate(self, gap, method_name):
        """Calculate the time to accumulate self.e_t eV of error above
        self.e_l eV"""

        taus = []

        for config in self.init_configs:
            tau = self._calculate_single(init_config=config,
                                         gap=gap,
                                         method_name=method_name)
            taus.append(tau)

        # Calculate τ_acc as the average ± the standard error in the mean
        self.value = np.average(np.array(taus))
        if len(taus) > 1:
            self.error = np.std(np.array(taus)) / np.sqrt(len(taus))   # σ / √N

        logger.info(str(self))
        return None

    def __init__(self, configs, e_lower=0.1, e_thresh=None, max_fs=1000,
                 interval_fs=20, temp=300, dt_fs=0.5):
        """
        τ_acc prospective error metric in fs

        ----------------------------------------------------------------------
        :param configs: (list(gt.Configuration) | gt.ConfigurationSet) A set of
                        initial configurations from which dynamics will be
                        propagated from

        :param e_lower: (float) E_l energy threshold in eV below which
                        the error is zero-ed, i.e. the acceptable level of
                        error possible in the system

        :param e_thresh: (float | None) E_t total cumulative error in eV. τ_acc
                         is defined at the time in the simulation where this
                         threshold is exceeded. If None then:
                         e_thresh = 10 * e_lower

        :param max_fs: (float) Maximum time in femto-seconds for τ_acc

        :zaram interval_fs: (float) Interval between which |E_true - E_GAP| is
                            calculated. *MUST* be at least one timestep

        :param temp: (float) Temperature of the simulation to perform

        :param dt_fs: (float) Timestep of the simulation in femto-seconds
        """
        if len(configs) < 1:
            raise ValueError('Must have at least one configuration to '
                             'calculate τ_acc from')

        if interval_fs < dt_fs:
            raise ValueError('The calculated interval must be more than a '
                             'single timestep')

        self.value = 0                          # τ_acc / fs
        self.error = None                       # standard error in the mean

        self.init_configs = configs

        self.dt = float(dt_fs)
        self.temp = float(temp)
        self.max_time = float(max_fs)
        self.interval_time = float(interval_fs)

        self.e_l = float(e_lower)
        self.e_t = 10 * self.e_l if e_thresh is None else float(e_thresh)

        logger.info('Successfully initialised τ_acc, will do a maximum of '
                    f'{int(self.max_time // self.interval_time)} reference '
                    f'calculations')
