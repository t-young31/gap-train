import matplotlib.pyplot as plt
import numpy as np
plt.style.use('paper')


def mean(list_arr):
    return [np.average(arr) for arr in list_arr]


def error(list_arr):
    return [np.std(arr)/np.sqrt(len(arr)-1) for arr in list_arr]


if __name__ == '__main__':

    n_carbons = [1, 2, 3, 4, 5, 6, 7, 8]

    n_evals = [
        [72, 48, 66],         # n = 1
        [480, 373, 416],
        [1353, 1611, 1442],
        [1735, 1815, 2018],

        # -- 50 max AL iters
        # [1514, 1486, 1367],   # n = 5
        # [1415, 1454, 1418],
        # [1247,1341, 1307],
        # [1177, 1023, 1206],

        # -- 100 max AL iters
        [3344, 3294, 3189],
        [3414, 3680, 3342],
        [2899, 2968, 2885],
        [2717, 2567, 2627]
    ]

    n_train_configs = [
        [31, 30, 35],
        [118, 102, 114],
        [348, 396, 363],
        [471, 455, 456],

        # [510, 508, 510],
        # [509, 509, 510],
        # [510, 510, 510],
        # [510, 510, 510]

        [988, 991, 987],
        [986, 986, 994],
        [1009, 1005, 1003],
        [1010, 1010, 1010]
    ]

    max_tau = 1000  # fs
    taus = [                           # Evaluated over 3 separate trajectories
        [max_tau, max_tau, max_tau],
        [max_tau, max_tau, max_tau],
        [max_tau, max_tau, max_tau],
        [max_tau, max_tau, max_tau],

        # [966.67, 1020.0, 846.67],
        # [1000.0, 986.67, 913.33],
        # [940.0, 1033.33, 800.0],
        # [300.0, 186.67, 653.33]

        [max_tau, max_tau, max_tau],
        [max_tau, max_tau, 873.33],
        [1026, max_tau, max_tau],
        [800.0, 866.67, 733.33]
    ]

    fig, ax = plt.subplots()
    evals_line = ax.errorbar(n_carbons, mean(n_evals),
                yerr=error(n_evals),
                marker='o', lw=1, mfc='blue', ecolor='blue', alpha=0.6,
                ls='--', color='blue',
                label='$n_{eval}$')

    train_line = ax.errorbar(n_carbons, mean(n_train_configs),
                yerr=error(n_train_configs),
                marker='s', lw=1, mfc='red', ecolor='red', alpha=0.6, ls='--',
                color='red',
                label='$n_{train}$')

    ax2 = ax.twinx()
    taus_line = ax2.errorbar(n_carbons, mean(taus),
                 yerr=error(taus),
                 marker='^', lw=1, mfc='purple', ecolor='purple', alpha=0.6,
                 color='purple',
                 label='$\\tau$')

    ax.set_xlabel('# carbons')
    ax.set_ylabel('n')

    ax2.set_ylabel('$\\tau_{acc}$ / fs')
    ax2.set_ylim(0, 1100)
    ax2.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])

    plt.legend(loc='lower right', handles=[evals_line, train_line, taus_line])

    plt.tight_layout()
    plt.savefig('alkane_scaling.pdf')
