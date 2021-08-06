import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
plt.style.use('paper')


def mean(list_arr):
    return [np.average(arr) if len(arr) > 0 else -1000
            for arr in list_arr]


def error(list_arr):
    return [np.std(arr)/np.sqrt(len(arr)-1) if len(arr) > 2 else 0
            for arr in list_arr]


if __name__ == '__main__':

    n_atoms_list = [8, 10, 6, 12, 3, 5, 5, 18, 16, 15, 12, 11, 6, 9, 14, 6,
                    12, 14, 7, 11, 13, 15, 13, 8, 9, 3, 18, 9, 13, 9, 10, 7]

    n_evals = [
        [433, 419, 369],      # 0
        [609, 510, 487],
        [100, 141, 149],
        [229, 142, 224],
        [80, 133, 56],
        [62, 51, 23],
        [130, 115, 82],       # 6
        [2385, 2612, 2666],
        [3245, 3085, 3183],
        [3525, 3945, 3744],
        [1844, 1850, 2142],    # 10
        [1026, 1250, 1215],
        [359, 352, 368],
        [1427, 1161, 1493],
        [3677, 3581, ],
        [348, 539, 384],      # 15
        [2614, 2526, 2575],
        [925, 828, 752],
        [273, 245, 254],
        [503, 563, 401],
        [2143, 1556, 1952],                   # 20
        [980, 1233, 1172],
        [840, 744, 1189],
        [399, 359, 365],
        [1072, 999, 972],
        [57, 77, 68],
        [2379, 2155, 2051],
        [402, 407, 340],
        [3551, 3464, 3372],
        [315, 372, 279],
        [],                 # 30
        [292, 235, 191]
    ]

    n_train_configs = [
        [125, 117, 102],
        [147, 150, 136],
        [44, 42, 43],
        [68, 60, 63],
        [24, 38, 23],
        [20, 17, 13],      # 5
        [34, 33, 27],
        [618, 643, 649],
        [1001, 1003, 1004],
        [955, 963, 953],
        [407, 455, 483],               # 10
        [247, 289, 284],
        [103, 103, 110],
        [365, 305, 352],
        [884, 846, ],
        [95, 119, 111],
        [633, 636, 628],
        [234, 196, 202],
        [71, 74, 64],
        [137, 140, 116],
        [495, 401, 507],
        [250, 303, 272],
        [210, 193, 270],
        [109, 89, 87],
        [255, 248, 232],
        [23, 31, 24],
        [550, 532, 490],
        [113, 114, 108],
        [802, 824, 802],
        [87, 104, 80],
        [],
        [73, 66, 62]
    ]

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

    evals_line = ax1.errorbar(n_atoms_list, mean(n_evals),
                yerr=error(n_evals),
                marker='o', lw=1, mfc='blue', ecolor='blue', alpha=0.6,
                fmt='o', color='blue',
                label='$n_{eval}$')

    train_line = ax2.errorbar(n_atoms_list, mean(n_train_configs),
                yerr=error(n_train_configs),
                marker='s', lw=1, mfc='red', ecolor='red', alpha=0.6, fmt='o',
                color='red',
                label='$n_{train}$')

    for ax in (ax1, ax2):
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.i'))

    ax1.set_xlabel('# atoms')
    ax1.set_ylabel('$n_{eval}$')
    ax1.set_ylim(-0.02*4000, 4000)
    ax1.set_xticks(list(range(min(n_atoms_list), max(n_atoms_list), 2)))
    ax1.legend(loc='lower right')

    ax2.set_xlabel('# atoms')
    ax2.set_ylabel('$n_{train}$')
    ax2.set_ylim(-0.02*1100, 1100)
    ax2.set_xticks(list(range(min(n_atoms_list), max(n_atoms_list), 2)))
    ax2.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('solvent_scaling.pdf')
