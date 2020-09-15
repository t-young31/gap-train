import matplotlib.pyplot as plt
from gaptrain.exceptions import PlottingFailed
from matplotlib.colors import LogNorm
import matplotlib as mpl
import numpy as np

mpl.rcParams['axes.labelsize'] = 13
mpl.rcParams['lines.linewidth'] = 1
mpl.rcParams['lines.markersize'] = 5
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
mpl.rcParams['axes.linewidth'] = 1


def histogram(energies=None, forces=None, name=None, relative_energies=True,
              ref_energy=None):
    """
    Plot a histogram of energies, forces or both

    :param energies: (list(float))
    :param forces: (list(float))
    :param name: (str) or None
    :param relative_energies: (bool)
    :param ref_energy: (None | float) Energy reference for relative energies
    """
    assert energies is not None or forces is not None
    fig, ax = fig_ax(energies, forces)

    if energies is not None:
        plot_energy_hist(ax=ax if forces is None else ax[0],
                         energies=energies,
                         relative_energies=relative_energies,
                         ref_energy=ref_energy)

    if forces is not None:
        plot_forces_hist(ax=ax if energies is None else ax[1],
                         forces=forces)

    return show_or_save(name)


def plot_energy_hist(ax, energies, relative_energies=True,
                     ref_energy=None, color=None, label=None):
    """
    Plot an energy histogram on a matplotlib set of axes

    :param ax: (matplotlib.axes)
    :param energies: (list(float))
    :param relative_energies: (bool)
    :param ref_energy: (None | float) Energy reference for relative energies if
                       none and relative_energies=True the use the minimum in
                       list of energies
    :param color: (str | None)
    :param label: (str | None)
    """
    if len(energies) == 0:
        raise PlottingFailed('No energies')

    if relative_energies:
        ref_energy = min(energies) if ref_energy is None else ref_energy
        energies = np.array(energies) - ref_energy

    ax.hist(energies,
            bins=np.linspace(min(energies), max(energies), 30),
            alpha=0.5,
            edgecolor='darkblue' if color is None else color,
            linewidth=0.2,
            label=label)

    # Energy histogram formatting
    ax.set_xlabel('Energy / eV')
    ax.set_ylabel('Frequency')

    return None


def plot_forces_hist(ax, forces, color=None, label=None):
    """
    Plot an energy histogram on a matplotlib set of axes

    :param ax: (matplotlib.axes)
    :param forces: (list(float))
    :param color: (str | None)
    :param label: (str | None)
    """

    if len(forces) == 0:
        raise PlottingFailed('No energies')

    ax.hist(forces,
            bins=np.linspace(min(forces), max(forces), 50),
            color='orange' if color is None else color,
            alpha=0.5,
            edgecolor='darkorange' if color is None else color,
            linewidth=0.2,
            label=label)

    # Force histogram formatting
    ax.set_xlabel('|$F$| / ev Å$^{-1}$')
    ax.set_ylabel('Frequency')

    return None


def correlation(true_energies=None,
                predicted_energies=None,
                true_forces=None,
                predicted_forces=None,
                name=None):
    """
    Plot a correlation plot between predicted energies and/or forces and their
    ground truth values

    :param true_energies: (list(float))
    :param predicted_energies: (list(float))
    :param true_forces: (list(float))
    :param predicted_forces: (list(float))
    :param name: (str) or None
    """
    if true_energies is not None:
        assert predicted_energies is not None

    if true_forces is not None:
        assert predicted_forces is not None

    fig, ax = fig_ax(true_energies, true_forces)

    if true_energies is not None:
        ax_e = ax if true_forces is None else ax[0]
        # Scatter the true and predicted data
        ax_e.scatter(true_energies, predicted_energies,
                     edgecolors='k',
                     linewidth=0.5)

        # Plot a y = x line
        all_energies = [e for energies in (true_energies, predicted_energies)
                        for e in energies]

        min_e, max_e = min(all_energies), max(all_energies)
        delta = np.abs(max_e - min_e)/50
        pair = [min_e - delta, max_e + delta]

        ax_e.plot(pair, pair, lw=0.5, c='k', zorder=0)
        ax_e.set_xlim(*pair)
        ax_e.set_ylim(*pair)

        # Energy plot formatting
        ax_e.set_xlabel('True Energy / eV', size=12)
        ax_e.set_ylabel('Predicted Energy / eV', size=12)

        for axis in (ax_e.xaxis, ax_e.yaxis):
            axis.set_ticks([np.round(val, 1)
                            for val in np.linspace(np.round(min_e, 1),
                                                   np.round(max_e, 1), 6)])
    if true_forces is not None:
        ax_f = ax if true_energies is None else ax[1]

        all_forces = [f for forces in (true_forces, predicted_forces)
                      for f in forces]

        if any(f < 0 for f in all_forces):
            max_f = max(all_forces)
            min_f = -max_f

        else:
            min_f, max_f = min(all_forces), max(all_forces)

        # Histogram the forces in 2D
        hist = ax_f.hist2d(true_forces, predicted_forces,
                           density=True,
                           bins=[np.linspace(min_f, max_f, 200),
                                 np.linspace(min_f, max_f, 200)],
                           norm=LogNorm())

        delta = np.abs(max_f - min_f)/50
        pair = [min_f - delta, max_f + delta]

        # y = x and extended limits
        ax_f.plot(pair, pair, lw=0.5, c='k')
        ax_f.set_xlim(*pair)
        ax_f.set_ylim(*pair)

        cbar = fig.colorbar(hist[3], ax=ax_f)
        cbar.set_label('Frequency')

        # Energy plot formatting
        ax_f.set_xlabel('True Force / eV Å$^{-1}$')
        ax_f.set_ylabel('Predicted Force / eV Å$^{-1}$')

        ticks = np.linspace(np.round(min_f, 1), np.round(max_f, 1), 5)

        # If there are any negative force (components) then ensure 0 is
        # included in the tick list
        if any(val < 0 for val in ticks):
            ticks = ([val for val in ticks if val < 0]
                     + [0]
                     + [val for val in ticks if val > 0])

        for axis in (ax_f.xaxis, ax_f.yaxis):
            axis.set_ticks([np.round(val, 1) for val in ticks])

    return show_or_save(name)


def show_or_save(name):
    """If name is None then show the plot otherwise save it as a .png"""
    plt.tight_layout()

    if name is None:
        plt.show()

    else:
        plt.savefig(f'{name}.png', dpi=300)

    plt.close()
    return None


def fig_ax(energies, forces):
    """Get the appropriate axes for a set of energies and forces"""

    if energies is not None and forces is not None:
        size = (11, 4.5)
        cols = 2
        gridspec = {'width_ratios': [1, 1.2]}

    else:
        size = (4.5, 4.5)
        cols = 1
        gridspec = None

    return plt.subplots(nrows=1, ncols=cols, figsize=size,
                        gridspec_kw=gridspec)
