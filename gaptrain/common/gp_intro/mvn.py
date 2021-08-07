import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
plt.style.use('paper')
plt.rcParams.update({
    "text.usetex": True,
})


def one_d_normal(x, sigma=1.0, mu=0.0):
    return 1.0/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x - mu)/sigma)**2)


def plot_1d(ax):

    xs = np.linspace(-4, 4, num=500)
    ax.plot(xs, one_d_normal(xs), color='blue')
    ax.set_xlim(-4, 4)
    ax.set_ylabel('PDF')
    ax.set_yticks([])

    return None


def plot_2d(ax, cov, add_cbar=False):

    cov = np.array(cov)
    if np.sum(np.abs(cov - cov.T)) > 1E-8:
        raise ValueError('Covariance matrix must be symmetric')

    x, y = np.mgrid[-4:4:.01, -4:4:.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean=[0.0, 0.0],
                             cov=cov)

    pdf1 = ax.contourf(x, y, rv.pdf(pos), cmap=plt.get_cmap('Blues'))

    if add_cbar:
        cbar1 = plt.colorbar(pdf1, ax=ax)
        cbar1.set_ticks([])
        cbar1.set_label('PDF')

    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])

    return None


if __name__ == '__main__':

    height = 2.3
    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3,
                                        figsize=(height*3.535, height))

    plot_1d(ax0)
    plot_2d(ax1, cov=np.eye(2))
    plot_2d(ax2,
            cov=[[1.0, 0.7],
                 [0.7, 1.0]],
            add_cbar=True)

    plt.tight_layout()
    plt.savefig('mvn.pdf')
