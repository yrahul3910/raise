import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import gamma


def get_smape(P, Q):
    """
    Returns the Symmetric Mean Absolute Percentage Error (SMAPE) of distributions of P and Q.

    :param P: Distribution P
    :param Q: Distribution Q
    """
    total = 0.
    for x in P:
        for y in Q:
            total += 1./ len(x) * np.sum(np.abs(y - x) / (np.abs(x) + np.abs(y) + np.finfo(float).eps))

    return total / (len(P) * len(Q))


def get_kdes(A, F, grid_size=100, k=3):
    """
    Based on https://faculty.washington.edu/yenchic/18W_425/Lec7_knn_basis.pdf
    """
    d = A.shape[1]
    n = A.shape[0]

    Vd = np.pi ** (d / 2) / gamma(d / 2 + 1)

    mins, maxes = np.min([A.min(axis=0), F.min(axis=0)], axis=0), np.max([A.max(axis=0), F.max(axis=0)], axis=0)

    # Generate grid_size points in the range mins to maxes
    grid = np.array([np.linspace(mins[i], maxes[i], grid_size) for i in range(d)]).T
    assert grid.shape == (grid_size, d)

    # Compute KDE for A
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(A, np.zeros(A.shape[0]))
    kde_A = []
    for point in grid:
        dist, _ = knn.kneighbors(point.reshape(1, -1), return_distance=True)
        kde_A.append(k / (n * Vd * dist[0][-1]))

    kde_A = np.array(kde_A)
    assert kde_A.shape == (grid_size,)

    # Compute KDE for F
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(F, np.zeros(F.shape[0]))
    kde_F = []
    for point in grid:
        dist, _ = knn.kneighbors(point.reshape(1, -1), return_distance=True)
        kde_F.append(k / (n * Vd * dist[0][-1]))

    kde_F = np.array(kde_F)
    assert kde_F.shape == (grid_size,)

    return kde_A, kde_F


def get_kullback_leibler(P, Q, grid_size=100, k=3):
    """
    Returns the Kullback-Leibler divergence of distributions of P and Q.

    `grid_size` and `k` are used in the KDE.

    :param P: Distribution P
    :param Q: Reference distribution Q
    :param grid_size: Number of points in the grid
    :param k: Number of nearest neighbors to consider
    """
    kl = 0.

    kde_P, kde_Q = get_kdes(P, Q, grid_size=grid_size, k=k)
    kl = np.sum(kde_P * np.log2(kde_P / kde_Q))

    return kl


def get_jensen_shannon(P, Q, grid_size=100, k=3):
    """
    Returns the Jensen-Shannon divergence of distributions of P and Q.

    `grid_size` and `k` are used in the KDE.

    :param P: Distribution P
    :param Q: Distribution Q
    :param grid_size: Number of points in the grid
    :param k: Number of nearest neighbors to consider
    """
    jsd = 0.

    kde_P, kde_Q = get_kdes(P, Q, grid_size=grid_size, k=k)
    
    # First, compute M
    M = (kde_P + kde_Q) / 2

    # Now, compute JSD using KL-divergences
    jsd = 0.5 * (np.sum(kde_P * np.log2(kde_P / M)) + np.sum(kde_Q * np.log2(kde_Q / M)))
    return jsd
