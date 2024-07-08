import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.special import gamma
from scipy.special import rel_entr

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

def get_smape_vectorized(A: pd.DataFrame, F: pd.DataFrame) -> float:
    """
    Returns the Symmetric Mean Absolute Percentage Error (SMAPE) of distributions of P and Q.
    Works faster than the previous method.

    :param A: Distribution A
    :param F: Distribution F
    """
    x = A.values[:, np.newaxis, :]
    y = F.values[np.newaxis, :, :]
    numerator = 2 * np.abs(y - x)
    denominator = np.abs(x) + np.abs(y) + np.finfo(float).eps
    result = np.mean(numerator / np.maximum(denominator, np.finfo(float).eps))    
    return result

def get_smape_vectorized_chunked(A: pd.DataFrame, F: pd.DataFrame, chunk_size: int = 1000) -> float:
    """
    Returns the Symmetric Mean Absolute Percentage Error (SMAPE) of distributions of P and Q.
    Balanced the memmory usage of previous method.

    :param A: Distribution A
    :param F: Distribution F
    :param chunk_size: Number of rows SMAPE is calculated for them each step
    """
    total_sum = 0
    total_count = 0
    
    for i in range(0, len(A), chunk_size):
        A_chunk = A.iloc[i:i+chunk_size]
        
        for j in range(0, len(F), chunk_size):
            F_chunk = F.iloc[j:j+chunk_size]
            
            x = A_chunk.values[:, np.newaxis, :]
            y = F_chunk.values[np.newaxis, :, :]
            
            numerator = 2 * np.abs(y - x)
            denominator = np.abs(x) + np.abs(y)
            
            chunk_result = numerator / np.maximum(denominator, np.finfo(float).eps)
            total_sum += np.sum(chunk_result)
            total_count += chunk_result.size
    
    result = (total_sum / total_count)
    return result


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


def js_divergence(p, q):
    """
    Compute the Jensen-Shannon divergence between two probability distributions.

    :param p: Distribution p
    :param q: Distribution q
    """
    # Ensure the distributions are probability distributions
    p = np.array(p)
    q = np.array(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    # Calculate the pointwise mean
    m = 0.5 * (p + q)
    
    # Compute the JSD
    jsd = 0.5 * ( np.sum(rel_entr(p, m)) + np.sum(rel_entr(q, m)) )
    return jsd

def normalize_histogram(H):
    """
    Normalize a histogram to form a probability distribution.

    :param H: Histogram H
    """
    H = np.array(H)
    return H / np.sum(H)

def compute_histograms(P, bins=10):
    """
    Compute normalized histograms for each column of the dataset.

    :param P: Dataset P
    :param bins: Number of bins in calculating histogram
    """
    P = np.array(P)
    histograms = []
    for col in range(P.shape[1]):
        hist, bin_edges = np.histogram(P[:, col], bins=bins, density=False)
        hist = normalize_histogram(hist)
        histograms.append(hist)
    return histograms

def get_jsd_for_multidimensional_data(P, Q, bins=10):
    """
    Compute the Jensen-Shannon divergence for multidimensional dataframes by Summing JSDs for each column.

    `bins` is used for histogram calculation.

    :param P: Dataset P
    :param Q: Dataset Q
    :param bins: Number of bins in calculating histogram
    """
    # Compute histograms for each dataset
    histograms_P = compute_histograms(P, bins)
    histograms_Q = compute_histograms(Q, bins)
    
    # Ensure both datasets have histograms for the same number of columns
    assert len(histograms_P) == len(histograms_Q), "Datasets must have the same number of columns"
    
    # Compute JSD for each column
    jsd_values = []
    for h1, h2 in zip(histograms_P, histograms_Q):
        jsd = js_divergence(h1, h2)
        jsd_values.append(jsd)
    
    # Sum of the JSD values
    sum_jsd = np.sum(jsd_values)
    return sum_jsd