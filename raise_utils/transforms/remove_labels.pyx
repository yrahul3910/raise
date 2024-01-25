# cython: language_level=3
import numpy as np
import scipy.spatial.ckdtree as ckdtree
from scipy.stats import mode

def remove_labels(x_train, y_train):
    cdef int size, lost_idx_len, i
    cdef long[:] lost_idx
    cdef long[:, :] idx
    cdef int k

    lost_idx_len = int(len(y_train) - np.sqrt(len(y_train)))
    lost_idx = np.random.choice(len(y_train), size=lost_idx_len, replace=False).astype(int)

    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=int)

    x_lost = x_train[lost_idx]
    x_rest = np.delete(x_train, lost_idx, axis=0)
    y_lost = y_train[lost_idx]
    y_rest = np.delete(y_train, lost_idx, axis=0)

    if x_lost.shape[0] == 1:
        x_lost = x_lost.reshape(1, -1)
    if x_rest.shape[0] == 1:
        x_rest = x_rest.reshape(1, -1)

    cdef float[:, :] x_lost_view = x_lost
    cdef float[:, :] x_rest_view = x_rest
    cdef long[:] y_lost_view = y_lost
    cdef long[:] y_rest_view = y_rest

    tree = ckdtree.cKDTree(x_rest_view)
    k = int(np.sqrt(len(x_rest_view)))
    _, idx = tree.query(x_lost_view, k=k, p=1)
    y_lost_view = mode(y_rest[idx], axis=1)[0].reshape(-1)

    x_train_view = np.concatenate((x_lost_view, x_rest_view), axis=0)
    y_train_view = np.concatenate((y_lost_view, y_rest_view), axis=0)
    return x_train, y_train


class Smooth:
    def fit_transform(self, x_train, y_train):
        return remove_labels(x_train, y_train)

    def transform(self, x_test):
        raise NotImplementedError("transform should not be called on smooth")
