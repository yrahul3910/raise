from raise_utils.learners.learner import Learner
import numpy as np
import cvxopt
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers


# from https://pythonprogramming.net/soft-margin-kernel-cvxopt-svm-machine-learning-tutorial/
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(p=3):
    def kernel(x, y, p=p):
        return (1 + np.dot(x, y)) ** p

    return kernel


def gaussian_kernel(sigma=5.0):
    def kernel(x, y, s=sigma):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (s ** 2)))

    return kernel


def solve_dual_problem(X, y, k=.1, kernel=linear_kernel, C=1.):
    m, _ = X.shape

    idx = np.where(y == -1)[0]
    q = -np.ones((m, 1))
    q[idx] = -k
    q = cvxopt_matrix(q)

    y = np.array(y).reshape(1, -1) * 1.

    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = kernel(X[i], X[j])

    P = cvxopt.matrix(np.outer(y, y) * K)
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))

    sol = cvxopt_solvers.qp(P, q, G, h, A, b)
    lambd = np.array(sol['x'])
    y = y.T
    w = ((y * lambd).T @ X).T

    ind = np.where(y == -1)[0]
    idx = np.where(y == 1)[0]
    p1 = -k - X[ind] @ w
    p2 = 1. - X[idx] @ w

    b = (np.max(p1) + np.min(p2)) / 2

    return w, b


class BiasedSVM(Learner):
    """
    A biased SVM learner. Implements a modified version of the SVM algorithm, using the dual
    problem solution set up above.
    """

    def __init__(self, c=1., kernel="rbf", degree=2, k=.1, sigma=3., *args, **kwargs):
        """Initializes the classifier"""
        super(BiasedSVM, self).__init__(*args, **kwargs)

        self.c = c
        self.kernel = kernel
        self.degree = degree
        self.k = k
        self.sigma = sigma

        self.learner = self
        self.random_map = {
            "c": [.01, .1, 1., 10., 100., 1000.],
            "kernel": ["poly", "rbf", "linear"],
            "degree": (2, 5),
            "sigma": (0., 5.),
            "k": (0., 1.)
        }
        self._instantiate_random_vals()

        self.w, self.b = None, None
        self.failed = False

    def _fit(self):
        """
        Solves the modified dual optimization problem

        :return: None
        """
        kernel_map = {
            "poly": polynomial_kernel(self.degree),
            "rbf": gaussian_kernel(self.sigma),
            "linear": linear_kernel
        }
        self.y_train = self.y_train.apply(lambda x: -1 if x == 0 else 1)
        self.w, self.b = solve_dual_problem(np.array(self.x_train), np.array(self.y_train),
                                            k=self.k,
                                            kernel=kernel_map[self.kernel])

    def fit(self) -> None:
        """Wrapper around _fit with a timeout."""
        self._fit()

    def predict(self, x_test):
        """
        Makes a prediction based on the solution of the dual optimization problem.

        :param x_test: Test data
        :return: Labels
        """
        if self.w is not None and self.b is not None:
            return ((np.array(x_test) @ self.w + self.b) > self.k).astype('int').squeeze()
        elif self.failed:
            return np.array([0.] * len(x_test))
        else:
            raise AssertionError("Learner not fit before predicting.")


class SVM(BiasedSVM):
    """The standard Support Vector Machine learner"""

    def __init__(self, *args, **kwargs):
        """Initializes the classifier."""
        super(SVM, self).__init__(*args, **kwargs)
        self.k = 1.
