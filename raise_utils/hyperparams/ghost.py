import os
from raise_utils.learners import FeedforwardDL, Autoencoder, Learner
from raise_utils.data import Data
from raise_utils.transforms import Transform
from raise_utils.hyperparams import DODGE
from raise_utils.utils import warn
import numpy as np


class BinaryGHOST(Learner):
    """
    Implements the original, 2-class GHOST algorithm.
    """

    def __init__(self, metrics: list, ultrasample: bool = True, smote: bool = True,
                 autoencode: bool = True, ae_thresh: float = 1e3, n_runs: int = 20,
                 ae_layers: list = (10, 7), ae_out: int = 5, n_epochs: int = 50,
                 max_evals: int = 30, bs=512, name='experiment', *args, **kwargs):
        """
        Initializes the GHOST algorithm. Several of these are internal parameters exposed for completeness.
        If you do not understand what a parameter does, the default value should work.

        :param metrics: A list of metrics supplied by raise_utils.metrics to print out.
        :param ultrasample: If True, perform ultrasampling.
        :param smote: If True, perform SMOTE.
        :param autoencode: If True, uses an autoencoder.
        :param ae_thresh: The threshold loss for the autoencoder.
        :param ae_layers: The number of units in each layer of the autoencoder.
        :param ae_out: The number of units the autoencoder outputs.
        :param n_runs: Number of times to run DODGE.
        :param n_epochs: The number of epochs to train for
        :param max_evals: The max number of hyper-parameter evaluations.
        :param bs: Batch size to use for feedforward learner.
        :param name: A name for the DODGE runs.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.metrics = metrics
        self.ultrasample = ultrasample
        self.smote = smote
        self.autoencode = autoencode
        self.n_epochs = n_epochs
        self.n_runs = n_runs
        self.ae_thresh = ae_thresh
        self.ae_layers = ae_layers
        self.max_evals = max_evals
        self.ae_out = ae_out
        self.bs = bs
        self.dodge = None

    def fit(self):
        self._check_data()

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train).squeeze()
        self.y_test = np.array(self.y_test).squeeze()

        data = Data(self.x_train, self.x_test, self.y_train, self.y_test)

        if self.ultrasample:
            transform = Transform('wfo')
            transform.apply(data)

            data.y_train = 1. - data.y_train
            data.y_test = 1. - data.y_test

            if not self.autoencode:
                warn(
                    '[GHOST] autoencode is False, but ultrasample is True. This is not the standard.')

            if self.autoencode:
                loss = 1 + self.ae_thresh
                tries = 0
                while loss > self.ae_thresh and tries < 3:
                    self.ae = Autoencoder(n_layers=len(self.ae_layers),
                                     n_units=self.ae_layers, n_out=self.ae_out)

                    # We can't use set_data because it does unwanted things with multi-class systems.
                    self.ae.set_data(*data)
                    self.ae.fit()

                    loss = self.ae.model.history.history['loss'][-1]
                    tries += 1

                if tries <= 3:
                    data.x_train = self.ae.encode(np.array(data.x_train))
                    data.x_test = self.ae.encode(np.array(data.x_test))

        if not os.path.exists('log/'):
            os.mkdir('./log/')

        dodge_config = {
            'n_runs': self.n_runs,
            'n_iters': self.max_evals,
            'data': [data],
            'metrics': self.metrics,
            'learners': [],
            'log_path': './log/',
            'transforms': ['standardize', 'normalize', 'minmax'] * 30,
            'random': True,
            'name': self.name
        }

        for _ in range(30):
            dodge_config['learners'].append(
                FeedforwardDL(weighted=True, wfo=True, smote=self.smote,
                              random={'n_units': (2, 6), 'n_layers': (2, 5)},
                              n_epochs=self.n_epochs, verbose=0)
            )

        self.dodge = DODGE(dodge_config)
        return self.dodge.optimize()

    def predict(self, x_test):
        """
        Makes predictions on x_test.
        """
        if self.dodge is None:
            raise AssertionError('fit must be called before predict.')

        if self.autoencode:
            x_test = self.ae.encode(x_test)

        return self.dodge.predict(x_test)
