from raise_utils.transforms import Transform
from raise_utils.hyperparams import DODGE
from raise_utils.learners import FeedforwardDL, Autoencoder
import numpy as np


class GHOST:
    """
    Implements the GHOST algorithm (Yedida & Menzies, 2021). The ablation study from the paper can be performed using this class as well.

    Reference: https://arxiv.org/abs/2008.03835
    """

    def __init__(self, data, metrics: list, log_path: str = './log/', name: str = 'ghost',
                 n_learners: int = 30, weighted: bool = True, wfo: bool = True, smote: bool = True,
                 ultrasample: bool = True, **ae_kws):
        """
        Initializes the configuration.

        :param {Data} data - A Data object.
        :param {list} metrics - A list of metrics. GHOST optimizes for the first metric.
        :param {str} log_path - The log path.
        :param {str} name - The name of the log file.
        :param {int} n_learners - The number of learners GHOST should use.
        :param {bool} weighted - Whether to use weighted loss functions.
        :param {bool} wfo - Whether to use weighted fuzzy oversampling.
        :param {bool} smote - Whether to use SMOTE.
        :param {bool} ultrasample - Whether to use ultrasampling
        :param **ae_kws - Keywords passed to the Autoencoder object. By default,
        sets `n_layers=2, n_units=[10, 7], n_out=5`.
        """
        self.data = data
        self.metrics = metrics
        self.weighted = weighted
        self.wfo = wfo
        self.ultrasample = ultrasample

        if ae_kws == {}:
            ae_kws = {
                'n_layers': 2,
                'n_units': [10, 7],
                'n_out': 5
            }

        self.ae_kws = ae_kws

    def optimize(self):
        if self.ultrasample:
            transform = Transform('wfo')
            transform.apply(self.data)

            # Reverse labels
            self.data.y_train = 1. - self.data.y_train
            self.data.y_test = 1. - self.data.y_test

            # Autoencode the inputs
            loss = 1e4
            while loss > 1e3:
                ae = Autoencoder(**self.ae_kws)
                ae.set_data(*self.data)
                ae.fit()

                loss = ae.model.history.history['loss'][-1]

            self.data.x_train = ae.encode(np.array(self.data.x_train))
            self.data.x_test = ae.encode(np.array(self.data.x_test))

        dodge_config = {
            'n_runs': 100,
            'data': [self.data],
            'metrics': self.metrics,
            'learners': [FeedforwardDL(weighted=self.weighted, wfo=self.wfo, smote=self.smote,
                                       random={'n_units': (
                                           2, 6), 'n_layers': (2, 5)},
                                       n_epochs=50) for _ in range(self.n_learners)],
            'log_path': self.log_path,
            'random': True,
            'name': self.name
        }

        dodge = DODGE(dodge_config)
        dodge.optimize()
