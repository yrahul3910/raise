from raise_utils.learners import FeedforwardDL, Autoencoder, Learner, MulticlassDL
from raise_utils.metrics import ClassificationMetrics
from raise_utils.data import Data
from raise_utils.transforms import Transform
from raise_utils.utils import warn, info
from hyperopt import hp, fmin, tpe
from keras.utils import to_categorical
from typing import Callable
from tabulate import tabulate
import numpy as np
import pandas as pd


class GHOST(Learner):
    """
    Implements the GHOST algorithm.
    """

    def __init__(self, obj_fn: Callable, metrics: list, n_classes: int = 2, ultrasample: bool = True,
                 autoencode: bool = True, tune: bool = True, ae_thresh: float = 1e3,
                 ae_layers: list = (10, 7), ae_out: int = 5, n_epochs: int = 100, hp_choices: dict = None,
                 max_evals: int = 30, bs=512, *args, **kwargs):
        """
        Initializes the GHOST algorithm. Several of these are internal parameters exposed for completeness.
        If you do not understand what a parameter does, the default value should work.

        :param metrics: A list of metrics supplied by raise_utils.metrics to print out.
        :param n_classes: The number of classes in the dataset.
        :param ultrasample: If True, perform ultrasampling.
        :param autoencode: If True, uses an autoencoder.
        :param tune: If True, runs hyper-parameter optimization
        :param ae_thresh: The threshold loss for the autoencoder.
        :param ae_layers: The number of units in each layer of the autoencoder.
        :param ae_out: The number of units the autoencoder outputs.
        :param n_epochs: The number of epochs to train for
        :param hp_choices: Hyper-parameter choice dict. It should be of the format:
            {
                'n_layers': list,
                'n_units': list,
                'wide_units': list  # no. of units for wide networks
            }
            A typical value for wide_units might be [64, 128, 256].
        :param obj_fn: A function that measures the performance of a model. It takes in y_true, y_preds
        (in that order), and should return a value where *lower* values are *better*.
        :param max_evals: The max number of hyper-parameter evaluations.
        :param bs: Batch size to use for feedforward learner.
        """
        super().__init__(*args, **kwargs)
        self.n_classes = n_classes
        self.metrics = metrics
        self.ultrasample = ultrasample
        self.autoencode = autoencode
        self.n_epochs = n_epochs
        self.tune = tune
        self.ae_thresh = ae_thresh
        self.ae_layers = ae_layers
        self.choices = hp_choices
        self.max_evals = max_evals
        self.ae_out = ae_out
        self.obj_fn = obj_fn
        self.bs = bs

    def set_data(self, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Overrides the set_data from Learner for multi-class data.
        """
        if self.n_classes == 2:
            super().set_data(x_train, y_train, x_test, y_test)
        else:
            self.x_train = x_train
            self.y_train = y_train
            self.x_test = x_test
            self.y_test = y_test

    def fit(self):
        self._check_data()

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.array(self.y_train).squeeze()
        self.y_test = np.array(self.y_test).squeeze()

        data = Data(self.x_train, self.x_test, self.y_train, self.y_test)
        data2 = Data(self.x_train, self.x_test, self.y_train, self.y_test)

        if self.n_classes != 2:
            data.y_train = np.argmax(data.y_train, axis=-1)
            data2.y_train = np.argmax(data2.y_train, axis=-1)

        if self.ultrasample:
            transform = Transform('wfo')
            transform.apply(data)
            transform.apply(data2)

            if not self.autoencode:
                warn(
                    '[GHOST] autoencode is False, but ultrasample is True. This is not the standard.')

            if self.autoencode:
                loss = 1 + self.ae_thresh
                while loss > self.ae_thresh:
                    ae = Autoencoder(n_layers=len(self.ae_layers),
                                     n_units=self.ae_layers, n_out=self.ae_out)

                    # We can't use set_data because it does unwanted things with multi-class systems.
                    ae.x_train = data.x_train
                    ae.x_test = data.x_test
                    ae.y_train = data.y_train
                    ae.y_test = data.y_test
                    ae._instantiate_model()

                    ae.fit()

                    loss = ae.model.history.history['loss'][-1]
                self.ae = ae

                data.x_train = ae.encode(np.array(data.x_train))
                data.x_test = ae.encode(np.array(data.x_test))

                data2.x_train = ae.encode(np.array(data2.x_train))
                data2.x_test = ae.encode(np.array(data2.x_test))

        if self.n_classes != 2:
            info('Running in multi-class mode.')
            data.y_train = to_categorical(data.y_train, num_classes=self.n_classes)
            data2.y_train = to_categorical(data2.y_train, num_classes=self.n_classes)

        # Set up hyperopt
        hp_space_dict = {}
        if self.choices is None:
            self.choices = {
                'n_units': list(range(data.x_train.shape[1] - 3, data.x_train.shape[1] + 5)),
                'n_layers': list(range(2, 6)),
                'wide_units': [64, 128, 256]
            }

        if self.choices.get('wide', False):
            hp_space_dict['n_layers'] = hp.choice(
                'n_layers', self.choices['wide_units'])

            hp_space = hp.choice('wide', [
                {
                    'n_layers': hp.choice('n_layers', self.choices['n_layers']),
                    'n_units': hp.choice('n_units', self.choices['n_units']),
                    'transform': hp.choice('transform', ['standardize', 'normalize', 'minmax'])
                },
                {
                    'n_layers': hp.choice('n_layers', self.choices['n_layers']),
                    'n_units': hp.choice('n_units', self.choices['wide_units']),
                    'transform': hp.choice('transform', ['standardize', 'normalize', 'minmax'])
                }
            ])
        else:
            hp_space = hp.choice('params', [{
                'n_layers': hp.choice('n_layers', self.choices['n_layers']),
                'n_units': hp.choice('n_units', self.choices['n_units']),
                'transform': hp.choice('transform', ['standardize', 'normalize', 'minmax'])
            }])

        results_table = []

        # Set up a hyperopt objective function
        def _objective(hp_args):
            n_units = hp_args['n_units']
            n_layers = hp_args['n_layers']
            trans = hp_args['transform']

            if self.n_classes == 2:
                learner = FeedforwardDL(
                    n_epochs=self.n_epochs, bs=self.bs, weighted=True, wfo=True, n_layers=n_layers, n_units=n_units, verbose=0)
            else:
                learner = MulticlassDL(
                    n_classes=self.n_classes, bs=self.bs, wfo=True, n_layers=n_layers, n_units=n_units, verbose=0)

            transform = Transform(trans)
            transform.apply(data2)

            learner.set_data(*data2)
            learner.fit()
            preds = learner.predict(data2.x_test)

            metr = ClassificationMetrics(data2.y_test, preds)
            metr.add_metrics(self.metrics)
            metrics = metr.get_metrics()

            obj = self.obj_fn(data2.y_test, preds)
            results_table.append([n_units, n_layers, obj, *metrics])

            return obj

        _ = fmin(_objective, hp_space, algo=tpe.suggest,
                    max_evals=self.max_evals)

        results_df = pd.DataFrame(results_table, columns=['n_units', 'n_layers', 'objective', *self.metrics])
        best_res = results_df[results_df['objective'] == results_df['objective'].min()].iloc[0,:]

        if self.n_classes == 2:
            self.best_model = FeedforwardDL(
                n_epochs=self.n_epochs, bs=self.bs, weighted=True, wfo=True, n_layers=int(best_res['n_layers']), n_units=int(best_res['n_units']), verbose=0)
        else:
            self.best_model = MulticlassDL(
                n_classes=self.n_classes, bs=self.bs, wfo=True, n_layers=int(best_res['n_layers']), n_units=int(best_res['n_units']), verbose=0)


        self.best_model.set_data(*data)
        self.best_model.fit()

        print(tabulate(results_table, headers=[
              'n_units', 'n_layers', 'objective', *self.metrics]))

    def predict(self, x_test):
        """
        Makes predictions on x_test.
        """
        x_test = np.array(x_test)
        if self.autoencode:
            x_test = self.ae.encode(x_test)
        return self.best_model.predict(x_test).squeeze()

