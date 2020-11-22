from keras.models import Model
from keras.layers import Dense
import numpy as np
from raise_utils.learners.learner import Learner


class Autoencoder(Learner):
    """
    A standard autoencoder architecture.
    """

    def __init__(self, n_layers: int = 2, n_units: list = [10, 10], n_out: int = 10, n_epochs=500, verbose=1, *args, **kwargs):
        """
        Initializes the autoencoder.

        :param n_layers: The number of *hidden* layers.
        :param n_units: The number of units in each hidden layer.
        :param n_out: The number of encoded dimensions.
        :param n_epochs: The number of epochs to train
        :param verbose: Whether training should be verbose
        """
        super().__init__(*args, **kwargs)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = 500
        self.n_out = n_out
        self.verbose = verbose
        self.loss = 'mse'

        self.learner = self
        self.random_map = {
            'n_layers': (1, 3),
            'n_out': (5, 11)
        }
        self._instantiate_random_vals()
        self._instantiate_model()

    def _instantiate_model(self):
        """
        Instantiates the autoencoder model according to the user parameters.
        """
        assert len(self.n_units) == self.n_layers

        enc_inp = Input((self.x_train.shape[1],))
        enc_intermediate = Dense(
            self.n_units[0], activation=self.activation)(enc_inp)
        for i in range(1, self.n_layers):
            enc_intermediate = Dense(
                self.n_units[i], activation=self.activation)(enc_intermediate)
        enc_intermediate = Dense(
            self.n_out, activation='relu')(enc_intermediate)

        dec_inp = Input((self.n_out,))
        dec_intermediate = Dense(
            self.n_units[-1], activation=self.activation)(dec_inp)
        for u in reversed(self.n_units[:-1]):
            dec_intermediate = Dense(
                u, activation=self.activation)(dec_intermediate)
        dec_intermediate = Dense(
            self.x_train.shape[1], activation='relu')(dec_intermediate)

        self.model = Model(inputs=enc_inp, outputs=dec_intermediate)
        self.encoder = Model(inputs=enc_inp, outputs=enc_intermediate)
        self.decoder = Model(inputs=dec_inp, outputs=dec_intermediate)

    def fit(self) -> None:
        """
        Fits the autoencoder.
        """
        self._check_data()

        if self.model is None:
            raise AssertionError('Model is None.')

        self.model.fit(self.x_train, self.x_train, epochs=self.n_epochs)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes an input.

        :param x - The input to be encoded
        """
        if self.model is None:
            raise AssertionError('Model is None.')

        return self.encoder(x)

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decodes an input from the encoded dimensionality.

        :param x - The encoded input.
        """
        if self.model is None:
            raise AssertionError('Model is None.')

        return self.decoder(x)
