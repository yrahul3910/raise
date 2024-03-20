import keras
from keras.models import Model
from keras.layers import Dense, Input
from keras.callbacks import EarlyStopping
import numpy as np
from raise_utils.learners.learner import Learner


class Autoencoder(Learner):
    """
    A standard autoencoder architecture.
    """

    def __init__(self, n_layers: int = 2, n_units: list = (10, 10), n_out: int = 10, n_epochs=500, activation='relu',
                 bs=128, verbose=1, *args, **kwargs):
        """
        Initializes the autoencoder.

        :param n_layers: The number of *hidden* layers.
        :param n_units: The number of units in each hidden layer.
        :param n_out: The number of encoded dimensions.
        :param n_epochs: The number of epochs to train
        :param bs: The batch size to use in training
        :param verbose: Whether training should be verbose
        """
        super().__init__(*args, **kwargs)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.n_out = n_out
        self.bs = bs
        self.verbose = verbose
        self.loss = 'mse'
        self.activation = activation

        self.learner = self
        self.random_map = {
            'n_layers': (1, 3),
            'n_out': (5, 11),
            'activation': ['relu', 'selu']
        }
        self._instantiate_random_vals()

    def set_data(self, x_train, y_train, x_test, y_test):
        super().set_data(x_train, y_train, x_test, y_test)
        self._instantiate_model()

    def _instantiate_model(self):
        """
        Instantiates the autoencoder model according to the user parameters.
        """
        assert len(self.n_units) == self.n_layers

        enc_inp = Input((self.x_train.shape[1],))
        layer_counter = 0

        enc_intermediate = Dense(
            self.n_units[0], name=f'layer_{layer_counter}', activation=self.activation)(enc_inp)
        layer_counter += 1

        for i in range(1, self.n_layers):
            enc_intermediate = Dense(
                self.n_units[i], name=f'layer_{layer_counter}', activation=self.activation)(enc_intermediate)
            layer_counter += 1

        enc_intermediate = Dense(
            self.n_out, name='encoded', activation='relu')(enc_intermediate)

        dec_intermediate = Dense(
            self.n_units[-1], name=f'layer_{layer_counter}', activation=self.activation)(enc_intermediate)
        layer_counter += 1

        for u in reversed(self.n_units[:-1]):
            dec_intermediate = Dense(
                u, name=f'layer_{layer_counter}', activation=self.activation)(dec_intermediate)
            layer_counter += 1

        dec_intermediate = Dense(
            self.x_train.shape[1], name='decoded', activation='relu')(dec_intermediate)

        self.model = Model(inputs=enc_inp, outputs=dec_intermediate)
        self.encoder = Model(inputs=enc_inp, outputs=enc_intermediate)

        dec_inp = Input(shape=(self.n_out,))
        for layer in self.model.layers[-1 - len(self.n_units):]:
            dec_intermediate_values = layer(
                dec_intermediate_values) if 'dec_intermediate_values' in locals() else layer(dec_inp)
        self.decoder = Model(inputs=dec_inp, outputs=dec_intermediate_values)

    def fit(self) -> None:
        """
        Fits the autoencoder.
        """
        self._check_data()

        if self.model is None:
            raise AssertionError('Model is None.')

        self.model.compile(loss='mse', optimizer='adam')
        early_stopping = EarlyStopping(monitor='loss', patience=10, min_delta=1e-3)
        self.model.fit(self.x_train, self.x_train,
                       epochs=self.n_epochs, batch_size=self.bs, callbacks=[early_stopping], verbose=self.verbose)
        self.learner = self.model

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encodes an input.

        :param x - The input to be encoded
        """
        if self.model is None:
            raise AssertionError('Model is None.')

        encoded = self.encoder(x)

        if keras.config.backend() == "torch":
            encoded = encoded.detach().numpy()

        return encoded

    def decode(self, x: np.ndarray) -> np.ndarray:
        """
        Decodes an input from the encoded dimensionality.

        :param x - The encoded input.
        """
        if self.model is None:
            raise AssertionError('Model is None.')

        return self.decoder(x)
