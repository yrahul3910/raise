from torch import nn, optim
import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from raise_utils.learners.learner import Learner
import numpy as np


def weighted_categorical_crossentropy(weights):
    def loss(y_pred, y_true):
        return F.cross_entropy(y_pred, y_true, weight=torch.from_numpy(weights).float())

    return loss


class FeedforwardDL(Learner):
    """
    A standard feed-forward neural network
    """

    def __init__(self, weighted=False, optimizer='adam', n_layers=3, n_units=19, activation='relu', n_epochs=10, *args, **kwargs):
        """
        Initializes the deep learner.

        Params:
        =======
        weighted: bool - Whether to use a weighted loss function
        optimizer: str - The choice of optimizer
        n_layers: int - Number of layers
        n_units: int - Number of units per layer
        activation: str - Activation function
        n_epochs: int - Number of epochs to train for
        *args, **kwargs - Args passed to Learner
        """
        super(FeedforwardDL, self).__init__(*args, **kwargs)

        self.activation = activation
        self.weighted = weighted
        self.optimizer = optimizer
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_epochs = n_epochs
        self.loss = 'crossentropy'

        self.learner = self
        self.model = nn.Sequential()
        self._instantiate_random_vals()

    def fit(self):
        self._check_data()

        self.x_train = np.array(self.x_train)
        self.y_train = np.array(self.y_train)
        self.x_test = np.array(self.x_test)
        self.y_test = np.array(self.y_test)

        train_ds = TensorDataset(torch.from_numpy(
            self.x_train), torch.from_numpy(self.y_train))
        valid_ds = TensorDataset(torch.from_numpy(
            self.x_test), torch.from_numpy(self.y_test))

        layers = []
        for i in range(self.n_layers):
            layers.append(
                nn.Linear(self.x_train.shape[1] if i == 0 else self.n_units, self.n_units))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(self.n_units, 2))
        layers.append(nn.Softmax())

        device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model = nn.Sequential(*layers)
        self.model.to(device)

        opt = optim.Adam(self.model.parameters())

        if self.weighted:
            frac = sum(self.y_train) * 1. / len(self.y_train)

            if not isinstance(self.weighted, float):
                self.weighted = 1.

            self.loss = weighted_categorical_crossentropy(
                np.array([1., self.weighted / frac]))
        else:
            self.loss = nn.CrossEntropyLoss()

        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
        valid_dl = DataLoader(valid_ds, batch_size=256, shuffle=True)

        for _ in range(self.n_epochs):
            train_loss = torch.zeros(1)
            for xb, yb in train_dl:
                xb.to(device)
                yb.to(device)

                preds = torch.squeeze(self.model(xb.float()))
                loss = self.loss(preds, yb)
                train_loss += loss
                loss.backward()

                opt.step()
                opt.zero_grad()

            with torch.no_grad():
                valid_loss = torch.zeros_like(loss)
                for xb, yb in valid_dl:
                    xb.to(device)
                    yb.to(device)

                    preds = torch.squeeze(self.model(xb.float()))
                    valid_loss += self.loss(preds, yb)

                print(
                    f'Epoch {_}: train_loss = {train_loss.item()} val_loss = {valid_loss.item()}')

    def predict(self, x_test):
        """
        Makes predictions.
        """
        test = torch.from_numpy(np.array(x_test)).float()
        return torch.argmax(self.model(test), dim=1).numpy()
