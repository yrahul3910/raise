from importlib import import_module

import numpy as np
import pytest
import torch

from raise_utils.data import Data
from raise_utils.hyperparams import DODGE
from raise_utils.learners import Autoencoder, FeedforwardDL, RandomForest
from raise_utils.transforms import Transform


@pytest.fixture(params=["cpu", "cuda", "mps"])
def tensor_data(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal hardware is unavailable")
    features = torch.arange(24, dtype=torch.float32, device=device).reshape(12, 2)
    labels = np.arange(12) % 2
    return Data(features.requires_grad_(), features.clone(), labels, labels.copy())


def test_autoencoder_encodes_tensor_inputs(tensor_data):
    keras = import_module("keras")
    with keras.device(tensor_data.x_train.device.type):
        learner = Autoencoder(n_layers=1, n_units=[4], n_out=2, n_epochs=1, verbose=0)
        learner.set_data(*tensor_data)
        encoded = learner.encode(tensor_data.x_test)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (12, 2)
    assert np.isfinite(encoded).all()


def test_feedforward_trains_with_tensor_inputs(tensor_data):
    learner = FeedforwardDL(n_layers=1, n_units=4, n_epochs=1, verbose=0)
    learner.set_data(*tensor_data)
    learner.fit()

    assert isinstance(learner.x_train, np.ndarray)
    assert isinstance(learner.x_test, np.ndarray)
    assert np.isfinite(learner.model.history.history["loss"]).all()


def test_transform_accepts_tensor_inputs(tensor_data):
    tensor_data.x_test = tensor_data.x_test.cpu()
    train_device = tensor_data.x_train.device
    test_device = tensor_data.x_test.device
    Transform("standardize").apply(tensor_data)

    assert tensor_data.x_train.device == train_device
    assert tensor_data.x_test.device == test_device
    assert torch.isfinite(tensor_data.x_train).all()
    assert torch.isfinite(tensor_data.x_test).all()
    torch.testing.assert_close(tensor_data.x_train.mean(dim=0), torch.zeros(2, device=train_device), atol=1e-6, rtol=0)


def test_dodge_accepts_tensor_inputs(tensor_data, tmp_path):
    optimizer = DODGE(
        {
            "data": [tensor_data],
            "learners": [RandomForest()],
            "transforms": ["standardize"],
            "metrics": ["accuracy"],
            "n_iters": 1,
            "n_runs": 1,
            "log_path": str(tmp_path),
            "name": "tensor-inputs",
        }
    )
    scores, _ = optimizer.optimize()

    assert np.isfinite(scores).all()
