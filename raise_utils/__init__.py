"""Utilities and learning algorithms from the RAISE lab."""

import os

# Keras selects its backend when the first submodule imports it.
os.environ.setdefault("KERAS_BACKEND", "torch")

__version__ = "3.0.0"
