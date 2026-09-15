from importlib import import_module
from importlib.machinery import EXTENSION_SUFFIXES

import raise_utils as ru


def test_version():
    assert ru.__version__ != ""


def test_cython_extension():
    module = import_module("raise_utils.transforms.remove_labels")

    assert module.__file__.endswith(tuple(EXTENSION_SUFFIXES))
    assert module.Smooth.__module__ == module.__name__
    assert module.remove_labels.__module__ == module.__name__
