import sys
from io import StringIO

import numpy as np
import pytest

from raise_utils.data import Data
from raise_utils.utils import info, warn, debug, error, _check_data


def test_check_data_checks_shapes_correctly():
    custom_output = StringIO()
    sys.stdout = custom_output

    data = Data(np.ones((10, 10)), np.ones((10, 5)), np.ones((5,)), np.ones((5)))

    with pytest.raises(AssertionError):
        _check_data(data)

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert output_content == "x_train: (10, 10)\n" \
        "y_train: (5,)\n" \
        "x_test: (10, 5)\n" \
        "y_test: (5,)\n"


def test_check_data_on_non2d_data():
    data = Data(np.ones((10, 10, 10)), np.ones((10,)), np.ones((5, 5)), np.ones((5)))

    with pytest.raises(AssertionError):
        _check_data(data)


def test_info():
    custom_output = StringIO()
    sys.stdout = custom_output

    info("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert "[INFO] test" in output_content


def test_warn():
    custom_output = StringIO()
    sys.stdout = custom_output

    warn("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert "[WARN] test" in output_content


def test_debug():
    custom_output = StringIO()
    sys.stdout = custom_output

    debug("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert "[DEBUG] test" in output_content


def test_error():
    custom_output = StringIO()
    sys.stdout = custom_output

    error("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert "[ERR] test" in output_content
