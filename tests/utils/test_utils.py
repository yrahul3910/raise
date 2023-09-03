import sys
from io import StringIO
from colorama import Fore

from raise_utils.utils import info, warn, debug, error


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
