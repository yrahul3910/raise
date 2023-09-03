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

    assert output_content == Fore.GREEN + "[INFO] " + Fore.RESET + Fore.GREEN + "test"


def test_warn():
    custom_output = StringIO()
    sys.stdout = custom_output

    warn("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert output_content == Fore.YELLOW + "[INFO] " + Fore.RESET + Fore.YELLOW + "test"


def test_debug():
    custom_output = StringIO()
    sys.stdout = custom_output

    debug("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert output_content == Fore.BLUE + "[INFO] " + Fore.RESET + Fore.BLUE + "test"


def test_error():
    custom_output = StringIO()
    sys.stdout = custom_output

    error("test")

    output_content = custom_output.getvalue()
    sys.stdout = sys.__stdout__

    assert output_content == Fore.RED + "[INFO] " + Fore.RESET + Fore.RED + "test"
