from colorama import init, Fore


def info(string: str):
    """
    Prints a information message to the terminal.
    :param {str} string - The message to print.
    """
    init()
    pre = Fore.GREEN + '[INFO] ' + Fore.RESET
    print(pre + string.replace('\n', '\n' + pre))


def warn(string: str):
    """
    Prints a warning message to the terminal.
    :param {str} string - The message to print.
    """
    init()
    pre = Fore.YELLOW + '[WARN] ' + Fore.RESET
    print(pre + string.replace('\n', '\n' + pre))


def debug(string: str):
    """
    Prints a debug message to the terminal.
    :param {str} string - The message to print.
    """
    init()
    pre = Fore.BLUE + '[DEBUG] ' + Fore.RESET
    print(pre + string.replace('\n', '\n' + pre))


def error(string: str):
    """
    Prints an error message to the terminal.
    :param {str} string - The message to print.
    """
    init()
    pre = Fore.RED + '[ERR] ' + Fore.RESET
    print(pre + string.replace('\n', '\n' + pre))