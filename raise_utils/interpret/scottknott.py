from raise_utils.interpret.sk import Rx


class ScottKnott:
    """A wrapper class around Rx for a simpler interface."""

    def __init__(self, data: dict):
        """
        Initializes the Scott-Knott class.

        :param {dict} data - A dictionary whose keys are the names of
        different methods to compare, and the values are results. The
        values must be list-like objects.
        """
        self.data = data

    def pprint(self):
        """Pretty-print the results."""
        Rx.show(Rx.sk(Rx.data(**self.data)))

    def get_results(self):
        """Gets the Scott-Knott results."""
        return Rx.sk(Rx.data(**self.data))
