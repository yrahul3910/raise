from raise_utils.interpret.sk import Rx


class ScottKnott:
    """A wrapper class around Rx for a simpler interface."""

    def __init__(self, data: dict, effect='small'):
        """
        Initializes the Scott-Knott class.

        :param {dict} data - A dictionary whose keys are the names of
        different methods to compare, and the values are results. The
        values must be list-like objects.
                :param {str|float} effect - The effect size to use for Cliff's delta.
                Can be passed a floating point value, or one of {"small", "medium", "large"}
                for predefined values from Hess & Kromney (2004).

                References:
                ===========
                Hess, Melinda R., and Jeffrey D. Kromrey. "Robust confidence intervals for effect 
                sizes: A comparative study of Cohens'd and Cliff's delta under non-normality and 
                heterogeneous variances." annual meeting of the American Educational Research 
                Association. 2004.
        """
        self.data = data
        self.effect = effect

    def pprint(self):
        """Pretty-print the results."""
        Rx.show(Rx.sk(Rx.data(**self.data), effect=self.effect))

    def get_results(self):
        """Gets the Scott-Knott results."""
        return Rx.sk(Rx.data(**self.data), effect=self.effect)
