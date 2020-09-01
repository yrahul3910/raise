# `raise.interpret`

## `raise.interpret.ResultsInterpreter`

Working with the `Experiment` class is `raise.interpret.ResultsInterpreter`. This class accepts a list of file names in the constructor, and the `compare` method is called, which runs a Scott-Knott test for each metric.

## `raise.interpret.DODGEInterpreter`

Analogoys to `ResultsInterpreter` for the `Experiment` class is the `DODGEInterpreter` for the `DODGE` class. This take a list of files along with a set of column indices to exclude, and a maximizing function (`max_by`). This can be an integer, in which case the DODGE iteration with the best value for that column index is chosen, `None`, which results in setting the value to `0`, or a custom function for complex use cases.