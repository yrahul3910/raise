# `raise.learners`

Learners form the core of the RAISE package, and are therefore designed flexibly. The base class for all learners is the `Learner` class; this is not meant to be used, since it does not by itself implement any learner; rather, it provides some basic functions used by all learners subclassing it.

Learners in RAISE are used slightly different from in `sklearn`. You start by creating an instance of the learner, passing it any desired parameters. Then, you call `set_data`, passing it the train and test splits; for conciseness, the `Data` class implements `__iter__`, allowing you to call `*data` to get the correct order of arguments to pass to `set_data`. Then, you call `fit` with no parameters, and later, `predict` can be called with the test set.

The `Learner` class implements some basic checks for the data and other cases, and raises exceptions accordingly, with error messages describing the issue. Learners also implement `__str__`, allowing you to call the `print` function to inspect the parameters.

A key feature of the learners is that they allow for random initialization, which is implemented to give the user as much control as possible. The simplest way to use this is to pass `random=True` to the learner while creating an instance (the default is `False`). The more advanced method is to pass in a `dict`, which is interpreted in the following way:

  * If a value is a tuple of (two) integers, a random integer between the two (both inclusive) is chosen.
  * If a value is a tuple of (two) floats, a random float between the two is chosen.
  * If a value is a list, then one of the list items is chosen randomly.

The keys of this `dict` must be members of the learner being used. The table below shows these members.

| **Learner**                    | **Parameters**                        |
| ------------------------------ | ------------------------------------- |
| `FeedforwardDL`                | `n_layers`, `n_units`, `weighted`     |
| `LogisticRegressionClassifier` | `penalty`, `C`                        |
| `NaiveBayes`                   | None                                  |
| `RandomForest`                 | `criterion`, `n_estimators`           |
| `SVM` and `BiasedSVM`          | `c`, `kernel`, `degree`, `k`, `sigma` |
| `DecisionTree`                 | `splitter`, `criterion`               |
| `TextDeepLearner`              | `max_words`, `max_len`, `n_layers`    |