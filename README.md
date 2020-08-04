

<p align="center">
<img src="http://ai4se.net/img/logo.png" height="80px" /><br />
<a href="http://ai4se.net">homepage</a>  | 
<a href="https://github.com/yrahul3910/raise/tree/master/docs">docs</a>  |
<a href="https://github.com/yrahul3910/raise/blob/master/LICENSE">license</a>  |
<a href="https://github.com/yrahul3910/raise/blob/master/CODE_OF_CONDUCT.md">contribute</a>  |
<a href="https://github.com/yrahul3910/raise/issues/">issues</a>  ::
<a href="mailto:timm@ieee.org">contact</a>
</p>
<p align="center">
<img src="https://img.shields.io/badge/language-python-orange.svg">&nbsp;
<img src="https://img.shields.io/badge/license-MIT-green.svg">&nbsp;
<img src="https://img.shields.io/badge/platform-mac,*nux-informational">&nbsp;
<img src="https://img.shields.io/badge/purpose-ai,se-blueviolet">&nbsp;
<img src="https://app.codacy.com/project/badge/Grade/8352fafd16454ea995f43891d9571d22" />&nbsp;
</p> <hr />

# The RAISE package

The RAISE package is an attempt to unify code, and incorporate PEP8 standards. The package takes a modular, object-oriented design, with each part of the ML pipeline encapsulated in a class. The purpose is to allow a streamlined, easy to read interface that allows for explainability of code, to enforce programming standards, and improve maintainability.

# Install

Run `setup.py` to install the package.

# Structure

## `raise.data`

This package provides the `Data` and `DataLoader` classes. The `Data` class is not to be used; use the `DataLoader` to obtain a `Data` object instead. 

`DataLoader` provides two methods: `from_file` and `from_files`: the first is used when one single CSV file contains the entire data; the second is used when a group of files provide the data, and the last file is used as the test set.

## `raise.learners`

Encapsulates different learners, with an option for random initialization. These classes may be used independently, or passed to `raise.experiments.Experiment` for a more automated setup. Each learner offers a `fit` and `predict` function, but `fit` does not take any arguments. Instead, the data is set using the `set_data` function.

The `raise.learners.FeedforwardDL` class implements a feed-forward neural network. It offers options to use a weighted loss function, weighted fuzzy oversampling, and change the optimizer, number of layers, number of units per layer, number of epochs, and the activation function used. Except the first two, these arguments must be recognized by `keras`.

## `raise.metrics`

Provides an object to compute multiple metrics for a set of predictions. The base class `Metric` should not be used directly; instead, use `ClassificationMetrics`. This class allows adding of single or multiple metrics as strings, and provides a `get_metrics()` function that returns a list of metrics in the same order as provided. If the popt20 metric is desired, a call to `set_data` is also required before calling `get_metrics()`, with the full training set (including the targets).

## `raise.transform`

Implements various data transforms, with an option for random arguments. This class may be used independently, but is also used to `raise.experiments.Experiment`, where transforms are applied in order.

## `raise.experiments`

Provides the `Experiment` class, which allows for a combination of the above, with logging to a file. Experiments can have their own name, which forms the filenames. If not provided, a random one is used. The constructor must be provided a dict in the following format:

```
{
	"n_runs": int,
	"transforms": list[str],
	"metrics": list[str],
  "random": bool,
  "learners": list[raise.learners.Learner],
  "log_path": str,
  "data": list[raise.data.Data],
  "name": str
}
```

## `raise.hyperparams`

Implements hyper-parameter optimizers from the RAISE lab; currently only has DODGE. Hyper-parameter optimizers expect a config file with the same format as `raise.experiments.Experiment`, and then can be run using `optimize()`.



