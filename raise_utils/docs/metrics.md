# `raise.metrics`

Metrics compute different metrics from classification metrics. `ClassificationMetrics`, a subclass of `Metric`, implements the functionality of this module. To get metric values, an instance of `ClassificationMetrics` is created, passing in the true and predicted labels (these are assumed to be binary). Then, one of `add_metric` or `add_metrics` is called, with strings denoting the desired metrics. The exception is for popt20: here, a call to `add_data` is required, which requires a `Data` instance formatted in a particular way. This instance can be obtained from an existing `Data` instance, by calling the `get_popt_data` method.

The following metrics are supported:

  * `accuracy`
  * `popt20`
  * `pf`
  * `pd` or `recall`
  * `auc`
  * `d2h`
  * `f1`
  * `prec`

