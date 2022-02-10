# `raise_utils.metrics`

Metrics compute different metrics from classification metrics. `ClassificationMetrics`, a subclass of `Metric`, implements the functionality of this module. To get metric values, an instance of `ClassificationMetrics` is created, passing in the true and predicted labels. Then, one of `add_metric` or `add_metrics` is called, with strings denoting the desired metrics. The exception is for popt20: here, a call to `add_data` is required, which requires a `Data` instance formatted in a particular way. This instance can be obtained from an existing `Data` instance, by calling the `get_popt_data` method.

The following metrics are supported:
 
* `accuracy`   
* `popt20`  
* `pf`  (false alarm rate)
* `pd` or `recall`  
* `auc`  
* `d2h`  (d2h, but with reduced emphasis on pf)
* `d2h2` (the original d2h)
* `f1`  
* `prec`
* `pd-pf` (recall - false alarm rate)
* `conf` (the confusion matrix)
