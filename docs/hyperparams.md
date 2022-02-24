# `raise_utils.hyperparams`

Implements hyperparameter optimizers. To use DODGE, `DODGE` instead is created, passing in a configuration dictionary, and then the `optimize` function is called. The structure of the dictionary is below:

```python
{
  "n_runs": int,
  "transforms": list[str],
  "metrics": list[str],
  "random": Union[bool, dict],
  "learners": list[Learner],
  "log_path": str,
  "data": list[Data],
  "n_iters": int,
  "post_train_hooks": list[Hook],
  "name": str
}
```

Some of these config parameters have default values. If `log_path` is `None`, then `DODGE` will write to `stdout`. `post_train_hooks` defaults to `None`, and is passed the model (at the current iteration), and the test data (`x_test` and `y_test`). `n_runs` defaults to 1, and `n_iters` (number of DODGE iterations) defaults to 30.

`BinaryGHOST` builds upon `DODGE`, and takes a list of metrics as input. This list is passed to `DODGE`, and so the first one is the metric that is optimized for.

`GHOST` takes an `obj_fn` (objective function), giving you flexibility in defining a metric or loss function. It also takes in a list of metrics, which are printed out in each iteration.