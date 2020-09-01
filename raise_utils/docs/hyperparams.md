# `raise.hyperparams`

Implements hyperparameter optimizers. For now, only DODGE is supported. A `DODGE` instead is created, passing in a configuration dictionary, and then the `optimize` function is called. The structure of the dictionary is below:

```python
{
  "n_runs": int,
  "transforms": list[str],
  "metrics": list[str],
  "random": Union[bool, dict],
  "learners": list[Learner],
  "log_path": str,
  "data": list[Data],
  "name": str
}
```
