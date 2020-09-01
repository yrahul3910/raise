# `raise.experiments`

This module allows for a slightly automated pipeline. An instance of `Experiment` is created, passing in a configuration dictionary, and the `run` method is called. The structure of the dictionary is below (comments show the default values:

```python
{
  "runs": int,  # 20
  "transforms": list[str]  # []
  "metrics": list[str]  # ["accuracy"]
  "random": Union[bool, dict]  # False
  "learners": list[Learner]  # None; will raise KeyError if not passed
  "log_path": str,  # ./log/;  path must exist
  "data": list[Data]  # None; raises KeyError if not present
  "name": str  # random string; name of the experiment
}
```

A few things to note here:
  
* The choice of `random` here does not affect the `random` parameter in the learners.  
* The `data` value is a list of `Data`; these are concatenated together.  
