# Change Log

## 2.2.0

* Fully moved to Keras 3--both TensorFlow and PyTorch backends are supported. JAX might work, but it is not officially
    supported.
* **Breaking change:** `raise_utils.learners.TextDeepLearner` has been removed.
* `KruskalWallis.pprint()` now returns a DataFrame of adjusted p-values, the best-performing group, and whether it is
    significantly different from the others.

## 2.1.4

* Bug fixes

## 2.1.3

* Bug fixes

## 2.1.2

* Bug fixes

## 2.1.0

* Added `KruskalWallis` class to `raise_utils.interpret`.
* Fixed installation issue.
* Add `EarlyStopping` to `FeedforwardDL`.
* Add `random` to `HPO` class.

## 2.0.3

* Useful debug info printed when data shape mismatch occurs.
* Add `HPO` class to add hyperopt and BOHB to `raise_utils.hyperparams`.
* Bug fix in `FeedforwardDL`.

## 2.0.2

* Bug fix: `wfo` transform can fail in some multi-class cases.
* Bug fix: `MulticlassGHOST` will now create a `log` directory if it does not already exist.
* Bug fix: Accuracy is now correctly computed for multi-class settings.
* Added `smooth` transform.

## 2.0.1

* Bug fix: Added one-hot encoding support to the `wfo` transform.

## 2.0.0

* **Breaking change:** `DODGE.optimize` now returns a tuple of median results and best configuration.
* `DODGE` now has a `predict` function.
* `BinaryGHOST` now has a `n_runs` parameter.
* `BinaryGHOST.fit()` now returns the same values as `DODGE`.
* `BinaryGHOST` now has a `predict` function.
* `BinaryGHOST` now has a `smote` parameter.

## 1.6.3

* Added G-1 score and IFA (initial false alarm) to metrics (thanks [@HuyTu7](https://github.com/HuyTu7)!)
* Added `GHOST` and `BinaryGHOST` to `raise_utils.hyperparams`

## 1.6.2

* Bug fix in `DODGE`
* Added `bs` option to `Autoencoder`
* Bug fix in `Autoencoder`
* SMOTE failing in `FeedforwardDL` now prints a warning instead of raising a `ValueError`.

## 1.6.1

* Bug fixes in `DODGE`
* `DODGE` does not print the learners anymore
* `DODGE` prints out the median performance every run
* New `n_iters` option added to `DODGEInterpreter`

## 1.6.0

* Added `bs` option to `FeedforwardDL`
* Bug fixes in Autoencoder
* FeedforwardDL now uses Early Stopping
* Added backward compatibility in `DODGEInterpreter` for older versions
* Bug fix in `DODGEInterpreter`
* Bug fixes in `ScottKnott`

## 1.5.1

* `Autoencoder` now uses Early Stopping.
* Bug fixes in import statements

## 1.5.0

* **Breaking change:** Renamed `raise_utils.transform` to `raise_utils.transforms`
* Added `GHOST` to `raise_utils.hyperparams`
* Relaxed `tensorflow` dependency to `>=2.0.0`
* Added `common_transforms` to `raise_utils.transforms`
* Added the `effect` option to `raise_utils.interpret.ScottKnott`.

## 1.4.0

* Bug fixes in `WeightedFuzzyOversampling`
* Added `raise_utils.interpret.ScottKnott`
* Relaxed TensorFlow version dependency
* Added many more test cases--and bug fixes along with them!

## 1.3.2

* Bug fixes in `DODGEInterpreter`
* Bug fixes in `Learner`
* Bug fixes in `DODGE`
* Bug fixes in `Transform`

## 1.3.1

* Bug fixes in `FeedforwardDL.predict(...)`
* Improved memory management

## 1.3.0

* Bug fixes
* Added the `RayTune` optimizer (Fixed [#10](https://github.com/yrahul3910/raise/issues/10))
* `DODGEInterpreter` now also reports best settings (Fixed [#11](https://github.com/yrahul3910/raise/issues/11))
* `DataLoader` now one-hot encodes multi-class targets (Fixed [#9](https://github.com/yrahul3910/raise/issues/9))
* Added Hooks support for `DataLoader`
* Added `Autoencoder` class to `raise_utils.learners`
