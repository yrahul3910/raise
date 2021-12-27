# Change Log

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
