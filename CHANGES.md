# Change Log

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
