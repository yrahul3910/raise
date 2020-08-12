import numpy as np


def fuzz_data(X, y, radii=(0., .3, .03)):
    idx = np.where(y == 1)[0]
    frac = len(idx) * 1. / len(y)

    fuzzed_x = []
    fuzzed_y = []

    for row in X[idx]:
        for i, r in enumerate(np.arange(*radii)):
            for j in range(int((1. / frac) / pow(2., i))):
                fuzzed_x.append([val - r for val in row])
                fuzzed_x.append([val + r for val in row])
                fuzzed_y.append(1)
                fuzzed_y.append(1)

    return np.concatenate((X, np.array(fuzzed_x)), axis=0), np.concatenate((y, np.array(fuzzed_y)))


class WeightedFuzzyOversampler:
    def fit_transform(self, x_train, y_train):
        shape = x_train.shape[0]
        x_train, y_train = fuzz_data(np.array(x_train), np.array(y_train))
        assert x_train.shape[0] >= shape

    def transform(self, x_test):
        raise NotImplementedError("transform should not be called on wfo.")
