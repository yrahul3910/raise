import numpy as np


def fuzz_data(X, y, radii=(0., .3, .03)):
    idx = np.where(y == 1)[0]
    frac = len(idx) * 1. / len(y)

    fuzzed_x = []
    fuzzed_y = []

    for row in X[idx]:
        for i, r in enumerate(np.arange(*radii)):
            for _ in range(int((1. / frac) / pow(2., i))):
                fuzzed_x.append([val - r for val in row])
                fuzzed_x.append([val + r for val in row])
                fuzzed_y.append(1)
                fuzzed_y.append(1)

    return np.concatenate((X, np.array(fuzzed_x)), axis=0), np.concatenate((y, np.array(fuzzed_y)))


class WeightedFuzzyOversampler:
    def fit_transform(self, x_train, y_train):
        return fuzz_data(np.array(x_train), np.array(y_train))

    def transform(self, x_test):
        raise NotImplementedError("transform should not be called on wfo.")


class RadiallyWeightedFuzzyOversampler:
    def fit_transform(self, x_train, y_train):
        # Use the Mueller method for generating a d-sphere

        idx = np.where(y_train == 1)[0]
        frac = len(idx) * 1. / len(y_train)

        fuzzed_x = []
        fuzzed_y = []
        x_train = np.array(x_train)
        radii = (.1, 1., .1)
        for row in x_train[idx]:
            for i, r in enumerate(np.arange(*radii)):
                for j in range(int((1. / frac) / pow(2., i))):
                    u = np.random.normal(0, r, x_train.shape[1])
                    norm = np.sum(u ** 2) ** 0.5
                    r = np.random.random() ** (1. / x_train.shape[1])
                    x = r * u / norm

                    fuzzed_x.append(row + x)
                    fuzzed_y.append(1)

        return np.concatenate((x_train, np.array(fuzzed_x)), axis=0), np.concatenate((y_train, np.array(fuzzed_y)))
