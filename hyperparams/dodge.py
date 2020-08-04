import random
import string

from data.data import Data
import itertools

from metrics.metrics import ClassificationMetrics
from transform.transform import Transform


class DODGE:
    """
    Implements the DODGE hyper-parameter optimizer
    """
    def __init__(self, config, verbose=True):
        """
        Initializes DODGE.
        :param config: The config object.
        :param verbose: Whether to print debug info.
        """
        self.config = config
        self.verbose = verbose

    def optimize(self):
        dic = {}
        for _ in range(self.config["n_runs"]):
            if self.verbose:
                print("Run #", _)
                print("=" * len("Run #" + str(_)))

            transforms = self.config["transforms"]
            learners = self.config["learners"]
            combine = list(itertools.product(transforms, learners))
            data: Data = self.config["data"][0]

            func_str_dic = {}
            func_str_counter_dic = {}
            lis_value = []
            for pair in combine:
                pair_name = pair[0] + random.choice(string.ascii_letters) + "|" + pair[1].name
                func_str_dic[pair_name] = [Transform(pair[0], random=True), pair[1]]
                func_str_counter_dic[pair_name] = 0

            for counter in range(30):
                try:
                    if self.verbose:
                        print(counter, flush=True)

                    keys = [k for k, v in func_str_counter_dic.items() if v == 0]
                    key = random.choice(keys)
                    transform, model = func_str_dic[key]
                    transform.apply(data)
                    model.set_data(data.x_train, data.y_train, data.x_test, data.y_test)
                    model.fit()
                    preds = model.predict(data.x_test)
                    metrics = ClassificationMetrics(data.y_test, preds)
                    metrics.add_metrics(self.config["metrics"])
                    metric = metrics.get_metrics()[0]

                    if all(abs(t - metric) > 0.2 for t in lis_value):
                        lis_value.append(metric)
                        func_str_counter_dic[key] += 1
                    else:
                        func_str_counter_dic[key] -= 1

                    if counter not in dic.keys():
                        dic[counter] = []
                    dic[counter].append(max(lis_value))
                except ValueError:
                    pass

        print(dic)
