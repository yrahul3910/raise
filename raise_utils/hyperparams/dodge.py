import random
import string
import os
import sys
import numpy as np

from raise_utils.data.data import Data
import itertools

from raise_utils.metrics.metrics import ClassificationMetrics
from raise_utils.transform.transform import Transform

import gc


class DODGE:
    """
    Implements the DODGE hyper-parameter optimizer
    """

    def __init__(self, config):
        """
        Initializes DODGE.
        :param config: The config object.
        :param verbose: Whether to print debug info.
        """
        self.config = config
        if self.config["log_path"] is None:
            self.file = sys.stdout
        else:
            self.file = open(os.path.join(
                self.config['log_path'], self.config['name'] + '.txt'), 'w')
        self.post_train_hooks = self.config.get("post_train_hooks", None)

    def __del__(self):
        self.file.close()
        gc.collect()

    def optimize(self):
        for learner in self.config["learners"]:
            print(learner, flush=True)

        dic = {}
        dic_func = {}
        for _ in range(self.config.get("n_runs", 1)):
            print("Run #", _, file=self.file)
            print("=" * len("Run #" + str(_)), file=self.file)
            print("Run #", _)
            print("=" * len("Run #" + str(_)))

            data: Data = self.config["data"][0]

            func_str_dic = {}
            func_str_counter_dic = {}
            lis_value = []
            for pair in itertools.product(self.config["transforms"], self.config["learners"]):
                pair_name = pair[0] + \
                    random.choice(string.ascii_letters) + "|" + pair[1].name
                func_str_dic[pair_name] = [
                    Transform(pair[0], random=True), pair[1]]
                func_str_counter_dic[pair_name] = 0

            for counter in range(self.config.get('n_iters', 30)):
                try:
                    if counter not in dic_func.keys():
                        dic_func[counter] = []

                    if counter not in dic.keys():
                        dic[counter] = []

                    keys = [k for k, v in func_str_counter_dic.items()
                            if v == 0]
                    key = random.choice(keys)
                    print('setting:', key)
                    print('setting:', key, file=self.file)
                    transform, model = func_str_dic[key]
                    transform.apply(data)
                    model.set_data(data.x_train, data.y_train,
                                   data.x_test, data.y_test)
                    model.fit()

                    # Run post-training hooks
                    if self.post_train_hooks is not None:
                        for hook in self.post_train_hooks:
                            hook.call(model, data.x_test, data.y_test)

                    preds = model.predict(data.x_test)

                    if len(data.y_test.shape) > 1:
                        metrics = ClassificationMetrics(
                            np.argmax(data.y_test, axis=-1), preds)
                    else:
                        metrics = ClassificationMetrics(data.y_test, preds)
                    metrics.add_metrics(self.config["metrics"])
                    print('iter', counter, ':',
                          metrics.get_metrics(), file=self.file)
                    print('iter', counter, ':',
                          metrics.get_metrics())
                    metric = metrics.get_metrics()[0]

                    if all(abs(t - metric) > 0.2 for t in lis_value):
                        lis_value.append(metric)
                        func_str_counter_dic[key] += 1
                    else:
                        func_str_counter_dic[key] -= 1

                    if counter not in dic.keys():
                        dic[counter] = []

                    dic_func[counter].append(key)
                    dic[counter].append(max(lis_value))
                except ValueError:
                    raise

        dic["settings"] = dic_func
        print(dic, file=self.file)
        self.file.flush()
        self.file.close()
