from data import DataLoader
from hyperparams.dodge import DODGE
from learners import FeedforwardDL
from transform.transform import Transform

if __name__ == "__main__":
    directories = ["1 day", "7 days", "14 days",
                   "30 days", "90 days", "180 days", "365 days"]
    datasets = ["camel", "cloudstack", "hadoop", "cocoon",
                "deeplearning", "hive", "node", "ofbiz", "qpid"]

    for dat in datasets:
        for time in directories:
            if dat != "hive" or time != "1 day":
                continue

            data = DataLoader.from_file("./issue_close_time/" + time + "/" + dat + ".csv",
                                        target="timeOpen", col_start=0)
            Transform("rwfo").apply(data)
            Transform("smote").apply(data)

            config = {
                "n_runs": 10,
                "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
                "metrics": ["f1", "pd", "pf"],
                "random": True,
                "learners": [FeedforwardDL(random=True, weighted=1., n_epochs=50)],
                "log_path": "../log/",
                "data": [data],
                "name": "hive-rwfo"
            }
            for _ in range(50):
                config["learners"].append(
                    FeedforwardDL(random=True, weighted=1., n_epochs=50))

            dodge = DODGE(config)
            dodge.optimize()
