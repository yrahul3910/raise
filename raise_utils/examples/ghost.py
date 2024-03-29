from raise_utils.data import DataLoader
from raise_utils.hyperparams.dodge import DODGE
from raise_utils.learners import FeedforwardDL
from raise_utils.transforms import Transform

if __name__ == "__main__":
    data = DataLoader.from_files("./promise/", ["camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"])
    Transform("wfo").apply(data)
    Transform("smote").apply(data)

    config = {
        "n_runs": 20,
        "transforms": ["normalize", "standardize", "robust", "maxabs", "minmax"] * 30,
        "metrics": ["f1", "pd", "pf"],
        "random": True,
        "learners": [FeedforwardDL(random=True)],
        "log_path": "./log/",
        "data": [data],
        "name": "camel-pd-pf"
    }
    for _ in range(50):
        config["learners"].append(FeedforwardDL(random=True))

    dodge = DODGE(config)
    dodge.optimize()
