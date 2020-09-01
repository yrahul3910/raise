from interpret import DODGEInterpreter
import pandas as pd


directories = ["1 day", "7 days", "14 days",
               "30 days", "90 days", "180 days", "365 days"]
datasets = ["cloudstack", "hadoop", "cocoon",
            "deeplearning", "hive", "node", "ofbiz", "qpid"]
path = "/Users/ryedida/Desktop/menzies/DL4SE/issue_close_time/ghost/log/"

for data in datasets:
    for time in directories:
        print(f"{data}-{time}")
        print("=" * len(f"{data}-{time}"))
        ri = DODGEInterpreter(
            [f"{path}{data}-{time}.txt"], max_by=lambda x: x[1]-10.*x[2], exclude_cols=[-1])
        print(ri.interpret())
        print()
