# `raise_utils.data`

This package provides the `Data` and `DataLoader` classes. Users should use the latter to obtain an object of type `Data`. `DataLoader` allows users to select either one or multiple files as the data; in the latter case, it is assumed that the last file is the test set. Both functions allow you to choose the start and stop columns to be considered.

## `raise_utils.data.DataLoader`

`from_files(base_path: str, files: list, target:str = "bug", col_start:int = 3, col_stop:int = -2) -> Data`

Returns a `Data` object by combining all but the last CSV file in `files`, all of which are located in `base_path/`, and the (binary) target variable is assumed to be `target`. Only considers columns from `col_start` (inclusive) to `col_stop` (exclusive). Default values are configured for the PROMISE defect prediction datasets.

`from_file(path:str, target, col_sart:int = 3, col_stop:int = -2) -> Data`

Returns a `Data` object built from the CSV file at `path`, considering `target` as the (binary) classification target.
