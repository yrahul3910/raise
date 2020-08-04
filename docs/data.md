# `raise.data`

Provides the `Data` and `DataLoader` classes. Users should use the latter to obtain an object of type `Data`.

## `raise.data.Data`

A wrapper object that contains the train and test splits of the data attributes and targets. Implements the `__add__` function to allow for combining of two `Data` objects.

## `raise.data.DataLoader`

### `from_files(base_path: str, files: list, target:str = "bug", col_start:int = 3, col_stop:int = -2) -> Data:`

Returns a `Data` object by combining all but the last CSV file in `files`, all of which are located in `base_path/`, and the (binary) target variable is assumed to be `target`. Only considers columns from `col_start` (inclusive) to `col_stop` (exclusive). Default values are configured for the PROMISE defect prediction datasets.

### `from_file(path:str, target) -> Data:`

Returns a `Data` object built from the CSV file at `path`, considering `target` as the (binary) classification target.