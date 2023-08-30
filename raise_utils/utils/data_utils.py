from raise_utils.data import Data


def _check_data(data: Data) -> None:
    """
    Ensures data is set

    :return: None
    """

    if (
        data.x_train is None or
        data.y_train is None or
        data.x_test is None or
        data.y_test is None
    ):
        raise AssertionError("Train/test data is None.")

    if len(data.x_train.shape) == 2:
        if (
            data.x_train.shape[0] != data.y_train.shape[0] or
            data.x_test.shape[0] != data.y_test.shape[0] or
            data.x_train.shape[1] != data.x_test.shape[1]
        ):
            print('x_train:', data.x_train.shape)
            print('y_train:', data.y_train.shape)
            print('x_test:', data.x_test.shape)
            print('y_test:', data.y_test.shape)
            raise AssertionError("Train/test data have a shape mismatch.")
    else:
        if (
            data.x_train.shape[0] != data.y_train.shape[0] or
            data.x_test.shape[0] != data.y_test.shape[0]
        ):
            raise AssertionError("x/y shape mismatch.")
