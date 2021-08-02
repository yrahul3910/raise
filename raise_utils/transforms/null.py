class NullTransform:
    def fit_transform(self, *args, **kwargs):
        return args

    def transform(self, *args, **kwargs):
        return args
