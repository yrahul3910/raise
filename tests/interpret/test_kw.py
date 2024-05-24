import sys
from io import StringIO

import numpy as np
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from raise_utils.interpret import KruskalWallis


def test_kruskal_wallis_when_similar():
    np.random.seed(0)
    custom_output = StringIO()
    sys.stdout = custom_output

    data = {
        'a': np.random.normal(0.5, 0.05, 100),
        'b': np.random.normal(0.51, 0.05, 100),
    }
    kw = KruskalWallis(data)
    post_hoc, max_group, is_better = kw.pprint()

    sys.stdout = sys.__stdout__
    assert post_hoc is None and max_group is None and is_better is None


def test_kruskal_wallis_when_different():
    np.random.seed(0)
    custom_output = StringIO()
    sys.stdout = custom_output

    data = {
        'a': np.random.normal(0.5, 0.02, 100),
        'b': np.random.normal(0.6, 0.02, 100),
    }
    kw = KruskalWallis(data)
    post_hoc, max_group, is_better = kw.pprint()

    sys.stdout = sys.__stdout__

    # Run stats ourselves
    _, p = kruskal(data['a'], data['b'])
    adjusted_p = multipletests([p], method='fdr_tsbh')[1][0]

    assert post_hoc is not None and max_group is not None and is_better is not None
    assert post_hoc.to_numpy()[0][0] - adjusted_p <= np.finfo(float).eps
    assert max_group == 'b'
    assert is_better
