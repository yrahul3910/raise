from raise_utils.interpret import DODGEInterpreter
import numpy as np


def test_dodge_interpreter():
    interp = DODGEInterpreter(files=['./interpret/test.txt'], max_by=0)
    results = interp.interpret()

    assert isinstance(results, dict)
    assert 'test.txt' in results.keys()
    assert np.median(results['test.txt'][0]) - 0.892 < 1e-3
