# Note: this is an internal function, and will be wrapped in a future release.
from raise_utils.interpret.sk import Rx


def test_sk():
    a = list(range(1, 6))
    b = list(range(4, 10, 2))

    d = {'a': a, 'b': b}
    sks = Rx.sk(Rx.data(**d))

    assert sks[0].rx == 'a'
    assert sks[1].rx == 'b'
    assert sks[0].rank == 1
    assert sks[1].rank == 2
