from raise_utils.interpret import ScottKnott


def test_sk():
    a = list(range(1, 6))
    b = list(range(4, 10, 2))

    d = {'a': a, 'b': b}
    sks = ScottKnott(d).get_results()

    assert sks[0].rx == 'a'
    assert sks[1].rx == 'b'
    assert sks[0].rank == 1
    assert sks[1].rank == 2
