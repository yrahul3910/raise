<p align="center">
<img src="logo.png" height="80px" /><br />
<a href="https://github.com/yrahul3910/raise/tree/master/docs">quick guide</a>  |
<a href="https://raise.readthedocs.io/en/latest/">docs</a>  |
<a href="https://github.com/yrahul3910/raise/blob/master/CHANGES.md">change log</a>  |
<a href="mailto:r.yedida@pm.me">contact</a>
</p>
<p align="center">
<img src="https://img.shields.io/badge/license-LGPL-green.svg">&nbsp;
<a href="https://badge.fury.io/py/raise-utils"><img src="https://badge.fury.io/py/raise-utils.svg" alt="PyPI version" height="20"></a>
<a href="https://pepy.tech/project/raise-utils"><img src="https://static.pepy.tech/badge/raise-utils" alt="Downloads" />
<a href='https://raise.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/raise/badge/?version=latest' alt='Documentation Status' />
</a>
<a href="https://circleci.com/gh/yrahul3910/raise/tree/master"><img src="https://circleci.com/gh/yrahul3910/raise/tree/master.svg?style=svg" alt="CircleCI" /></a>
<a href="https://app.codacy.com/gh/yrahul3910/raise/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade"><img src="https://app.codacy.com/project/badge/Grade/a777f84abcfa41ccaefaf4325d2d5e3b"/></a>&nbsp;
<a href="https://codecov.io/gh/yrahul3910/raise">
    <img src="https://codecov.io/gh/yrahul3910/raise/branch/master/graph/badge.svg?token=6U8KR6PYZA"/>
</a>&nbsp;
</p> <hr />

# The RAISE package

This package provides implementations for some algorithms from the RAISE lab at NC State University. These should be taken as the reference implementations for the papers proposing them, although the code versions that appear in this repo are typically cleaned up versions of the ones actually used. In general, this repo aims to be PEP8-compliant, well-tested, and reasonably-documented. Lab members are responsible for adding their own code via PRs.

The package's intended users are the lab members, who can access a standardized way of loading data, computing metrics, and running statistical tests, and other researchers that wish to reuse/reproduce our algorithms. This code is provided under an LGPL 2.1 license.

## Install

The `raise_utils` package is available on PyPI. Your package manager of choice should handle it. Python 2 is unsupported, but we recommend Python 3.10+.

```
pip3 install raise_utils
```

## Contributing

Generally speaking, while this repo is open to PRs, it is intended to be contributed to by lab members. However, feel free to submit PRs to:
* Improve code quality
* Add documentation in `/docs`, or `sphinx`
* Add more or better tests
* Fix bugs

## Documentation

Read our docs [here](https://github.com/yrahul3910/raise/tree/master/docs). Although we do have a readthedocs page, that has proven to be unreliable, historically speaking. We will (probably) fix it at some point.
You can find examples [in our examples directory](https://github.com/yrahul3910/raise/tree/master/raise_utils/examples).

## Citing

You do not need to cite this repo, but we are grateful if you do. We ask that you instead cite the original paper proposing the method that you are reusing in your work.

## Contact

The current maintainer of this repository is [Rahul Yedida](https://ryedida.me).
