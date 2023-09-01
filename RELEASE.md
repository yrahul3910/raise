# Release instructions

This document is for the devs only, and contains release instructions.

1. Set up a `.pypirc` file. It should look like this:
```yaml
[distutils]
  index-servers =
    pypi
    raise_utils

[pypi]
  username = __token__
  password = ENTER TOKEN HERE

[raise_utils]
  repository = https://upload.pypi.org/legacy/
  username = __token__
  password = ENTER TOKEN HERE
```
2. Change the version number in `raise_utils/__init__.py`, `setup.py`, `pyproject.toml`, and `sphinx-docs/conf.py`.
3. Run `release.sh` and enter the PyPI password when prompted.
4. Go to `sphinx-docs` and run `make html`.
5. Go to [Read The Docs](https://readthedocs.io), log in, and prompt a build of the docs.