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
3. Push your changes and create a PR to `master`, which will trigger the `wheels` pipeline. This creates wheels for every platform.
4. Run `uv build`
5. Download the artifacts from the GitHub Actions run into `dist/`. Overwrite files.
6. Run `uv publish --token [pypi-token]`
7. Go to `sphinx-docs` and run `make html`.
8. Go to [Read The Docs](https://readthedocs.io), log in, and prompt a build of the docs.
9. Create a GitHub Release.
