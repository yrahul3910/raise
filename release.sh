#!/bin/bash

rm -rf dist/

echo "(1 / 2) Building packages"
python3.8 setup.py sdist bdist_wheel

echo "(2 / 2) Uploading to PyPI"
python3.8 -m twine upload --repository pypi dist/*



