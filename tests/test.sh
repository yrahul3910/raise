#!/bin/bash

export PYTHONPATH=$PWD/../:$PYTHONPATH
pytest --cov=raise_utils --cov-report=xml ../

if [ "$1" = "--coverage" ]; then
    coverage report -m --skip-covered --sort=cover
fi
