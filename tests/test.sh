#!/bin/bash

export PYTHONPATH=$PWD/../:$PYTHONPATH
pytest --cov=raise_utils --cov-report=xml ../

SUCCESS=$?

if [ "$1" = "--coverage" ]; then
    coverage report -m --skip-covered --sort=cover
fi

# Remove temp files
rm test_dodge.txt test_experiment

exit $SUCCESS
