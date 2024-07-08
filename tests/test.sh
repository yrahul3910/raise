#!/usr/bin/env bash

if [[ "$OSTYPE" == "darwin"* ]]; then
    export KERAS_BACKEND=torch
else
    export KERAS_BACKEND=tensorflow
fi

export PYTHONPATH=$PWD/../:$PYTHONPATH

if command -v poetry &> /dev/null; then
    poetry run pytest --new-first --lf -vv -x --cov=raise_utils --cov-report=xml ../
else
    pytest --new-first --lf -vv -x --cov=raise_utils --cov-report=xml ../
fi
    

SUCCESS=$?

if [ "$1" = "--coverage" ]; then
    coverage report -m --skip-covered --sort=cover
fi

# Remove temp files
rm -rf log/ test_dodge.txt test_experiment

exit $SUCCESS
