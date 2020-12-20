#!/bin/bash
export PYTHONPATH=$PWD/../:$PYTHONPATH
pytest --cov=raise_utils ../
# coverage report -m
