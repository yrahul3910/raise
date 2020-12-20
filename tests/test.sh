#!/bin/bash
export PYTHONPATH=$PWD/../:$PYTHONPATH
pytest --cov=raise_utils --cov-report=xml ../
# coverage report -m
