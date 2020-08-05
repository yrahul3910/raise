#!/bin/sh
export PYTHONPATH=$PYTHONPATH:$PWD../
pytest --cov=../ ./
