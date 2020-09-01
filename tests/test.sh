#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD../
coverage run -m pytest ../
