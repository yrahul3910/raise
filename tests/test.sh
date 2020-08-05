#!/bin/sh
export PYTHONPATH=$PYTHONPATH:$PWD../
coverage run -m pytest ../
coverage xml -o cobertura.xml
