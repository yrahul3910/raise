#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$PWD../
coverage run -m pytest ../
coverage xml -o cobertura.xml
bash <(curl -Ls https://coverage.codacy.com/get.sh)