#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=$1
if [[ ${venv} == "" ]]; then
	venv="venv"
fi

virtualenv -p python3.8 ${venv}
source ${venv}/bin/activate
pip install --upgrade pip  # Ensure we have the newest pip
pip install .[cpu,docs,mujoco,test]
