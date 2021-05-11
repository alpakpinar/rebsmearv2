#!/bin/bash

set -e

echo "Starting: $(date)"
echo "Running on: $(hostname)"

source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc8-opt/setup.sh
echo "Using python at: $(which python)"
python --version

ARGS=("$@")

# Extract the repo from the gridpack
tar -xvf *tgz
rm -rvf *tgz

# Virtual environment setup
ENVNAME="bucoffeaenv"
python -m venv ${ENVNAME}
source ${ENVNAME}/bin/activate
python -m pip install -e rebsmearv2 --no-cache-dir
export PYTHONPATH="${PWD}/${ENVNAME}/lib/python3.6/site-packages":${PYTHONPATH}

echo "Directory content---"
ls -lah .
echo "===================="

echo "Setup done: $(date)"
time rebexec ${ARGS[@]}
echo "Run done: $(date)"

echo "Directory content---"
ls -lah .
echo "===================="

# echo "Cleaning up."
# rm -vf *.root
# echo "End: $(date)"