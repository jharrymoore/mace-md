#!/bin/bash -l
mamba env create -f mace-openmm.yml

conda activate mace-openmm

pip install git+https://github.com/ACEsuit/mace.git
pip install git+https://github.com/jharrymoore/openmm-ml.git@ml_alchemy
pip install .
