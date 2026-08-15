#!/bin/bash

module --force purge

module load StdEnv/2023

module load gcc/12.3
module load openmpi/4.1.5

module load python/3.11
module load arrow
module load eccodes

module load mpi4py

source ~/model_env/bin/activate
