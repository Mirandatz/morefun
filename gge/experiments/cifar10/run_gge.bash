#!/bin/bash -x

# this script assumes its being run in a containerized environment configured by run_container.bash

if [[ -z $RNG_SEED ]]; then
    echo "RNG_SEED not defined"
    exit -1
fi

tar -xf /dataset/train_val_test.tar.gz -C /dev/shm/

python -m gge.experiments.cifar10.run_evolution \
    -d /dev/shm/train_val_test \
    -o /output \
    --seed $RNG_SEED
