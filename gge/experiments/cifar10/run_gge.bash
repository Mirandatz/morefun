#!/bin/bash -eux

# this script assumes its being run in a containerized environment configured by run_container.bash

tar -xf /dataset/train_val_test.tar.gz -C /dev/shm/

python -m gge.experiments.cifar10.run_evolution \
    --train /dev/shm/train_val_test/train \
    --validation /dev/shm/train_val_test/val \
    --output /output \
    --seed $RNG_SEED
