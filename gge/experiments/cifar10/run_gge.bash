#!/bin/bash -x

# assumes containerized environment, configured by run_container.bash

tar -xf /dataset/train_val_test.tar.gz -C /dev/shm/

python -m gge.experiments.cifar10.run_evolution \
    -d /dev/shm/train_val_test \
    -o /output
