#!/bin/bash

set -o errexit  # abort on nonzero exitstatus
set -o nounset  # abort on unbound variable
set -o pipefail # don't hide errors within pipes
set -o xtrace   # echo commands as they are run

scenario=single_objective

for i in {0..4}; do
    export exp_dir="/disk2/thiago/experiments/v2/${scenario}/run_${i}"
    
    /home/thiago/source/morefun/scripts/initialize_population.py \
        -s "${exp_dir}/settings.yaml" \
        -d /disk2/thiago/datasets/cifar10 \
        -o ${exp_dir}/output

    /home/thiago/source/morefun/scripts/evolve_population.py \
        -s "${exp_dir}/settings.yaml" \
        -d /disk2/thiago/datasets/cifar10 \
        -o ${exp_dir}/output \
        --generations 50
done

for i in {0..4}; do
    export exp_dir="/disk2/thiago/experiments/v2/${scenario}/run_${i}"

    /home/thiago/source/morefun/scripts/analyze_results.py \
        -s "${exp_dir}/settings.yaml" \
        -d /disk2/thiago/datasets/cifar10 \
        -o ${exp_dir}/output
done
