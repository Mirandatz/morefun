#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o xtrace

SCRIPT_DIR=$(dirname $(realpath "$0"))
GGE_SRC=$(realpath "${SCRIPT_DIR}/../../..")

docker run \
	--user $(id -u):$(id -g) \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir=/src \
	--env RNG_SEED=$RNG_SEED \
	-v "$GGE_SRC":/src:ro \
	-v "$DATASETS_DIR/cifar10":/dataset:ro \
	-v "$OUTPUT_DIR":/output \
	mirandatz/gge:dev_env \
	python -m gge.experiments.cifar10.evolution \
		--train /dataset/train \
		--validation /dataset/val \
		--output /output \
		--seed $RNG_SEED
