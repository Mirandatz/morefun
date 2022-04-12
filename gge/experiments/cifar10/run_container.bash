#!/bin/bash -eux

DOCKER_ENV_TAG=mirandatz/gge:dev_env

SCRIPT_DIR=$(dirname $(realpath "$0"))
GGE_SRC=$(realpath "${SCRIPT_DIR}/../../..")

docker run \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir=/src \
	--env RNG_SEED=$RNG_SEED \
	-v "$GGE_SRC":/src:ro \
	-v "$DATASET_DIR":/dataset:ro \
	-v "$OUTPUT_DIR":/output \
	"$DOCKER_ENV_TAG" \
	/src/gge/experiments/cifar10/run_gge.bash
