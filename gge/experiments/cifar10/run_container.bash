#!/bin/bash -x

if [[ -z $RNG_SEED ]]; then
    echo "RNG_SEED not defined"
    exit -1
fi

# customize for different machines
GGE_SRC="${GGE_SRC:=/home/thiago/source/gge}"
DATASETS_DIR="${DATASETS_DIR:=/home/thiago/source/datasets/cifar10}"
OUTPUT_DIR="${OUTPUT_DIR:=$(mktemp -d -p /dev/shm)}"

DOCKER_ENV_TAG=mirandatz/gge:dev_env

docker run \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir=/src \
	--env RNG_SEED=$RNG_SEED \
	-v "$GGE_SRC":/src:ro \
	-v "$DATASETS_DIR":/dataset:ro \
	-v "$OUTPUT_DIR":/output \
	"$DOCKER_ENV_TAG" \
	bash /src/gge/experiments/cifar10/run_gge.bash
