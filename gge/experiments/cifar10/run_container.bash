#!/bin/bash -x

GGE_SRC="${GGE_SRC:=/home/thiago/source/gge}"
DATASETS_DIR="${DATASETS_DIR:=/home/thiago/source/datasets/cifar10}"
OUTPUT_DIR="${OUTPUT_DIR:=/dev/shm/hehe}"

DOCKER_ENV_TAG=mirandatz/gge:dev_env

docker run \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir=/src \
	-v "$GGE_SRC":/src:ro \
	-v "$DATASETS_DIR":/dataset:ro \
	-v "$OUTPUT_DIR":/output \
	"$DOCKER_ENV_TAG" \
	bash /src/gge/experiments/cifar10/run_gge.bash