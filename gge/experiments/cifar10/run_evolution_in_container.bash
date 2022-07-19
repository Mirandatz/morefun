#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# used to generate default values for paths
GGE_EXPERIMENT_DIR=$(dirname "$(realpath "$0")")
GGE_ROOT_DIR=$(realpath "${GGE_EXPERIMENT_DIR}/../../..")
GGE_DATASET_DIR=${GGE_DATASET_DIR:="/home/datasets/cifar10"}

# params
GGE_OUTPUT_DIR=${GGE_OUTPUT_DIR:="$GGE_EXPERIMENT_DIR/output"}
GGE_LOG_DIR=${GGE_LOG_DIR="$GGE_EXPERIMENT_DIR/log"}
GGE_GRAMMAR_PATH=${GGE_GRAMMAR_PATH:="$GGE_EXPERIMENT_DIR/grammar.lark"}
GGE_INITIAL_POPULATION_DIR=${GGE_INITIAL_POPULATION_DIR:="$GGE_EXPERIMENT_DIR/initial_population"}
GGE_TRAIN_DIR=${GGE_TRAIN_DIR:="$GGE_DATASET_DIR/train"}
GGE_VALIDATION_DIR=${GGE_VALIDATION_DIR:="$GGE_DATASET_DIR/validation"}
GGE_LOG_LEVEL=${GGE_LOG_LEVEL:="INFO"}

mkdir -p "$GGE_OUTPUT_DIR"
mkdir -p "$GGE_LOG_DIR"

# validate paths
DIRECTORIES=("$GGE_EXPERIMENT_DIR"
	"$GGE_INITIAL_POPULATION_DIR"
	"$GGE_TRAIN_DIR"
	"$GGE_VALIDATION_DIR"
	"$GGE_ROOT_DIR")
for DIR in "${DIRECTORIES[@]}"; do
	if [ ! -d "$DIR" ]; then
		echo "directory does not exist: $DIR"
		exit 255
	fi
done

if [ ! -f "$GGE_ROOT_DIR/.gge_root" ]; then
	echo "environment variable GGE_ROOT_DIR does not point to the root of GGE"
	exit 255
fi

if [ ! -f "$GGE_GRAMMAR_PATH" ]; then
	echo "file does not exist: $GGE_GRAMMAR_PATH"
	exit 255
fi

docker run \
	--user "$(id -u)":"$(id -g)" \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir="/gge/gge" \
	--env GGE_RNG_SEED="$GGE_RNG_SEED" \
	--env GGE_LOG_LEVEL="$GGE_LOG_LEVEL" \
	-v "$GGE_OUTPUT_DIR":"/gge/output" \
	-v "$GGE_LOG_DIR":"/gge/log" \
	-v "$GGE_GRAMMAR_PATH":"/gge/grammar.lark":ro \
	-v "$GGE_INITIAL_POPULATION_DIR":"/gge/initial_population":ro \
	-v "$GGE_TRAIN_DIR":"/gge/dataset/train":ro \
	-v "$GGE_VALIDATION_DIR":"/gge/dataset/validation":ro \
	-v "$GGE_ROOT_DIR":"/gge/gge":ro \
	mirandatz/gge:dev_env \
	python -m gge.experiments.cifar10.evolution
