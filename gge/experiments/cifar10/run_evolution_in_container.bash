#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

EXPERIMENT_DIR="/home/thiago/source/gge/gge/experiments/cifar10"
OUTPUT_DIR="$EXPERIMENT_DIR/output"
LOG_DIR="$EXPERIMENT_DIR/log"
GRAMMAR_PATH="$EXPERIMENT_DIR/grammar.lark"
INITIAL_POPULATION_DIR="$EXPERIMENT_DIR/initial_population"

DATASET_DIR="/home/thiago/source/datasets/cifar10"
TRAIN_DIR="$DATASET_DIR/train"
VALIDATION_DIR="$DATASET_DIR/validation"
TEST_DIR="$DATASET_DIR/test"

GGE_CODE_DIR="/home/thiago/source/gge"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

DIRECTORIES=("$EXPERIMENT_DIR" "$INITIAL_POPULATION_DIR" "$TRAIN_DIR" "$VALIDATION_DIR" "$TEST_DIR" "$GGE_CODE_DIR")
for DIR in "${DIRECTORIES[@]}"; do
	if [ ! -d "$DIR" ]; then
		echo "directory does not exist: $DIR"
		exit -1
	fi
done

if [ ! -f "$GRAMMAR_PATH" ]; then
	echo "file does not exist: $GRAMMAR_PATH"
	exit -1
fi

docker run \
	--user $(id -u):$(id -g) \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir="/gge/gge" \
	--env GGE_RNG_SEED=$GGE_RNG_SEED \
	-v "$OUTPUT_DIR":"/gge/output" \
	-v "$LOG_DIR":"/gge/log" \
	-v "$GRAMMAR_PATH":"/gge/grammar.lark":ro \
	-v "$INITIAL_POPULATION_DIR":"/gge/initial_population":ro \
	-v "$TRAIN_DIR":"/gge/dataset/train":ro \
	-v "$VALIDATION_DIR":"/gge/dataset/validation":ro \
	-v "$GGE_CODE_DIR":"/gge/gge":ro \
	mirandatz/gge:dev_env \
	python -m gge.experiments.cifar10.evolution 
