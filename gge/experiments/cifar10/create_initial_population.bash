#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

# params
GGE_POPULATION_SIZE=${GGE_POPULATION_SIZE:=40}
GGE_MAX_DEPTH=${GGE_MAX_DEPTH:=12}
GGE_MAX_WIDE_LAYERS=${GGE_MAX_WIDE_LAYERS:=2}
GGE_MAX_LAYER_WIDTH=${GGE_MAX_LAYER_WIDTH:=512}
GGE_MAX_NETWORK_PARAMS=${GGE_MAX_NETWORK_PARAMS:=750000}
GGE_LOG_LEVEL=${GGE_LOG_LEVEL:=WARNING}

# used to generate default values for grammar path and output dir
GGE_EXPERIMENT_DIR=$(dirname $(realpath "$0"))
GGE_ROOT_DIR=$(realpath "${GGE_EXPERIMENT_DIR}/../../..")

GGE_GRAMMAR_PATH=${GGE_GRAMMAR_PATH:="$GGE_EXPERIMENT_DIR/grammar.lark"}
GGE_OUTPUT_DIR=${GGE_OUTPUT_DIR:="$GGE_EXPERIMENT_DIR/initial_population"}

# validate paths
if [ ! -f $GGE_GRAMMAR_PATH ]; then
    echo "file not found: $GGE_GRAMMAR_PATH"
    exit  -1
fi

mkdir -p "$GGE_OUTPUT_DIR"

docker run \
	--user $(id -u):$(id -g) \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir="/gge/gge" \
    -v "$GGE_OUTPUT_DIR":"/gge/output" \
    -v "$GGE_GRAMMAR_PATH":"/gge/grammar.lark":ro \
	-v "$GGE_ROOT_DIR":"/gge/gge":ro \
    --env GGE_RNG_SEED=$GGE_RNG_SEED \
    --env GGE_LOG_LEVEL=$GGE_LOG_LEVEL \
    --tmpfs "/gge/log" \
	mirandatz/gge:dev_env \
	python -m gge.experiments.create_initial_population \
        --population-size=$GGE_POPULATION_SIZE \
        --max-depth=$GGE_MAX_DEPTH \
        --max-wide-layers=$GGE_MAX_WIDE_LAYERS \
        --max-layer-width=$GGE_MAX_LAYER_WIDTH \
        --max-network-params=$GGE_MAX_NETWORK_PARAMS
