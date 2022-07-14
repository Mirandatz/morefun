#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

SCRIPT_DIR=$(dirname $(realpath "$0"))
GRAMMAR_PATH="$SCRIPT_DIR/grammar.lark"
OUTPUT_DIR="$SCRIPT_DIR/initial_population"

GGE_CODE_DIR=$"/home/thiago/source/gge"

mkdir -p "$OUTPUT_DIR"

docker run \
	--user $(id -u):$(id -g) \
	--rm \
	--runtime=nvidia \
	--shm-size=8G \
	--workdir="/gge/gge" \
    -v "$OUTPUT_DIR":"/gge/output" \
    -v "$GRAMMAR_PATH":"/gge/grammar.lark":ro \
	-v "$GGE_CODE_DIR":"/gge/gge":ro \
	mirandatz/gge:dev_env \
	python -m gge.experiments.create_initial_population \
        --grammar-path="/gge/grammar.lark" \
        --output-dir="/gge/output" \
        --population-size=40 \
        --max-depth=12 \
        --max-wide-layers=2 \
        --max-layer-width=512 \
        --max-network-params=750000 \
        --rng-seed=0
