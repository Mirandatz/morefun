SHELL := /bin/bash 
.SHELLFLAGS := -o errexit -o nounset -o pipefail -c


ifndef GGE_RNG_SEED
$(error GGE_RNG_SEED is not set)
endif

export GGE_POPULATION_SIZE 	  ?= 40
export GGE_MAX_DEPTH 		  ?= 12
export GGE_MAX_WIDE_LAYERS 	  ?= 2
export GGE_MAX_LAYER_WIDTH 	  ?= 512
export GGE_MAX_NETWORK_PARAMS ?= 750000
export GGE_LOG_LEVEL 		  ?= WARNING

MAKEFILE_PATH := $(shell realpath $(lastword $(MAKEFILE_LIST)))
export GGE_EXPERIMENT_DIR := $(shell dirname $(MAKEFILE_PATH))
export GGE_ROOT_DIR := $(shell realpath ${GGE_EXPERIMENT_DIR}/../../..)

export GGE_GRAMMAR_PATH := ${GGE_EXPERIMENT_DIR}/grammar.lark

export SEED_DIR :=${GGE_EXPERIMENT_DIR}/seed_${GGE_RNG_SEED}
export GGE_INITIAL_POPULATION_DIR := ${SEED_DIR}/initial_population
export GGE_OUTPUT_DIR := ${SEED_DIR}/output
export GGE_LOG_DIR := ${SEED_DIR}/log


.PHONY: init
init: 
	test -f ${GGE_GRAMMAR_PATH}
	test ! -d ${GGE_INITIAL_POPULATION_DIR}

	mkdir -p ${GGE_INITIAL_POPULATION_DIR}
	mkdir -p ${GGE_OUTPUT_DIR}

	docker run \
		--user $(shell id -u):$(shell id -g) \
		--rm \
		--runtime=nvidia \
		--shm-size=8G \
		--workdir=/gge/gge \
		--env GGE_RNG_SEED=${GGE_RNG_SEED} \
		--env GGE_LOG_LEVEL=${GGE_LOG_LEVEL} \
		-v ${GGE_INITIAL_POPULATION_DIR}:/gge/output \
		-v ${GGE_GRAMMAR_PATH}:/gge/grammar.lark:ro \
		-v ${GGE_ROOT_DIR}:/gge/gge:ro \
		--tmpfs /gge/log \
		mirandatz/gge:dev_env \
		python -m gge.experiments.create_initial_population \
		--population-size=${GGE_POPULATION_SIZE} \
		--max-depth=${GGE_MAX_DEPTH} \
		--max-wide-layers=${GGE_MAX_WIDE_LAYERS} \
		--max-layer-width=${GGE_MAX_LAYER_WIDTH} \
		--max-network-params=${GGE_MAX_NETWORK_PARAMS}


.PHONY: run
run: directories
	test -f ${GGE_GRAMMAR_PATH}
	test -d ${GGE_TRAIN_DIR}
	test -d ${GGE_VALIDATION_DIR}
	test -d ${GGE_ROOT_DIR}
	test -d ${GGE_INITIAL_POPULATION_DIR}

	mkdir -p ${GGE_OUTPUT_DIR}
	mkdir -p ${GGE_LOG_DIR}

	docker run \
		--user $(shell id -u):$(shell id -g) \
		--rm \
		--runtime=nvidia \
		--shm-size=8G \
		--workdir=/gge/gge \
		--env GGE_RNG_SEED=${GGE_RNG_SEED} \
		--env GGE_LOG_LEVEL=${GGE_LOG_LEVEL} \
		-v ${GGE_OUTPUT_DIR}:/gge/output \
		-v ${GGE_LOG_DIR}:/gge/log \
		-v ${GGE_GRAMMAR_PATH}:/gge/grammar.lark:ro \
		-v ${GGE_INITIAL_POPULATION_DIR}:/gge/initial_population:ro \
		-v ${GGE_TRAIN_DIR}:/gge/dataset/train:ro \
		-v ${GGE_VALIDATION_DIR}:/gge/dataset/validation:ro \
		-v ${GGE_ROOT_DIR}:/gge/gge:ro \
		mirandatz/gge:dev_env \
		python -m gge.experiments.cifar10.evolution

.PHONY: init_and_run
init_and_run: init run
	;
