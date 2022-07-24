SHELL := /bin/bash
.SHELLFLAGS := -o errexit -o nounset -o pipefail -c 

ifndef GGE_RNG_SEED
$(error GGE_RNG_SEED is not set)
endif

export GGE_MAX_GENERATIONS    	  ?= 1000
export GGE_MUTANTS_PER_GENERATION ?= 2
export GGE_BATCH_SIZE 			  ?= 64
export GGE_EPOCHS 				  ?= 40

export GGE_IMG_WIDTH   ?= 32
export GGE_IMG_HEIGHT  ?= 32
export GGE_IMG_DEPTH   ?= 3
export GGE_CLASS_COUNT ?= 10

export GGE_POPULATION_SIZE 	  ?= 40
export GGE_MAX_DEPTH 		  ?= 12
export GGE_MAX_WIDE_LAYERS 	  ?= 2
export GGE_MAX_LAYER_WIDTH 	  ?= 512
export GGE_MAX_NETWORK_PARAMS ?= 750000

export GGE_EXPECTED_NUMBER_OF_TRAIN_INSTANCES 	   = 37500
export GGE_EXPECTED_NUMBER_OF_VALIDATION_INSTANCES = 12500

export GGE_LOG_LEVEL ?= INFO

MAKEFILE_PATH := $(shell realpath $(lastword $(MAKEFILE_LIST)))
EXPERIMENT_DIR := $(shell dirname $(MAKEFILE_PATH))
export HOST_GGE_ROOT_DIR := $(shell realpath ${EXPERIMENT_DIR}/../../..)

export GRAMMAR_PATH := ${EXPERIMENT_DIR}/grammar.lark

SEED_DIR :=${EXPERIMENT_DIR}/seed_${GGE_RNG_SEED}
export INITIAL_POPULATION_DIR := ${SEED_DIR}/initial_population
export OUTPUT_DIR := ${SEED_DIR}/output
export LOG_DIR := ${SEED_DIR}/log

.PHONY: init
init: 
	test -f $${GRAMMAR_PATH}
	test ! -d $${INITIAL_POPULATION_DIR}

	mkdir -p $${INITIAL_POPULATION_DIR}
	mkdir -p $${OUTPUT_DIR}

	docker run \
		--user $(shell id -u):$(shell id -g) \
		--rm \
		--runtime=nvidia \
		--shm-size=8G \
		--workdir=/gge/gge \
		--env GGE_RNG_SEED=${GGE_RNG_SEED} \
		--env-file <(env | grep ^GGE_) \
		-v ${INITIAL_POPULATION_DIR}:/gge/output \
		-v ${GRAMMAR_PATH}:/gge/grammar.lark:ro \
		-v ${HOST_GGE_ROOT_DIR}:/gge/gge:ro \
		--tmpfs /gge/log \
		mirandatz/gge:dev_env \
		python -m gge.experiments.create_initial_population \
		--population-size=${GGE_POPULATION_SIZE} \
		--max-depth=${GGE_MAX_DEPTH} \
		--max-wide-layers=${GGE_MAX_WIDE_LAYERS} \
		--max-layer-width=${GGE_MAX_LAYER_WIDTH} \
		--max-network-params=${GGE_MAX_NETWORK_PARAMS}


.PHONY: run
run:
	$(eval export HOST_TRAIN_DIR=${GGE_DATASET_DIR}/train)
	$(eval export HOST_VALIDATION_DIR=${GGE_DATASET_DIR}/validation)

	test -d $${HOST_GGE_ROOT_DIR}
	test -f $${GRAMMAR_PATH}
	test -d $${HOST_TRAIN_DIR}
	test -d $${HOST_VALIDATION_DIR}
	test -d $${INITIAL_POPULATION_DIR}
	test -d $${GGE_DATASET_DIR}

	mkdir -p ${OUTPUT_DIR}
	mkdir -p ${LOG_DIR}

	docker run \
		--user $(shell id -u):$(shell id -g) \
		--rm \
		--runtime=nvidia \
		--shm-size=8G \
		--workdir=/gge/gge \
		--env-file <(env | grep GGE_) \
		-v ${OUTPUT_DIR}:/gge/output \
		-v ${LOG_DIR}:/gge/log \
		-v ${GRAMMAR_PATH}:/gge/grammar.lark:ro \
		-v ${INITIAL_POPULATION_DIR}:/gge/initial_population:ro \
		-v ${HOST_TRAIN_DIR}:/gge/dataset/train:ro \
		-v ${HOST_VALIDATION_DIR}:/gge/dataset/validation:ro \
		-v ${HOST_GGE_ROOT_DIR}:/gge/gge:ro \
		mirandatz/gge:dev_env \
		python -m gge.experiments.evolution

.PHONY: init_and_run
init_and_run: init run
	;
