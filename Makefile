# we must use absolute paths because we want to mount them on containers
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# used to ensure files/directories are created with the correct user:group
username := $(shell whoami)
user_id := $(shell id -u)
group_id := $(shell id -g)

# project settings
PROJECT_NAME := morefun
python_version := 3.11.6

# container tags
DEV_ENG_TAG := mirandatz/$(PROJECT_NAME):dev_env

# using buildkit improves build times and decreases image sizes
export DOCKER_BUILDKIT=1

.PHONY: dev_env
dev_env:
	docker build \
		--build-arg username=$(username) \
		--build-arg user_id=$(user_id) \
		--build-arg group_id=$(group_id) \
		--build-arg python_version=$(python_version) \
		-f Dockerfile \
		-t $(DEV_ENG_TAG) .

.PHONY: run_tests
run_tests: dev_env
	docker run \
		--rm \
		--runtime=nvidia \
		--user $(user_id):$(group_id) \
		-v $(ROOT_DIR):/app/$(PROJECT_NAME) \
		--workdir /app/$(PROJECT_NAME) \
		$(DEV_ENG_TAG) \
		bash -c "pytest ./$(PROJECT_NAME)/tests --numprocesses=auto --hypothesis-profile=parallel"

.PHONY: run_tests_sequential
run_tests_sequential: dev_env
	docker run \
		--rm \
		--runtime=nvidia \
		--user $(user_id):$(group_id) \
		-v $(ROOT_DIR):/app/$(PROJECT_NAME) \
		--workdir /app/$(PROJECT_NAME) \
		$(DEV_ENG_TAG) \
		bash -c "pytest ./$(PROJECT_NAME)/tests"


.PHONY: playground
playground: dev_env
	docker run \
		--rm \
		--runtime=nvidia \
		--user $(user_id):$(group_id) \
		-it \
		-v $(ROOT_DIR):/app/$(PROJECT_NAME) \
		$(DEV_ENG_TAG) \
		/bin/bash

.PHONY: update_requirements
update_requirements:
	docker run \
		--rm \
		--env HOST_UID=$(user_id) \
		--env HOST_GID=$(group_id) \
		-v $(ROOT_DIR)/requirements:/requirements \
		python:${python_version}-slim-bullseye \
			/bin/bash -c 'python3 -m pip install uv \
			&& uv pip compile requirements/base.in > requirements/base.txt \
			&& uv pip compile requirements/dev.in > requirements/dev.txt \
			&& uv pip compile requirements/test.in > requirements/test.txt \
			&& chown -R "$${HOST_UID}":"$${HOST_GID}" /requirements'

.PHONY: clean
clean:
	docker rmi $(DEV_ENG_TAG)
