# we must use absolute paths because we want to mount them on containers
root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# used to ensure files/directories are created with the correct user:group
uid := $(shell id -u)
gid := $(shell id -g)

# container tags
dev_env_tag := mirandatz/gge:dev_env

# using buildkit improves build times and decreases image sizes
export DOCKER_BUILDKIT=1

.PHONY: dev_env
dev_env:
	docker build -f Dockerfile -t $(dev_env_tag) .

.PHONY: run_tests
run_tests: dev_env
	docker run --rm --user $(uid):$(gid) -v $(root_dir):/gge $(dev_env_tag) \
		pytest

.PHONY: run_tests_parallel
run_tests_parallel: dev_env
	docker run --rm --user $(uid):$(gid) -v $(root_dir):/gge $(dev_env_tag) \
		pytest --numprocesses=auto --hypothesis-profile=parallel

.PHONY: playground
playground: dev_env
	docker run --rm --user $(uid):$(gid) -it -v $(root_dir):/gge $(dev_env_tag) \
		/bin/bash

.PHONY: update_requirements
update_requirements:
	docker run --rm \
		--env HOST_UID=$(uid) \
		--env HOST_GID=$(gid) \
		-v $(root_dir)/requirements:/requirements \
		python:3.10.4-slim-bullseye \
			/bin/bash -c 'python3 -m pip install pip-compile-multi==2.4.5 \
			&& pip-compile-multi \
			&& chown -R "$${HOST_UID}":"$${HOST_GID}" /requirements'

.PHONY: clean
clean:
	docker rmi $(dev_env_tag)
