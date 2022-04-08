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
	docker run \
		--rm \
		--user $(uid):$(gid) \
		-v $(root_dir):/gge \
		$(dev_env_tag) \
		pytest

.PHONY: playground
playground: dev_env
	docker run \
		--rm \
		--user $(uid):$(gid) \
		-it \
		-v $(root_dir):/gge \
		$(dev_env_tag) \
		/bin/bash

.PHONY: clean
clean:
	docker rmi $(dev_env_tag)
