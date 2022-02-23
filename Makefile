# we must use absolute paths because we want to mount them on containers
root_dir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
reqs_dir := $(root_dir)/requirements

test_env_tag=mirandatz/gge:test_env
dev_env_tag=mirandatz/gge:dev_env
update_requirements_tag=mirandatz/gge:update_requirements

uid := $(shell id -u)
gid := $(shell id -g)

export DOCKER_BUILDKIT=1

.PHONY: run_tests
run_tests: test_env
	docker run --rm --runtime=nvidia $(test_env_tag) pytest gge

.PHONY: update_requirements
update_requirements:
	docker build -f Docker/Dockerfile.update_requirements -t $(update_requirements_tag) .
	docker run --rm \
		--user $(uid):$(gid) \
		--mount type=bind,src=$(reqs_dir),dst=/tmp/requirements \
		--mount type=tmpfs,dst=/.cache \
		$(update_requirements_tag) pip-compile-multi -d /tmp/requirements

.PHONY: test_env
test_env:
	docker build -f Docker/Dockerfile.test_env -t $(test_env_tag) .

.PHONY: dev_env
dev_env: test_env
	docker build --build-arg test_env_tag=$(test_env_tag) -f Docker/Dockerfile.dev_env -t $(dev_env_tag) .
