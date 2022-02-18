update_requirements_tag=mirandatz/gge-update_requirements
test_env_tag=mirandatz/gge-test_env
dev_env_tag=mirandatz/gge-dev_env

export DOCKER_BUILDKIT=1

.PHONY: run_tests
run_tests: test_env
	docker run --runtime=nvidia $(test_env_tag) pytest /gge/gge/tests

.PHONY: update_requirements
update_requirements:
	docker build -f Docker/Dockerfile.update_requirements -t $(update_requirements_tag) .
	IMG_ID=$$(docker create $(update_requirements_tag)) \
		&& docker cp $${IMG_ID}:/requirements/base.txt ./requirements \
		&& docker cp $${IMG_ID}:/requirements/test.txt ./requirements \
		&& docker cp $${IMG_ID}:/requirements/dev.txt ./requirements \
		&& docker rm -v $${IMG_ID}

.PHONY: test_env
test_env:
	docker build -f Docker/Dockerfile.test_env -t $(test_env_tag) .

.PHONY: dev_env
dev_env: test_env
	docker build --build-arg test_env_tag=$(test_env_tag) -f Docker/Dockerfile.dev_env -t $(dev_env_tag) .
