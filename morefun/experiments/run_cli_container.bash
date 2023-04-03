#!/bin/bash

# before running, set the following envinroment variables:
# MOREFUN_DATASETS_PATH   path to a directory with datasets
# MOREFUN_REPOSITORY_PATH path to the root of the MOREFUN git repository

# usage:
# ./cli.bash <path-to-settings.yaml> <morefun-python-cli-commands>

set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

CODE_MOUNT_POINT=$(realpath "$MOREFUN_REPOSITORY_PATH")
DATASETS_MOUNT_POINT="$(realpath "$MOREFUN_DATASETS_PATH")"

# ensuring directories exist so they are no created (as root) by docker
if [ ! -d "$CODE_MOUNT_POINT" ]; then
    echo "envvar CODE_MOUNT_POINT does not point to a directory: $CODE_MOUNT_POINT"
    exit 1
fi
if [ ! -d "$DATASETS_MOUNT_POINT" ]; then
    echo "envvar DATASETS_MOUNT_POINT does not point to a directory: $DATASETS_MOUNT_POINT"
    exit 1
fi

# validate settings path
SETTINGS_MOUNT_POINT=$(realpath "$1")
if [ ! -f "$SETTINGS_MOUNT_POINT" ]; then
    echo "settings file (provided by first arugment) not found: $SETTINGS_MOUNT_POINT"
    exit 1
fi

OUTPUT_MOUNT_POINT=$(dirname "$SETTINGS_MOUNT_POINT")/output
mkdir -p "$OUTPUT_MOUNT_POINT"

shift
docker run \
    --user "$(id -u)":"$(id -g)" \
    --rm \
    --runtime=nvidia \
    --shm-size=8gb \
    --workdir=/app/code \
    -v "$SETTINGS_MOUNT_POINT":/app/settings.yaml:ro \
    -v "$DATASETS_MOUNT_POINT":/app/datasets:ro \
    -v "$CODE_MOUNT_POINT":/app/code:ro \
    -v "$OUTPUT_MOUNT_POINT":/app/output \
    mirandatz/morefun:dev_env \
    bash -c "source /venv/bin/activate && python -m morefun.experiments.cli $*"
