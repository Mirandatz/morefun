#!/bin/bash

# before running, set the following envinroment variables:
# GGE_DATASETS_PATH      path to a directory with datasets
# GGE_REPOSITORY_PATH   path to the root of the GGE git repository

# usage:
# ./cli.bash <path-to-settings.toml> <gge-python-cli-commands>

set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

CODE_MOUNT_POINT=$(realpath "$GGE_REPOSITORY_PATH")
DATASETS_MOUNT_POINT="$(realpath "$GGE_DATASETS_PATH")"

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

docker run \
    --user "$(id -u)":"$(id -g)" \
    --rm \
    --runtime=nvidia \
    --workdir=/gge/code \
    -v "$SETTINGS_MOUNT_POINT":/gge/settings.toml:ro \
    -v "$DATASETS_MOUNT_POINT":/gge/datasets:ro \
    -v "$CODE_MOUNT_POINT":/gge/code:ro \
    -v "$OUTPUT_MOUNT_POINT":/gge/output \
    mirandatz/gge:dev_env \
    python -m gge.experiments.cli "$@"
