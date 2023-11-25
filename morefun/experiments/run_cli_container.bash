#!/bin/bash

# before running, set the following envinroment variables:
# MOREFUN_DATASETS_PATH   path to a directory with datasets
# MOREFUN_REPOSITORY_PATH path to the root of the MOREFUN git repository

# usage:
# ./cli.bash <path-to-settings.yaml> <morefun-python-cli-commands>

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

function find_project_root_dir {
    root_identifier="$1"

    script_path=$(realpath "$0")
    current_dir=$(dirname "$script_path")

    while true
    do
        if [ -f "${current_dir}/${root_identifier}" ]; then
            echo "$current_dir"
            return 0
        fi

        if [ "$current_dir" == "/" ]; then
            return 1
        fi
        
        current_dir=$(realpath "${current_dir}/..")
    done
}

if [ $# -eq 0 ]; then
    echo "this script expects at least one argument, the path of the settings.yaml file"
    exit 1
fi

code_mount_point=$(find_project_root_dir ".morefun_root")
# ensuring directories exist so they are no created (as root) by docker
if [ ! -d "$code_mount_point" ]; then
    echo "envvar 'code_mount_point' does not point to a directory: $code_mount_point"
    exit 1
fi

datasets_mount_point="$(realpath "$MOREFUN_DATASETS_DIR")"
if [ ! -d "$datasets_mount_point" ]; then
    echo "envvar 'datasets_mount_point' does not point to a directory: $datasets_mount_point"
    exit 1
fi

settings_mount_point=$(realpath "$1")
if [ ! -f "$settings_mount_point" ]; then
    echo "settings file (provided by first argument) not found: $settings_mount_point"
    exit 1
fi

# ensure output dir exists, so it is not created (as root) by docker
output_mount_point=$(dirname "$settings_mount_point")/output
mkdir -p "$output_mount_point"

shift
docker run \
    --user "$(id -u)":"$(id -g)" \
    --rm \
    --runtime=nvidia \
    --shm-size=8gb \
    --workdir=/app/code \
    -v "$settings_mount_point":/app/settings.yaml:ro \
    -v "$datasets_mount_point":/app/datasets:ro \
    -v "$code_mount_point":/app/code:ro \
    -v "$output_mount_point":/app/output \
    mirandatz/morefun:dev_env \
    bash -c "source /app/.venv/bin/activate && python -m morefun.experiments.cli $*"
