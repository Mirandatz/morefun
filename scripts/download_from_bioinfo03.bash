#!/bin/bash

set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

REMOTE=bioinfo03
REMOTE_GGE_DIR=/home/thiago/source/gge

DOWNLOAD_DIR=$(date '+%Y_%m_%d_%H_%M_%S')
mkdir "$DOWNLOAD_DIR"

rsync -av \
    --exclude '.mypy_cache/*' \
    --exclude '*.pyc' \
    "$REMOTE:$REMOTE_GGE_DIR" \
    "$DOWNLOAD_DIR"
