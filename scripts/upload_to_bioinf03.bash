#!/bin/bash

set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

SCRIPT_DIR="$(dirname "$(realpath -s "$0")")"
GGE_ROOT_DIR="$(realpath "$SCRIPT_DIR"/..)"

if [ ! -f "$GGE_ROOT_DIR/.gge_root" ]; then
    echo "Unable to find gge repository root dir"
    exit 1
fi

REMOTE=bioinfo03
REMOTE_GGE_DIR=/home/thiago/source/gge

ssh "$REMOTE" "rm -rf $REMOTE_GGE_DIR"

cd "$GGE_ROOT_DIR"

git archive --format=tgz HEAD . \
    | ssh "$REMOTE" "\
        mkdir -p $REMOTE_GGE_DIR \
        && cd $REMOTE_GGE_DIR \
        && tar -zxf - "
