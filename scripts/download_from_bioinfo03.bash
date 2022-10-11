#!/bin/bash

set -o errexit
set -o nounset
set -o xtrace
set -o pipefail

# LOCAL_GGE_DIR="/dev/shm/gge_from_bioinfo03_$(date '+%Y_%m_%d_%H_%M_%S')"
LOCAL_GGE_DIR=/home/thiago/source/gge

REMOTE=bioinfo03
REMOTE_GGE_DIR=/home/thiago/source/gge

cd "$LOCAL_GGE_DIR"

ssh "$REMOTE" " \
    cd $REMOTE_GGE_DIR \
    && git archive --format=tgz HEAD" \
    | tar -zxvf - 

