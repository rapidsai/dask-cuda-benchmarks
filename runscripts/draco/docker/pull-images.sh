#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
DATE=$(date +%Y%m%d)
DOCKER_HOST=gitlab-master.nvidia.com
REPO=lmitchell/docker
OUTPUT_DIR=$(readlink -f ~/workdir/enroot-images)
for ucx_version in v1.12.x v1.13.x v1.14.x master; do
    TAG=${DOCKER_HOST}\#${REPO}:ucx-py-${ucx_version}-${DATE}
    srun -p interactive_dgx1_m2 -t 00:30:00 -A sw_rapids_testing \
        --nv-meta=ml-model.rapids-debug --gpus-per-node 0 --nodes 1  \
        --exclusive \
        enroot import -o ${OUTPUT_DIR}/ucx-py-${ucx_version}-${DATE}.sqsh docker://${TAG}
done
