#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

DATE=$(date +%Y%m%d)
DOCKER_HOST=gitlab-master.nvidia.com:5005
REPO=lmitchell/docker
for ucx_version in v1.12.x v1.13.x v1.14.x master; do
    TAG=${DOCKER_HOST}/${REPO}:ucx-py-${ucx_version}-${DATE}
    docker build --build-arg UCX_VERSION_TAG=${ucx_version} --no-cache -t ${TAG} -f UCXPy-rdma-core.dockerfile .
    docker push ${TAG}
done
