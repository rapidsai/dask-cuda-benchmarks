#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -ex
DOCKER_BUILD_SERVER=$1
DOCKER_BUILD_DIRECTORY=$3
JOB_SUBMISSION_SERVER=$2
JOB_SUBMISSION_DIRECTORY=$4

ssh ${DOCKER_BUILD_SERVER} "(cd ${DOCKER_BUILD_DIRECTORY}; ./build-images.sh)"
ssh ${JOB_SUBMISSION_SERVER} "(cd ~/${JOB_SUBMISSION_DIRECTORY}/docker; ./pull-images.sh)"
ssh ${JOB_SUBMISSION_SERVER} "(cd ~/${JOB_SUBMISSION_DIRECTORY}; for n in 1 2 4 8 16; do sbatch --nodes \$n job.slurm; done)"
