#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
set -ex

CONDA_HOME=${1:-"/opt/conda"}
CONDA_ENV=${2:-"ucx"}

source ${CONDA_HOME}/etc/profile.d/conda.sh
source ${CONDA_HOME}/etc/profile.d/mamba.sh
mamba activate ${CONDA_ENV}

git clone https://github.com/gjoseph92/dask-noop.git
pip install --no-deps dask-noop/
