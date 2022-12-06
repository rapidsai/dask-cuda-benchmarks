#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -ex

SINGLE_NODE_REMOTE_LOCATION=$1
MULTI_NODE_REMOTE_LOCATION=$2
LOCAL_DATA_LOCATION=$3
WEBSITE_DIRECTORY=$4

rsync -rvupm ${SINGLE_NODE_REMOTE_LOCATION} ${LOCAL_DATA_LOCATION}/single-node \
      --filter '+ */' \
      --filter '+ local*.log' \
      --filter '+ ucx-py-bandwidth.csv' \
      --filter '- *' \
      --max-size=50K

if [[ ! -d ${WEBSITE_DIRECTORY} ]]; then
    echo "Output directory ${WEBSITE_DIRECTORY} not found!"
    exit 1
fi

LOC=$(dirname $0)
python ${LOC}/make-single-node-charts.py ${LOCAL_DATA_LOCATION}/single-node/ ${WEBSITE_DIRECTORY} --make-charts

rsync -rvupm ${MULTI_NODE_REMOTE_LOCATION} ${LOCAL_DATA_LOCATION}/multi-node/ \
      --filter '+ */' \
      --filter '+ *.json' \
      --filter '- *'

python ${LOC}/make-multi-node-charts.py ${LOCAL_DATA_LOCATION}/multi-node/ ${WEBSITE_DIRECTORY} --make-charts
