#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

source /opt/conda/etc/profile.d/conda.sh
source /opt/conda/etc/profile.d/mamba.sh
mamba activate ucx

SCHED_FILE=${SCRATCHDIR}/scheduler-${SLURM_JOBID}.json
if [[ $SLURM_PROCID == 0 ]]; then
       echo "******* UCX INFORMATION *********"
       ucx_info -v
fi


NGPUS=$(nvidia-smi -L | wc -l)
export EXPECTED_NUM_WORKERS=$((SLURM_JOB_NUM_NODES * NGPUS))

export UCX_HANDLE_ERRORS=bt,freeze
# https://github.com/openucx/ucx/issues/8639
export UCX_RNDV_SCHEME=get_zcopy
export PROTOCOL=ucx
# FIXME is the interface correct?
export COMMON_ARGS="--protocol ${PROTOCOL} \
       --interface ibp5s0 \
       --scheduler-file ${SCRATCHDIR}/scheduler-${SLURM_JOBID}.json"
export PROTOCOL_ARGS=""
export WORKER_ARGS="--local-directory /tmp/dask-${SLURM_PROCID} \
       --multiprocessing-method forkserver"

export PTXCOMPILER_CHECK_NUMBA_CODEGEN_PATCH_NEEDED=0
# Still needed?
export UCX_MEMTYPE_CACHE=n

NUM_WORKERS=$(printf "%03d" ${EXPECTED_NUM_WORKERS})
UCX_VERSION=$(python -c "import ucp; print('.'.join(map(str, ucp.get_ucx_version())))")
OUTPUT_DIR=${OUTDIR}/ucx-${UCX_VERSION}

# Idea: we allocate ntasks-per-node for workers, but those are started in the
# background by dask-cuda-worker.
# So we need to pick one process per node to run the worker commands.
# This assumes that the mapping from nodes to ranks is dense and contiguous. If
# there is rank-remapping then something more complicated would be needed.
if [[ $(((SLURM_PROCID / SLURM_NTASKS_PER_NODE) * SLURM_NTASKS_PER_NODE)) == ${SLURM_PROCID} ]]; then
    # rank zero starts scheduler and client as well
    if [[ $SLURM_NODEID == 0 ]]; then
        echo "Environment status"
        mkdir -p $OUTPUT_DIR
        python ${RUNDIR}/get-versions.py ${OUTPUT_DIR}/version-info.json
        mamba list --json > ${OUTPUT_DIR}/environment-info.json
        echo "${SLURM_PROCID} on node ${SLURM_NODEID} starting scheduler/client"
        dask scheduler \
               --no-dashboard \
               ${COMMON_ARGS} &
        sleep 6
        dask cuda worker \
               --no-dashboard \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               ${WORKER_ARGS} &
        # Weak scaling
        # Only the first run initializes the RMM pool, which is then set up on
        # workers. After that the clients connect to workers with a pool already
        # in place, so we pass --disable-rmm-pool
        python -m dask_cuda.benchmarks.local_cudf_merge \
               -c 40_000_000 \
               --frac-match 0.6 \
               --runs 30 \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
	        --backend dask \
               --output-basename ${OUTPUT_DIR}/nnodes-${NUM_WORKERS}-cudf-merge-dask \
               --multiprocessing-method forkserver \
               --no-show-p2p-bandwidth \
            || /bin/true        # always exit cleanly
        python ${RUNDIR}/gc-workers.py ${SCHED_FILE} || /bin/true
        python -m dask_cuda.benchmarks.local_cudf_merge \
               -c 40_000_000 \
               --frac-match 0.6 \
               --runs 30 \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               --disable-rmm-pool \
               --backend dask-noop \
               --no-show-p2p-bandwidth \
               --output-basename ${OUTPUT_DIR}/nnodes-${NUM_WORKERS}-cudf-merge-dask-noop \
               --multiprocessing-method forkserver \
            || /bin/true        # always exit cleanly
        python ${RUNDIR}/gc-workers.py ${SCHED_FILE} || /bin/true
        python -m dask_cuda.benchmarks.local_cudf_merge \
               -c 40_000_000 \
               --frac-match 0.6 \
               --runs 30 \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               --disable-rmm-pool \
	        --backend explicit-comms \
               --output-basename ${OUTPUT_DIR}/nnodes-${NUM_WORKERS}-cudf-merge-explicit-comms \
               --multiprocessing-method forkserver \
               --no-show-p2p-bandwidth \
            || /bin/true        # always exit cleanly
        python ${RUNDIR}/gc-workers.py ${SCHED_FILE} || /bin/true
        python -m dask_cuda.benchmarks.local_cupy \
               -o transpose_sum \
               -s 50000 \
               -c 2500 \
               --runs 30 \
               --disable-rmm-pool \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               --output-basename ${OUTPUT_DIR}/nnodes-${NUM_WORKERS}-transpose-sum \
               --multiprocessing-method forkserver \
               --no-show-p2p-bandwidth \
            || /bin/true
        python ${RUNDIR}/gc-workers.py ${SCHED_FILE} || /bin/true
        # Strong scaling
        python -m dask_cuda.benchmarks.local_cupy \
               -o transpose_sum \
               -s 50000 \
               -c 2500 \
               --runs 30 \
               --disable-rmm-pool \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               --backend dask-noop \
               --no-show-p2p-bandwidth \
               --shutdown-external-cluster-on-exit \
               --output-basename ${OUTPUT_DIR}/nnodes-${NUM_WORKERS}-transpose-sum-noop \
               --multiprocessing-method forkserver \
            || /bin/true
    else
        echo "${SLURM_PROCID} on node ${SLURM_NODEID} starting worker"
        sleep 6
        dask cuda worker \
               --no-dashboard \
               ${COMMON_ARGS} \
               ${PROTOCOL_ARGS} \
               ${WORKER_ARGS} \
            || /bin/true        # always exit cleanly
    fi
else
    echo "${SLURM_PROCID} on node ${SLURM_NODEID} sitting in background"
fi
