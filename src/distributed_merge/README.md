## Overview

This implements some distributed memory joins using CUDF and Pandas
built on top of MPI and UCX-Py.

The CUDF implementation uses MPI for UCX bringup (so UCX must be
CUDA-aware, but the MPI need not be), but then the core all-to-all is
performed using UCX-Py calls.

The Pandas implementation just uses MPI.

### Dependencies

- `ucx-py`
- `cudf`
- `rmm`
- `mpi4py`
- `cupy`
- `numpy`
- `pandas`
- `typer`
- `nvtx` (optional, for hookup with [Nisght
  Systems](https://developer.nvidia.com/nsight-systems))

## Algorithm

A straightforward in-core implementation:

1. Bucket the rows of the dataframe by a deterministic hash (one
   bucket per rank)
2. Exchange sizes, and data using `MPI_Alltoall`-like and
   `MPI_Alltoallv`-like patterns respectively. For the pandas
   implementation actually uses the MPI version, for the cudf
   implementation can use MPI (if CUDA-aware), or else uses UCX
   non-blocking point to point tag send/receives.
3. Locally merge exchanged data

A more complicated approach is described in [Gao and Sakharnykh,
_Scaling Joins to a Thousand GPUs_, ADMS
2021](http://www.adms-conf.org/2021-camera-ready/gao_adms21.pdf).
