# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import sys
from typing import Any, Tuple

import numpy as np
import pandas as pd
import typer
from mpi4py import MPI
from pandas._libs import algos as libalgos
from pandas.core.util.hashing import hash_pandas_object

try:
    import nvtx
except ImportError:

    class nvtx:
        @staticmethod
        def noop(*args, **kwargs):
            pass

        push_range = noop
        pop_range = noop

        @staticmethod
        def annotate(*args, **kwargs):
            def noop_wrapper(fn):
                return fn

            return noop_wrapper


DataFrame = pd.DataFrame


def format_bytes(b):
    return f"{b/1e9:.2f} GB"


@nvtx.annotate(domain="MERGE")
def build_dataframes(
    comm: MPI.Intracomm,
    chunk_size: int,
    match_fraction: float,
) -> Tuple[DataFrame, DataFrame]:
    np.random.seed(10)
    rng = np.random
    rank = comm.rank
    size = comm.size
    start = chunk_size * rank
    stop = start + chunk_size
    left = pd.DataFrame(
        {
            "key": np.arange(start, stop, dtype=np.int64),
            "payload": np.arange(start, stop, dtype=np.int64),
        }
    )

    piece_size = chunk_size // size
    piece_size_used = max(int(piece_size * match_fraction), 1)
    arrays = []
    for i in range(size):
        start = chunk_size * i + piece_size * rank
        stop = start + piece_size
        arrays.append(np.arange(start, stop, dtype=np.int64))

    key_match = np.concatenate(
        [rng.permutation(array)[:piece_size_used] for array in arrays], axis=0
    )
    missing = chunk_size - key_match.shape[0]
    start = chunk_size * size + chunk_size * rank
    stop = start + missing
    key_no_match = np.arange(start, stop, dtype=np.int64)
    key = np.concatenate([key_match, key_no_match], axis=0)
    right = pd.DataFrame(
        {
            "key": rng.permutation(key),
            "payload": np.arange(
                chunk_size * rank, chunk_size * (rank + 1), dtype=np.int64
            ),
        }
    )
    return (left, right)


@nvtx.annotate(domain="MERGE")
def partition_by_hash(
    df: DataFrame,
    npartitions: int,
) -> Tuple[DataFrame, np.ndarray]:
    indexer, locs = libalgos.groupsort_indexer(
        (hash_pandas_object(df["key"], index=False) % npartitions)
        .astype(np.int32)
        .values.view()
        .astype(np.intp, copy=False),
        npartitions,
    )
    return df.take(indexer), locs[1:].astype(np.int32)


@nvtx.annotate(domain="MERGE")
def exchange_by_hash_bucket(
    comm: MPI.Intracomm,
    left: DataFrame,
    right: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    left_send_df, left_sendcounts = partition_by_hash(left, comm.size)
    nvtx.push_range(domain="MERGE", message="Allocate left")
    left_recvcounts = np.zeros(comm.size, dtype=np.int32)
    comm.Alltoall(left_sendcounts, left_recvcounts)
    nrows = left_recvcounts.sum()
    left_recv_df = pd.DataFrame(
        {name: np.empty(nrows, dtype=left[name].dtype) for name in left.columns}
    )
    nvtx.pop_range(domain="MERGE")
    requests = list(
        comm.Ialltoallv(
            (left_send_df[name].values, left_sendcounts),
            (left_recv_df[name].values, left_recvcounts),
        )
        for name in left_send_df.columns
    )
    right_send_df, right_sendcounts = partition_by_hash(right, comm.size)
    nvtx.push_range(domain="MERGE", message="Allocate right")
    right_recvcounts = np.zeros(comm.size, dtype=np.int32)
    comm.Alltoall(right_sendcounts, right_recvcounts)
    nrows = right_recvcounts.sum()
    right_recv_df = pd.DataFrame(
        {name: np.empty(nrows, dtype=right[name].dtype) for name in right.columns}
    )
    nvtx.pop_range(domain="MERGE")
    requests.extend(
        comm.Ialltoallv(
            (right_send_df[name].values, right_sendcounts),
            (right_recv_df[name].values, right_recvcounts),
        )
        for name in right_send_df.columns
    )
    MPI.Request.Waitall(requests)
    return left_recv_df, right_recv_df


@nvtx.annotate(domain="MERGE")
def distributed_join(
    comm: MPI.Intracomm,
    left: DataFrame,
    right: DataFrame,
) -> DataFrame:
    left, right = exchange_by_hash_bucket(comm, left, right)
    nvtx.push_range(domain="MERGE", message="pandas_merge")
    val = left.merge(right, on="key")
    nvtx.pop_range(domain="MERGE")
    return val


def sync_print(comm: MPI.Intracomm, val: Any) -> None:
    if comm.rank == 0:
        print(f"[{comm.rank}]\n{val}", flush=True)
        for source in range(1, comm.size):
            val = comm.recv(source=source)
            print(f"[{source}]\n{val}", flush=True)
    else:
        comm.send(f"{val}", dest=0)


def one_print(comm: MPI.Intracomm, val: Any) -> None:
    if comm.rank == 0:
        print(f"{val}", flush=True)


@nvtx.annotate(domain="MERGE")
def bench_once(
    comm: MPI.Intracomm,
    left: DataFrame,
    right: DataFrame,
) -> float:
    start = MPI.Wtime()
    _ = distributed_join(comm, left, right)
    end = MPI.Wtime()
    val = np.array(end - start, dtype=float)
    comm.Allreduce(MPI.IN_PLACE, val, op=MPI.MAX)
    return float(val)


def main(
    rows_per_rank: int = typer.Option(
        1000, help="Number of dataframe rows on each rank"
    ),
    match_fraction: float = typer.Option(
        0.3, help="Fraction of rows that should match"
    ),
    warmup_iterations: int = typer.Option(
        2, help="Number of warmup iterations that are not benchmarked"
    ),
    iterations: int = typer.Option(10, help="Number of iterations to benchmark"),
):
    comm = MPI.COMM_WORLD

    start = MPI.Wtime()
    left, right = build_dataframes(comm, rows_per_rank, match_fraction)
    end = MPI.Wtime()
    duration = comm.allreduce(end - start, op=MPI.MAX)
    one_print(comm, f"Dataframe build: {duration:.2g}s")
    for _ in range(warmup_iterations):
        bench_once(comm, left, right)
    comm.Barrier()
    total = 0

    nvtx.push_range(domain="MERGE", message="Benchmarking")
    for _ in range(iterations):
        duration = bench_once(comm, left, right)
        one_print(comm, f"Total join time: {duration:.2g}s")
        total += duration
    nvtx.pop_range(domain="MERGE")

    def nbytes(df):
        return comm.allreduce(len(df) * sum(t.itemsize for t in df.dtypes), op=MPI.SUM)

    data_volume = nbytes(left) + nbytes(right)
    mean_duration = total / iterations
    throughput = data_volume / mean_duration
    one_print(comm, "Dataframe type: pandas")
    one_print(comm, f"Rows per rank: {rows_per_rank}")
    one_print(comm, f"Data processed: {format_bytes(data_volume)}")
    one_print(comm, f"Mean join time: {mean_duration:.2g}s")
    one_print(comm, f"Throughput: {format_bytes(throughput)}/s")
    one_print(comm, f"Throughput/rank: {format_bytes(throughput/comm.size)}/s/rank")
    one_print(comm, f"Total ranks: {comm.size}")


if __name__ == "__main__":
    if "--help" in sys.argv:
        # Only print help on a single rank.
        if MPI.COMM_WORLD.rank == 0:
            typer.run(main)
    else:
        typer.run(main)
