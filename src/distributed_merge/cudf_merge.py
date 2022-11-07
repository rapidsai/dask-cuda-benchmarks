# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from enum import Enum
from itertools import chain
from typing import TYPE_CHECKING, Any, Optional, Tuple

import cupy as cp
import numpy as np
import typer
from cuda import cuda, cudart
from mpi4py import MPI
from ucp._libs import ucx_api
from ucp._libs.arr import Array

import cudf
import rmm

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


# UCP must be imported after cudaSetDevice on each rank (for correct IPC
# registration?), why?
if TYPE_CHECKING:
    import ucp
else:
    ucp = None


def format_bytes(b):
    return f"{b/1e9:.2f} GB"


class Request:
    __slots__ = ("n",)
    n: int

    def __init__(self):
        self.n = 0


class CommunicatorBase:
    def __init__(self, comm: MPI.Intracomm):
        self.mpicomm = comm
        self.rank = comm.rank
        self.size = comm.size

    def _send(self, ep, msg: "ucx_api.arr.Array", tag: int, request: Optional[Request]):
        raise NotImplementedError()

    def _recv(self, msg: "ucx_api.arr.Array", tag: int, request: Optional[Request]):
        raise NotImplementedError()

    @nvtx.annotate(domain="MERGE")
    def wireup(self):
        # Perform an all-to-all to wire up endpoints
        sendbuf = np.zeros(self.size, dtype=np.uint8)
        recvbuf = np.empty_like(sendbuf)
        request = self.ialltoall(sendbuf, recvbuf)
        self.wait(request)

    def isend(
        self,
        buf: np.ndarray,
        dest: int,
        tag: int = 0,
        request: Optional[Request] = None,
    ) -> Optional["ucx_api.UCXRequest"]:
        msg = Array(buf)
        # Tag matching to distinguish by source.
        comm_tag = (tag << 32) | self.rank
        return self._send(self.endpoints[dest], msg, comm_tag, request)

    def irecv(
        self,
        buf: np.ndarray,
        source: int,
        tag: int = 0,
        request: Optional[Request] = None,
    ) -> Optional["ucx_api.UCXRequest"]:
        msg = Array(buf)
        comm_tag = (tag << 32) | source
        return self._recv(msg, comm_tag, request)

    def __getattr__(self, name):
        try:
            return getattr(self.mpicomm, name)
        except AttributeError:
            raise AttributeError(f"No support for {name}")


class AsyncIOCommunicator(CommunicatorBase):
    @nvtx.annotate(domain="MERGE", message="AsyncIO-init")
    def __init__(self, comm: MPI.Intracomm):
        super().__init__(comm)
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        self.address = ucp.get_worker_address()
        buf = np.array(self.address)
        addresses = np.empty((comm.size, self.address.length), dtype=buf.dtype)
        comm.Allgather(buf, addresses)
        self.endpoints = self.event_loop.run_until_complete(
            asyncio.gather(
                *(
                    ucp.create_endpoint_from_worker_address(
                        ucp.get_ucx_address_from_buffer(address)
                    )
                    for address in addresses
                )
            )
        )

    @nvtx.annotate(domain="MERGE")
    def ialltoall(self, sendbuf: np.ndarray, recvbuf: np.ndarray):
        return asyncio.gather(
            *chain(
                (
                    self.irecv(recvbuf[i, ...], source=i, tag=10)
                    for i in range(self.size)
                ),
                (self.isend(sendbuf[i, ...], dest=i, tag=10) for i in range(self.size)),
            )
        )

    @nvtx.annotate(domain="MERGE")
    def ialltoallv(self, sendbuf, sendcounts, recvbuf, recvcounts):
        requests = []
        off = 0
        for i in range(self.size):
            count = recvcounts[i]
            requests.append(self.irecv(recvbuf[off : off + count], source=i, tag=10))
            off += count
        off = 0
        for i in range(self.size):
            count = sendcounts[i]
            requests.append(self.isend(sendbuf[off : off + count], dest=i, tag=10))
            off += count
        return asyncio.gather(*requests)

    def _send(self, ep, msg, tag, _request):
        return ep.send(msg, tag=tag, force_tag=True)

    def _recv(self, msg, tag, _request):
        return ucp.recv(msg, tag=tag)

    @nvtx.annotate(domain="MERGE")
    def wait(self, request):
        return self.event_loop.run_until_complete(request)

    @nvtx.annotate(domain="MERGE")
    def waitall(self, requests):
        return self.event_loop.run_until_complete(asyncio.gather(*requests))


class UCXCommunicator(CommunicatorBase):
    @staticmethod
    def _callback(ucx_req, exception, msg, req):
        assert exception is None
        req.n -= 1

    @nvtx.annotate(domain="MERGE", message="UCXPy-init")
    def __init__(self, comm: MPI.Intracomm):
        super().__init__(comm)
        ctx = ucx_api.UCXContext(feature_flags=(ucx_api.Feature.TAG,))
        self.worker = ucx_api.UCXWorker(ctx)
        self.address = self.worker.get_address()
        buf = np.array(self.address)
        addresses = np.empty((comm.size, self.address.length), dtype=buf.dtype)
        comm.Allgather(buf, addresses)
        self.endpoints = tuple(
            ucx_api.UCXEndpoint.create_from_worker_address(
                self.worker,
                ucx_api.UCXAddress.from_buffer(address),
                endpoint_error_handling=True,
            )
            for address in addresses
        )

    def _send(self, ep, msg, tag, request):
        req = request or Request()
        if (
            ucx_api.tag_send_nb(
                ep,
                msg,
                msg.nbytes,
                tag,
                cb_func=UCXCommunicator._callback,
                cb_args=(msg, req),
            )
            is not None
        ):
            req.n += 1
        return req

    def _recv(self, msg, tag, request):
        req = request or Request()
        if (
            ucx_api.tag_recv_nb(
                self.worker,
                msg,
                msg.nbytes,
                tag,
                cb_func=UCXCommunicator._callback,
                cb_args=(msg, req),
            )
            is not None
        ):
            req.n += 1
        return req

    @nvtx.annotate(domain="MERGE")
    def wait(self, request):
        while request.n > 0:
            self.worker.progress()

    @nvtx.annotate(domain="MERGE")
    def waitall(self, requests):
        while any(r.n > 0 for r in requests):
            self.worker.progress()

    @nvtx.annotate(domain="MERGE")
    def ialltoall(self, sendbuf: np.ndarray, recvbuf: np.ndarray):
        req = Request()
        for i in range(self.size):
            req = self.irecv(recvbuf[i, ...], source=i, tag=10, request=req)
        for i in range(self.size):
            req = self.isend(sendbuf[i, ...], dest=i, tag=10, request=req)
        return req

    @nvtx.annotate(domain="MERGE")
    def ialltoallv(self, sendbuf, sendcounts, recvbuf, recvcounts):
        req = Request()
        off = 0
        for i in range(self.size):
            count = recvcounts[i]
            req = self.irecv(recvbuf[off : off + count], source=i, tag=10, request=req)
            off += count
        off = 0
        for i in range(self.size):
            count = sendcounts[i]
            req = self.isend(sendbuf[off : off + count], dest=i, tag=10, request=req)
            off += count
        return req


class MPICommunicator(CommunicatorBase):
    def __init__(self, comm: MPI.Intracomm):
        self.mpicomm = comm

    def ialltoall(self, send, recv):
        return (self.mpicomm.Ialltoall(send, recv),)

    def ialltoallv(self, sendbuf, sendcounts, recvbuf, recvcounts):
        return self.mpicomm.Ialltoallv((sendbuf, sendcounts), (recvbuf, recvcounts))

    def wireup(self):
        self.mpicomm.Barrier()

    def wait(self, request):
        return MPI.Request.Wait(request)

    def waitall(self, requests):
        return MPI.Request.Waitall(requests)

    def __getattr__(self, name):
        try:
            return getattr(self.mpicomm, name)
        except AttributeError:
            raise AttributeError(f"No support for {name}")


@nvtx.annotate(domain="MERGE")
def initialize_rmm(device: int):
    # Work around cuda-python initialization bugs
    _, dev = cudart.cudaGetDevice()
    cuda.cuDevicePrimaryCtxRelease(dev)
    cuda.cuDevicePrimaryCtxReset(dev)
    cudart.cudaSetDevice(device)
    # It should be possible to just do
    # cudart.cudaSetDevice(device)
    # but this doesn't setup cudart.cudaGetDevice() correctly right now
    rmm.reinitialize(
        pool_allocator=True,
        managed_memory=False,
        devices=device,
    )
    cp.cuda.set_allocator(rmm.rmm_cupy_allocator)


@nvtx.annotate(domain="MERGE")
def build_dataframes(
    comm: CommunicatorBase,
    chunk_size: int,
    match_fraction: float,
) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    cp.random.seed(10)
    rng = cp.random
    rank = comm.rank
    size = comm.size
    start = chunk_size * rank
    stop = start + chunk_size
    left = cudf.DataFrame(
        {
            "key": cp.arange(start, stop, dtype=np.int64),
            "payload": cp.arange(start, stop, dtype=np.int64),
        }
    )

    piece_size = chunk_size // size
    piece_size_used = max(int(piece_size * match_fraction), 1)
    arrays = []
    for i in range(size):
        start = chunk_size * i + piece_size * rank
        stop = start + piece_size
        arrays.append(cp.arange(start, stop, dtype=np.int64))

    key_match = cp.concatenate(
        [rng.permutation(array)[:piece_size_used] for array in arrays], axis=0
    )
    missing = chunk_size - key_match.shape[0]
    start = chunk_size * size + chunk_size * rank
    stop = start + missing
    key_no_match = cp.arange(start, stop, dtype=np.int64)
    key = cp.concatenate([key_match, key_no_match], axis=0)
    right = cudf.DataFrame(
        {
            "key": rng.permutation(key),
            "payload": cp.arange(
                chunk_size * rank, chunk_size * (rank + 1), dtype=np.int64
            ),
        }
    )
    return (left, right)


@nvtx.annotate(domain="MERGE")
def partition_by_hash(
    df: cudf.DataFrame,
    npartitions: int,
) -> Tuple[cudf.DataFrame, np.ndarray]:
    hash_partition = cudf._lib.hash.hash_partition
    columns = ["key"]
    key_indices = [df._column_names.index(k) for k in columns]
    output_columns, offsets = hash_partition([*df._columns], key_indices, npartitions)
    out_df = cudf.DataFrame(dict(zip(df._column_names, output_columns)))
    counts = np.concatenate([np.diff(offsets), [len(out_df) - offsets[-1]]]).astype(
        np.int32
    )
    return out_df, counts


@nvtx.annotate(domain="MERGE")
def exchange_by_hash_bucket(
    comm: CommunicatorBase,
    left: cudf.DataFrame,
    right: cudf.DataFrame,
) -> Tuple[cudf.DataFrame, cudf.DataFrame]:
    left_send_df, left_sendcounts = partition_by_hash(left, comm.size)
    nvtx.push_range(domain="MERGE", message="Allocate left")
    left_recvcounts = np.zeros(comm.size, dtype=np.int32)
    comm.wait(comm.ialltoall(left_sendcounts, left_recvcounts))
    nrows = left_recvcounts.sum()
    left_recv_df = cudf.DataFrame(
        {name: cp.empty(nrows, dtype=left[name].dtype) for name in left.columns}
    )
    nvtx.pop_range(domain="MERGE")
    requests = list(
        comm.ialltoallv(
            left_send_df[name].values,
            left_sendcounts,
            left_recv_df[name].values,
            left_recvcounts,
        )
        for name in left_send_df.columns
    )
    right_send_df, right_sendcounts = partition_by_hash(right, comm.size)
    nvtx.push_range(domain="MERGE", message="Allocate right")
    right_recvcounts = np.zeros(comm.size, dtype=np.int32)
    comm.wait(comm.ialltoall(right_sendcounts, right_recvcounts))
    nrows = right_recvcounts.sum()
    right_recv_df = cudf.DataFrame(
        {name: cp.empty(nrows, dtype=right[name].dtype) for name in right.columns}
    )
    nvtx.pop_range(domain="MERGE")
    requests.extend(
        comm.ialltoallv(
            right_send_df[name].values,
            right_sendcounts,
            right_recv_df[name].values,
            right_recvcounts,
        )
        for name in right_send_df.columns
    )
    comm.waitall(requests)
    return left_recv_df, right_recv_df


@nvtx.annotate(domain="MERGE")
def distributed_join(
    comm: CommunicatorBase,
    left: cudf.DataFrame,
    right: cudf.DataFrame,
) -> cudf.DataFrame:
    left, right = exchange_by_hash_bucket(comm, left, right)
    nvtx.push_range(domain="MERGE", message="cudf_merge")
    val = left.merge(right, on="key")
    nvtx.pop_range(domain="MERGE")
    return val


def sync_print(comm: CommunicatorBase, val: Any) -> None:
    if comm.rank == 0:
        print(f"[{comm.rank}]\n{val}", flush=True)
        for source in range(1, comm.size):
            val = comm.recv(source=source)
            print(f"[{source}]\n{val}", flush=True)
    else:
        comm.send(f"{val}", dest=0)


def one_print(comm: CommunicatorBase, val: Any) -> None:
    if comm.rank == 0:
        print(f"{val}", flush=True)


@nvtx.annotate(domain="MERGE")
def bench_once(
    comm: CommunicatorBase,
    left: cudf.DataFrame,
    right: cudf.DataFrame,
) -> float:
    start = MPI.Wtime()
    _ = distributed_join(comm, left, right)
    end = MPI.Wtime()
    val = np.array(end - start, dtype=float)
    comm.Allreduce(MPI.IN_PLACE, val, op=MPI.MAX)
    return float(val)


class CommunicatorType(str, Enum):
    MPI = "mpi"
    UCXPY_ASYNC = "ucxpy-asyncio"
    UCXPY_NB = "ucxpy-nb"


def main(
    rows_per_rank: int = typer.Option(
        1000, help="Number of dataframe rows on each rank"
    ),
    match_fraction: float = typer.Option(
        0.3, help="Fraction of rows that should match"
    ),
    communicator_type: CommunicatorType = typer.Option(
        CommunicatorType.UCXPY_NB, help="Which communicator to use"
    ),
    warmup_iterations: int = typer.Option(
        2, help="Number of warmup iterations that are not benchmarked"
    ),
    iterations: int = typer.Option(10, help="Number of iterations to benchmark"),
    gpus_per_node: Optional[int] = typer.Option(
        None,
        help="Number of GPUs per node, used to assign MPI ranks to GPUs, "
        "if not provided will use cuDeviceGetCount",
    ),
):
    global ucp
    mpicomm = MPI.COMM_WORLD
    cuda.cuInit(0)
    if gpus_per_node is None:
        gpus_per_node: int
        err, gpus_per_node = cuda.cuDeviceGetCount()
        if err != 0:
            raise RuntimeError("Can't get device count")
        initialize_rmm(mpicomm.rank % gpus_per_node)
    # Must happen after initializing RMM (which sets up device contexts)
    if communicator_type != CommunicatorType.MPI:
        import ucp

        ucp.init()

    if communicator_type == CommunicatorType.UCXPY_ASYNC:
        comm = AsyncIOCommunicator(mpicomm)
    elif communicator_type == CommunicatorType.UCXPY_NB:
        comm = UCXCommunicator(mpicomm)
    elif communicator_type == CommunicatorType.MPI:
        comm = MPICommunicator(mpicomm)
    else:
        raise ValueError(f"Unsupported communicator type {communicator_type}")
    start = MPI.Wtime()
    left, right = build_dataframes(comm, rows_per_rank, match_fraction)
    end = MPI.Wtime()
    duration = np.asarray(end - start, dtype=float)
    comm.Allreduce(MPI.IN_PLACE, duration, op=MPI.MAX)
    one_print(comm, f"Dataframe build: {duration:.2g}s")
    start = MPI.Wtime()
    comm.wireup()
    end = MPI.Wtime()
    duration = np.asarray(end - start, dtype=float)
    comm.Allreduce(MPI.IN_PLACE, duration, op=MPI.MAX)
    one_print(comm, f"Wireup time: {duration:.2g}s")
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
        size = np.asarray(len(df) * sum(t.itemsize for t in df.dtypes), dtype=np.int64)
        comm.Allreduce(MPI.IN_PLACE, size, op=MPI.SUM)
        return size

    data_volume = nbytes(left) + nbytes(right)
    mean_duration = total / iterations
    throughput = data_volume / mean_duration
    one_print(comm, "Dataframe type: cudf")
    one_print(comm, f"Rows per rank: {rows_per_rank}")
    one_print(comm, f"Communicator type: {communicator_type}")
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
