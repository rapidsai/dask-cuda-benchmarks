# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time
from functools import partial
from pathlib import Path

import more_itertools
import numpy as np
import pyarrow as pa
import typer

import cudf
import cudf._lib as libcudf
import rmm

# Exploring out-of-core shuffle (disk to disk) via parquet
# We use pyarrow for writing since we can keep the files open
# and do partial writes (caveat, can't put pandas metadata in, but
# that's ok because it's not storing anything useful).
# Algorithm:
# Read n parquet files at a time into device memory and concat
#   Build hash key reordered dataframe (hash_partition)
#   Stream reordered dataframe back to host
#   Slice host dataframe
#   Accumulate host slices in staging buffers
# Every p parquet files (tunable)
#   Concat and write (append) staging buffers to files
# Done
# This has excellent control over peak device memory usage during
# shuffle (guaranteed to be 2x input dataframe size);
# and host memory usage (guaranteed to be p/n x input dataframe size)
#
# To avoid excessively large output partitions we could do a pre-pass
# hashing everything to determine which (if any) key partitions are
# excessively large and then come up with a splitting rule
#
# There's some magic to handle a single large output partition.


def partition(
    df: cudf.DataFrame,
    columns: list[str],
    npartitions: int,
    *,
    split_largest_by: int | None = None,
) -> tuple[cudf.DataFrame, np.ndarray]:
    start = time.time()
    key_indices = [df._column_names.index(k) for k in columns]
    output_columns, offsets = libcudf.hash.hash_partition(
        [*df._columns], key_indices, npartitions
    )
    out = cudf.DataFrame(dict(zip(df._column_names, output_columns)))
    offsets = np.concatenate([offsets, [len(out)]])
    if split_largest_by is not None:
        part_sizes = np.diff(offsets)
        largest = part_sizes.max()
        (index,) = np.where(part_sizes == largest)[0]
        before = offsets[:index]
        after = offsets[index + 1 :]
        mid = offsets[index] + (largest // split_largest_by) * np.arange(
            split_largest_by, dtype=offsets.dtype
        )
        offsets = np.concatenate([before, mid, after])
    print(f"Device partition took: {time.time() - start:.2f}s")
    return out, offsets


def build_host_pieces(df: cudf.DataFrame, offsets: np.ndarray) -> list[pa.Table]:
    start = time.time()
    tab = df.to_arrow()
    val = [tab[s:e] for s, e in zip(offsets, offsets[1:])]
    bytes = df.memory_usage().sum()
    duration = time.time() - start
    bw = bytes / duration / (1024**3)
    print(f"Device to host took {duration:.2f}s [{bw:.2f} GiB/s]")
    return val


# Can't use this because cudf's to_arrow() doesn't produce "string not
# null" schema entries.
def check_schema_consistent(input_files: list[Path]) -> pa.Schema:
    schema, *rest = (pa.parquet.ParquetFile(f).schema for f in input_files)
    if any(s != schema for s in rest):
        raise ValueError("All parquet files must advertise identical schema")
    schema = schema.to_arrow_schema()
    metadata = schema.metadata.copy()
    # Remove pandas-based metadata since we can't preserve the correct
    # information (index size), and it is unnecessary anyway.
    metadata.pop(b"pandas", None)
    return schema.with_metadata(metadata)


def read_inputs(files: list[Path]) -> cudf.DataFrame:
    start = time.time()
    val = cudf.concat([cudf.read_parquet(f) for f in files], ignore_index=True)
    bytes = val.memory_usage().sum()
    duration = time.time() - start
    bw = bytes / duration / (1024**3)
    print(f"Read input took: {duration:.2f}s [{bw:.2f} GiB/s]")
    return val


def write_files(
    files_chunks: list[tuple[pa.parquet.ParquetWriter, pa.Table]], nthread: int
):
    def write_pieces(fchunks):
        for f, tab in fchunks:
            f.write_table(tab)

    n = len(files_chunks)
    chunk = n // (nthread - 1)
    threads = [
        threading.Thread(
            target=partial(
                write_pieces, files_chunks[min(i * chunk, n) : min((i + 1) * chunk, n)]
            ),
        )
        for i in range(nthread)
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def read_and_stage_partitions(
    input_files: list[Path],
    output_directory: Path,
    num_writer_threads,
    device_chunk_size: int,
    host_chunk_size: int,
    columns: list[str],
    npartitions: int,
    split_largest_by: int | None = None,
):
    start = time.time()
    # We don't use this one.
    schema = check_schema_consistent(input_files)

    if split_largest_by is not None:
        noutput_files = npartitions + split_largest_by - 1
    else:
        noutput_files = npartitions

    output_directory.mkdir(exist_ok=True)
    output_files: list[pa.parquet.ParquetWriter | None] = [None] * noutput_files

    for host_chunk in more_itertools.chunked(input_files, host_chunk_size):
        output_chunks = [[] for _ in range(noutput_files)]
        for files in more_itertools.chunked(host_chunk, device_chunk_size):
            host_pieces = build_host_pieces(
                *partition(
                    read_inputs(files),
                    columns,
                    npartitions,
                    split_largest_by=split_largest_by,
                )
            )
            for chunk, piece in zip(output_chunks, host_pieces):
                chunk.append(piece)
        s = time.time()
        files_chunks = []
        bytes = 0
        for i, chunk in enumerate(output_chunks):
            f = output_files[i]
            tab = pa.concat_tables(chunk)
            bytes += tab.nbytes
            if f is None:
                schema = tab.schema
                metadata = schema.metadata.copy()
                # Remove pandas-based metadata since we can't preserve the correct
                # information (index size), and it is unnecessary anyway.
                metadata.pop(b"pandas", None)
                f = pa.parquet.ParquetWriter(
                    str(output_directory / f"part.{i}.parquet"),
                    schema=schema.with_metadata(metadata),
                )
                output_files[i] = f
            files_chunks.append((f, tab))
        write_files(files_chunks, num_writer_threads)
        duration = time.time() - s
        bw = bytes / duration / (1024**3)
        print(f"Writing host chunks took: {duration:.2f}s [{bw:.2f} GiB/s]")
    for f in output_files:
        if f is not None:
            f.close()
    print(f"Shuffle took: {time.time() - start:.2f}s")


def main(
    input_dir: str = typer.Argument(..., help="Input parquet directory"),
    output_dir: str = typer.Argument(..., help="Output parquet directory"),
    num_writer_threads: int = typer.Option(40, help="Number of threads for writing"),
    num_device_chunks: int = typer.Option(
        5, help="Number of input parquet files to concat on device before partitioning"
    ),
    num_host_chunks: int = typer.Option(
        20,
        help=(
            "Number of input parquet files to stage on host "
            "before writing to disk (should be larger than num_device_chunks"
        ),
    ),
    npartitions: int = typer.Option(313, help="Target number of output partitions"),
    split_largest_by: int = typer.Option(
        15, help="Magic number to split largest partition by"
    ),
):
    rmm.reinitialize(pool_allocator=True)

    input_files = list(Path(input_dir).glob("part.*.parquet"))
    read_and_stage_partitions(
        input_files,
        Path(output_dir),
        num_writer_threads,
        num_device_chunks,  # Number of parquet files to read and concat on GPU
        num_host_chunks,  # Number of parquet files to stage on CPU before writing
        ["ss_sold_date_sk"],
        npartitions,
        split_largest_by=split_largest_by,  # Magic number that splits
        # nulls in input parquet
        # to the mean partition
        # size
    )


if __name__ == "__main__":
    typer.run(main)
