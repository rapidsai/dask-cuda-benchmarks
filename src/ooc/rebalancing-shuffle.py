# Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from collections import deque
from functools import reduce
from math import ceil
from operator import or_

import pandas as pd

import dask.dataframe as dd
from dask.dataframe.shuffle import partitioning_index, rearrange_by_column


def find_new_partition_allocation(index: pd.Series):
    """
    Given a global partitioning, try to find a rebalancing that does
    not lead to outsized (or very small) partitions.

    Parameters
    ----------
    index
        existing partitioning, providing number of values in each
        partition

    Returns
    -------
    tuple
        2-tuple of new pieces and the target output partition size
    """
    # How big do we want to be?
    target_size = int(index.mean())
    # Treat partitions from smallest to largest
    to_handle = deque(index.sort_values().items())

    new_pieces = []
    cur_size = 0
    cur_piece = []
    while to_handle:
        x = to_handle[0][1]
        if cur_size + x < target_size:
            # We can aggregate this partition to the previous ones
            # we've seen.
            cur_piece.append(to_handle.popleft())
            cur_size += x
        elif cur_piece:
            # TODO: handle glom and keep the same
            # Candidate partition is big enough, remember it
            new_pieces.append(("glom" if len(cur_piece) > 1 else "keep", cur_piece))
            cur_piece = []
            cur_size = 0
        else:
            # Now the individual partitions are already big enough
            # (keep) or too big (split).
            new_pieces.append(
                ("keep" if x < 1.5 * target_size else "split", [to_handle.popleft()])
            )
    return new_pieces, target_size


def num_new_partitions(new_pieces: list, target_size: int) -> int:
    """
    Given a new repartitioning, count the number of output partitions

    Parameters
    ----------
    new_pieces
        return value from find_new_partition_allocation
    target_size
        target partition size

    Returns
    -------
    int
        Number of partitions we will get with this scheme
    """
    i = 0
    for style, pieces in new_pieces:
        if style in {"glom", "keep"}:
            i += 1
        else:
            ((_, size),) = pieces
            i += int(ceil(size / target_size))
    return i


def redo_partition(partition, *, new_pieces, target_size, num_output_partitions):
    """
    Redo a partition given some new allocation scheme

    Parameters
    ----------
    partition
        partition to redo
    new_pieces
        return value from find_new_partition_allocation
    target_size
        target partition size
    num_output_partitions
        Number of partitions we're aiming for (sanity check)

    Returns
    -------
    Series
        New partitioning
    """
    opartition = partition
    partition = opartition.copy(deep=True)
    new_pieces = list(reversed(new_pieces))
    i = 0
    # This needs to be much faster, should figure out a way just using
    # numpy or something.
    while new_pieces:
        style, pieces = new_pieces.pop()
        if style == "glom":
            parts = set(idx for idx, _ in pieces)
            partition.loc[reduce(or_, (opartition == p for p in parts))] = i
            i += 1
        elif style == "keep":
            ((idx, _),) = pieces
            partition.loc[opartition == idx] = i
            i += 1
        else:
            assert style == "split"
            # Global size for splitting
            ((idx, size),) = pieces
            nsplit = int(ceil(size / target_size))
            bits = opartition.loc[opartition == idx]
            nvals = len(bits)
            piece = int(ceil(nvals / nsplit))
            start = 0
            end = start
            skip = 0
            while end < nvals:
                end += piece
                chunk = bits.iloc[start : min(end, nvals)].index
                partition.loc[chunk] = i
                i += 1
                skip += 1
                start = end
            i += nsplit - skip
    # Checking for off-by-ones
    assert i == num_output_partitions
    return partition


def shuffle(df: dd.DataFrame, by: str, npartitions: int):
    """
    Shuffle a dataframe on a single column

    This attempts to rebalance the output partitions such that all of
    them are approximately the same size. With a target size
    determined by len(df) / npartitions.

    Note therefore that the number of output partitions may differ
    from the provided npartitions argument.

    Parameters
    ----------
    df
        DataFrame to shuffle
    by
        Column to shuffle on
    npartitions
        Target number of output partitions

    Returns
    -------
    dd.DataFrame
        New dataframe shuffled on the provided column with
        approximately balanced partitions.
    """
    # Compute the original partitioning index, we're going to try and rebalance this.
    orig_index: dd.Series = df[by].map_partitions(
        partitioning_index,
        npartitions=npartitions,
        transform_divisions=False,
        meta=df[by]._meta,
    )

    # Count total number of contributions to each output partition on
    # each process
    index: dd.Series = orig_index.map_partitions(lambda x: x.groupby(x).count())

    # Aggregate. This is O(npartitions) so it's small.
    partition_counts: pd.Series = index.groupby(index.index).sum().compute().to_pandas()

    # Figure out how we're going to split these partition allocations up
    new_pieces, target_size = find_new_partition_allocation(partition_counts)
    # How many output partitions is this shuffle going to produce
    num_output_partitions = num_new_partitions(new_pieces, target_size)
    # Reassign the output partition id based on our rebalancing
    new_index = orig_index.map_partitions(
        redo_partition,
        new_pieces=new_pieces,
        target_size=target_size,
        num_output_partitions=num_output_partitions,
        meta=orig_index._meta,
    ).persist()

    del orig_index
    del index
    # And do the "by-hand" shuffle
    df2 = df.assign(_partitions=new_index)
    df3 = rearrange_by_column(
        df2, "_partitions", npartitions=num_output_partitions, shuffle="tasks"
    )
    del df3["_partitions"]
    return df3
