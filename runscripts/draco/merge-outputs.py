import glob
import os
from itertools import chain

import altair as alt
import click
import numpy as np
import pandas as pd
from altair import datum, expr
from altair.utils import sanitize_dataframe


def hmean(a):
    """Harmonic mean"""
    if len(a):
        return 1 / np.mean(1 / a)
    else:
        return 0


def hstd(a):
    """Harmonic standard deviation"""
    if len(a):
        rmean = np.mean(1 / a)
        rvar = np.var(1 / a)
        return np.sqrt(rvar / (len(a) * rmean**4))
    else:
        return 0


def remove_warmup(df):
    summary = df.groupby("num_workers")
    return df.loc[
        df.wallclock.values
        < (summary.wallclock.mean() + summary.wallclock.std() * 2)[
            df.num_workers
        ].values
    ].copy()


def process_output(directories):
    all_merge_data = []
    all_transpose_data = []
    for d in directories:
        _, date, ucx_version = d.split("/")
        date = pd.to_datetime(date)
        dfs = []
        for f in chain(
            glob.glob(os.path.join(d, "nnodes*cudf-merge-dask.json")),
            glob.glob(os.path.join(d, "nnodes*cudf-merge-explicit-comms.json")),
        ):
            merge_df = pd.read_json(f)
            dfs.append(merge_df)
        if dfs:
            merge_df = pd.concat(dfs, ignore_index=True)
            merge_df["date"] = date
            merge_df["ucx_version"] = ucx_version
            all_merge_data.append(merge_df)
        dfs = []
        for f in glob.glob(os.path.join(d, "nnodes*transpose-sum.json")):
            transpose_df = pd.read_json(f)
            dfs.append(transpose_df)
        if dfs:
            transpose_df = pd.concat(dfs, ignore_index=True)
            transpose_df["date"] = date
            transpose_df["ucx_version"] = ucx_version
            all_transpose_data.append(transpose_df)

    merge_df = pd.concat(all_merge_data, ignore_index=True)
    transpose_df = pd.concat(all_transpose_data, ignore_index=True)
    return merge_df, transpose_df


def summarise_merge_data(df):
    # data = data.groupby(["num_workers", "backend", "date"], as_index=False).mean()
    df["throughput"] = (df.data_processed / df.wallclock / df.num_workers) / 1e9
    grouped = df.groupby(["date", "num_workers", "backend", "ucx_version"])
    throughput = grouped["throughput"]
    throughput = throughput.aggregate(throughput_mean=hmean, throughput_std=hstd)
    grouped = grouped.mean().drop(columns="throughput")
    grouped = grouped.merge(
        throughput, on=["date", "num_workers", "backend", "ucx_version"]
    ).reset_index()
    tmp = grouped.loc[
        lambda df: (df.backend == "dask") & (df.ucx_version == "ucx-1.12.1")
    ].copy()
    tmp["backend"] = "no-dask"
    # distributed-joins measurements
    for n, bw in zip(
        [8, 16, 32, 64, 128, 256],
        [5.4875, 4.325, 3.56875, 2.884375, 2.090625, 1.71835937],
    ):
        tmp.loc[lambda df: df.num_workers == n, "throughput_mean"] = bw
    tmp["throughput_std"] = 0

    return pd.concat([grouped, tmp], ignore_index=True)


def summarise_transpose_data(df):
    # df = remove_warmup(df)
    df["throughput"] = (df.data_processed / df.wallclock / df.num_workers) / 1e9
    grouped = df.groupby(["date", "num_workers", "ucx_version"])
    throughput = grouped["throughput"]
    throughput = throughput.aggregate(
        throughput_mean=hmean,
        throughput_std=hstd,
        wallclock_mean="mean",
        wallclock_std="std",
    )
    grouped = grouped.mean().drop(columns="throughput")
    df = grouped.merge(
        throughput, on=["date", "num_workers", "ucx_version"]
    ).reset_index()
    return df


def make_merge_chart(df):
    data = (
        alt.Chart(df)
        .encode(
            x=alt.X("date:T", title="Date"),
        )
        .transform_calculate(category="datum.backend + '-' + datum.ucx_version")
    )
    selector = alt.selection(
        type="point", fields=["category"], bind="legend", name="selected-version"
    )

    line = data.mark_line(point=True).encode(
        y=alt.Y("throughput_mean:Q", title="Throughput [GB/s/GPU]"),
        color="category:N",
        opacity=alt.condition(selector, alt.value(1), alt.value(0.25)),
    )
    band = (
        data.mark_area()
        .transform_calculate(
            y=expr.toNumber(datum.throughput_mean)
            - expr.toNumber(datum.throughput_std),
            y2=expr.toNumber(datum.throughput_mean)
            + expr.toNumber(datum.throughput_std),
        )
        .encode(
            y="y:Q",
            y2="y2:Q",
            color="category:N",
            opacity=alt.condition(selector, alt.value(0.3), alt.value(0.025)),
        )
    )
    chart = line + band

    return chart.add_params(selector).facet(
        facet=alt.Text(
            "num_workers:N",
            title="Number of GPUs",
            # Hacky, since faceting on quantitative data is
            # not allowed? And sorting is lexicographic.
            sort=["8", "16", "32", "64", "128", "256"],
        ),
        columns=3,
    )


def make_transpose_chart(df):
    data = alt.Chart(df).encode(
        x=alt.X("date:T", title="Date"),
    )
    selector = alt.selection(
        type="point", fields=["ucx_version"], bind="legend", name="selected-version"
    )
    line = data.mark_line(point=True).encode(
        y=alt.Y("throughput_mean:Q", title="Throughput [GB/s/GPU]"),
        color="ucx_version:N",
        opacity=alt.condition(selector, alt.value(1), alt.value(0.25)),
    )
    band = (
        data.mark_area()
        .transform_calculate(
            y=expr.toNumber(datum.throughput_mean)
            - expr.toNumber(datum.throughput_std),
            y2=expr.toNumber(datum.throughput_mean)
            + expr.toNumber(datum.throughput_std),
        )
        .encode(
            y="y:Q",
            y2="y2:Q",
            opacity=alt.condition(selector, alt.value(0.3), alt.value(0.025)),
            color="ucx_version:N",
        )
    )
    chart = line + band
    return chart.add_params(selector).facet(
        facet=alt.Text(
            "num_workers:N",
            title="Number of GPUs",
            # Hacky, since faceting on quantitative data is
            # not allowed? And sorting is lexicographic.
            sort=["8", "16", "32", "64", "128", "256"],
        ),
        columns=3,
    )


@click.command()
@click.argument("merge_filename")
@click.argument("transpose_filename")
@click.option("--charts/--no-charts", type=bool, default=False, help="Make charts?")
def main(merge_filename: str, transpose_filename: str, charts: bool):
    directories = glob.glob("outputs/*/ucx-*")
    merge, transpose = process_output(directories)
    merge = summarise_merge_data(merge)
    transpose = summarise_transpose_data(transpose)
    sanitize_dataframe(merge).to_csv(merge_filename, index=False)
    sanitize_dataframe(transpose).to_csv(transpose_filename, index=False)
    if charts:
        merge = make_merge_chart(f"./{merge_filename}")
        transpose = make_transpose_chart(f"./{transpose_filename}")
        merge_basename, _ = os.path.splitext(os.path.basename(merge_filename))
        transpose_basename, _ = os.path.splitext(os.path.basename(transpose_filename))
        merge.save(f"{merge_basename}.html")
        transpose.save(f"{transpose_basename}.html")


if __name__ == "__main__":
    main()
