# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
from collections.abc import Callable
from functools import partial
from itertools import chain
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import cast
from warnings import warn

import altair as alt
import numpy as np
import pandas as pd
import typer
from altair import datum
from altair.utils import sanitize_dataframe

from dask.utils import parse_bytes, parse_timedelta


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


def parse_filename(filename: Path):
    splits = set(filename.stem.split("_"))
    if "ucx" in splits:
        return {
            "protocol": "ucx",
            "nvlink": "nvlink" in splits,
            "infiniband": "ib" in splits,
            "tcp": "tcp" in splits,
        }
    else:
        assert "tcp" in splits
        return {"protocol": "tcp", "nvlink": False, "infiniband": False, "tcp": True}


def parse_merge(dirname: Path):
    filenames = sorted(dirname.glob("local_cudf_merge*.log"))
    series = []
    try:
        df = pd.read_csv(dirname / "ucx-py-bandwidth.csv")
        (numpy_version,) = set(df["NumPy Version"])
        (cupy_version,) = set(df["CuPy Version"])
        (rmm_version,) = set(df["RMM Version"])
        (ucx_version,) = set(df["UCX Version"])
        (ucxpy_version,) = set(df["UCX-Py Version"])
        (ucx_revision,) = set(df["UCX Revision"])
        version_info = {
            "numpy_version": numpy_version,
            "cupy_version": cupy_version,
            "rmm_version": rmm_version,
            "ucx_version": ucx_version,
            "ucxpy_version": ucxpy_version,
            "ucx_revision": ucx_revision,
        }
    except (FileNotFoundError, ValueError):
        version_info = {
            "numpy_version": None,
            "cupy_version": None,
            "rmm_version": None,
            "ucx_version": None,
            "ucxpy_version": None,
            "ucx_revision": None,
        }
    for filename in filenames:
        with open(filename, "r") as f:
            fileinfo = parse_filename(filename)
            data = f.read()
            start = data.find("Merge benchmark")
            if start < 0:
                warn(f"Can't parse data for {filename}")
                continue
            data = data[start:].strip().split("\n")
            info = []
            _, _, *data = data
            data = iter(data)
            for line in data:
                if line.startswith("="):
                    break
                header, val = line.split("|")
                info.append((header, val))
            line, *data = data
            if line.startswith("Wall-clock") or line.startswith("Wall clock"):
                _, *data = data
            else:
                raise RuntimeError(f"Invalid file format {filename}")
            data = iter(data)
            for line in data:
                if line.startswith("="):
                    break
                header, val = line.split("|")
                info.append((header, val))
            line, *data = data
            if line.startswith("Throughput"):
                line, *data = data
                if line.startswith("Bandwidth"):  # New format
                    line, *data = data
                    assert line.startswith("Wall clock"), filename
                else:
                    assert line.startswith("Wall-Clock") or line.startswith(
                        "Wall clock"
                    ), filename
                line, *data = data
                assert line.startswith("=")
                line, *data = data
            assert line.startswith("(w1,w2)")
            _, *data = data
            for line in data:
                if line.startswith("Worker index"):
                    break
                try:
                    header, val = line.split("|")
                    info.append((header, val))
                except ValueError:
                    continue
        mangled_info = []
        name_map = {
            "backend": (lambda a: "backend", str),
            "merge type": (lambda a: "merge_type", str),
            "rows-per-chunk": (lambda a: "rows_per_chunk", int),
            "base-chunks": (lambda a: "base_chunks", int),
            "other-chunks": (lambda a: "other_chunks", int),
            "broadcast": (lambda a: "broadcast", str),
            "protocol": (lambda a: "protocol", lambda a: a),
            "device(s)": (lambda a: "devices", lambda a: tuple(map(int, a.split(",")))),
            "rmm-pool": (lambda a: "rmm_pool", ast.literal_eval),
            "frac-match": (lambda a: "frac_match", float),
            "tcp": (lambda a: "tcp", str),
            "ib": (lambda a: "ib", str),
            "infiniband": (lambda a: "ib", str),
            "nvlink": (lambda a: "nvlink", str),
            "data-processed": (lambda a: "data_processed", parse_bytes),
            "data processed": (lambda a: "data_processed", parse_bytes),
            "rmm pool": (lambda a: "rmm_pool", ast.literal_eval),
            "worker thread(s)": (lambda a: "worker_threads", int),
            "number of workers": (lambda a: "num_workers", int),
        }
        wallclocks = []
        bandwidths = []
        for name, val in info:
            name = name.strip().lower()
            val = val.strip()
            if name in {"tcp", "ib", "nvlink", "infiniband"}:
                # Get these from the filename
                continue
            try:
                mangle_name, mangle_val = name_map[name]
                name = mangle_name(name)
                val = mangle_val(val)
                mangled_info.append((name, val))
            except KeyError:
                if name.startswith("("):
                    source, dest = map(int, name[1:-1].split(","))
                    *bw_quartiles, data_volume = val.split("/s")
                    bw_quartiles = tuple(map(parse_bytes, bw_quartiles))
                    data_volume = parse_bytes(data_volume.strip()[1:-1])
                    bw = [
                        ("source_device", source),
                        ("destination_device", dest),
                        ("data_volume", data_volume),
                    ]
                    for n, q in zip(
                        (
                            "bandwidth_quartile_25",
                            "bandwidth_quartile_50",
                            "bandwidth_quartile_75",
                        ),
                        bw_quartiles,
                    ):
                        bw.append((n, q))
                    bandwidths.append(bw)
                else:
                    wallclocks.append((parse_timedelta(name), parse_bytes(val[:-2])))
        wallclocks = np.asarray(wallclocks)
        num_gpus = 8
        wallclocks[:, 1] /= num_gpus
        mangled_info.append(("wallclock_mean", np.mean(wallclocks[:, 0])))
        mangled_info.append(("wallclock_std", np.std(wallclocks[:, 0])))
        mangled_info.append(("throughput_mean", hmean(wallclocks[:, 1])))
        mangled_info.append(("throughput_std", hstd(wallclocks[:, 1])))
        mangled_info.append(("nreps", len(wallclocks)))
        date, _ = re.match(".*(202[0-9]{5})([0-9]{4}).*", str(filename)).groups()
        date = pd.to_datetime(f"{date}")
        mangled_info.append(("timestamp", date))
        mangled_info = dict(mangled_info)
        assert mangled_info["protocol"] == fileinfo["protocol"]
        mangled_info.update(fileinfo)
        mangled_info.update(version_info)
        series.append(pd.Series(mangled_info))
        # for bw in bandwidths:
        #     series.append(pd.Series(mangled_info | dict(bw)))
    return series


def parse_transpose(dirname: Path):
    filenames = sorted(dirname.glob("local_cupy_transpose*.log"))
    series = []
    try:
        df = pd.read_csv(dirname / "ucx-py-bandwidth.csv")
        (numpy_version,) = set(df["NumPy Version"])
        (cupy_version,) = set(df["CuPy Version"])
        (rmm_version,) = set(df["RMM Version"])
        (ucx_version,) = set(df["UCX Version"])
        (ucxpy_version,) = set(df["UCX-Py Version"])
        (ucx_revision,) = set(df["UCX Revision"])
        version_info = {
            "numpy_version": numpy_version,
            "cupy_version": cupy_version,
            "rmm_version": rmm_version,
            "ucx_version": ucx_version,
            "ucxpy_version": ucxpy_version,
            "ucx_revision": ucx_revision,
        }
    except (FileNotFoundError, ValueError):
        version_info = {
            "numpy_version": None,
            "cupy_version": None,
            "rmm_version": None,
            "ucx_version": None,
            "ucxpy_version": None,
            "ucx_revision": None,
        }
    for filename in filenames:
        old_format = True
        with open(filename, "r") as f:
            fileinfo = parse_filename(filename)
            data = f.read()
            start = data.find("Roundtrip benchmark")
            if start < 0:
                warn(f"Can't parse data for {filename}")
                continue
            data = data[start:].strip().split("\n")
            info = []
            _, _, *data = data
            data = iter(data)
            for line in data:
                if line.startswith("="):
                    break
                header, val = line.split("|")
                info.append((header, val))
            line, *data = data
            if line.startswith("Wall-clock") or line.startswith("Wall clock"):
                _, x = line.split("|")
                if x.strip().lower() == "throughput":
                    old_format = False
                _, *data = data
            else:
                raise RuntimeError(f"Invalid file format {filename}")
            data = iter(data)
            for line in data:
                if line.startswith("="):
                    break
                header, val = line.split("|")
                info.append((header, val))
            line, *data = data
            if old_format:
                assert line.startswith("(w1,w2)")
                _, *data = data
            else:
                assert line.startswith("Throughput")
                line, *data = data
                assert line.startswith("Bandwidth")
                line, *data = data
                assert line.startswith("Wall clock")
                line, *data = data
                assert line.startswith("=")
                line, *data = data
                assert line.startswith("(w1,w2)")
                _, *data = data
            for line in data:
                if line.startswith("Worker index"):
                    break
                try:
                    header, val = line.split("|")
                    info.append((header, val))
                except ValueError:
                    continue
        mangled_info = []
        name_map = {
            "operation": (lambda a: "operation", lambda a: a),
            "backend": (lambda a: "backend", lambda a: a),
            "array type": (lambda a: "array_type", lambda a: a),
            "user size": (lambda a: "user_size", int),
            "user second size": (lambda a: "user_second_size", int),
            "user chunk-size": (lambda a: "user_chunk_size", int),
            "user chunk size": (lambda a: "user_chunk_size", int),
            # TODO, what to do with these tuples?
            "compute shape": (
                lambda a: "compute_shape",
                lambda a: tuple(map(int, a[1:-1].split(","))),
            ),
            "compute chunk-size": (
                lambda a: "compute_chunk_size",
                lambda a: tuple(map(int, a[1:-1].split(","))),
            ),
            "compute chunk size": (
                lambda a: "compute_chunk_size",
                lambda a: tuple(map(int, a[1:-1].split(","))),
            ),
            "tcp": (lambda a: "tcp", str),
            "ib": (lambda a: "ib", str),
            "infiniband": (lambda a: "ib", str),
            "nvlink": (lambda a: "nvlink", str),
            "ignore-size": (lambda a: "ignore_size", parse_bytes),
            "ignore size": (lambda a: "ignore_size", parse_bytes),
            "data processed": (lambda a: "data_processed", parse_bytes),
            "rmm pool": (lambda a: "rmm_pool", ast.literal_eval),
            "protocol": (lambda a: "protocol", lambda a: a),
            "device(s)": (lambda a: "devices", lambda a: tuple(map(int, a.split(",")))),
            "worker thread(s)": (
                lambda a: "worker_threads",
                lambda a: tuple(map(int, a.split(","))),
            ),
            "number of workers": (lambda a: "num_workers", int),
        }
        wallclocks = []
        bandwidths = []
        for name, val in info:
            name = name.strip().lower()
            val = val.strip()
            if name in {"tcp", "ib", "nvlink", "infiniband"}:
                # Get these from the filename
                continue
            try:
                mangle_name, mangle_val = name_map[name]
                name = mangle_name(name)
                val = mangle_val(val)
                mangled_info.append((name, val))
            except KeyError:
                if name.startswith("("):
                    source, dest = map(int, name[1:-1].split(","))
                    *bw_quartiles, data_volume = val.split("/s")
                    bw_quartiles = tuple(map(parse_bytes, bw_quartiles))
                    data_volume = parse_bytes(data_volume.strip()[1:-1])
                    bw = [
                        ("source_device", source),
                        ("destination_device", dest),
                        ("data_volume", data_volume),
                    ]
                    for n, q in zip(
                        (
                            "bandwidth_quartile_25",
                            "bandwidth_quartile_50",
                            "bandwidth_quartile_75",
                        ),
                        bw_quartiles,
                    ):
                        bw.append((n, q))
                    bandwidths.append(bw)
                else:
                    wallclocks.append(parse_timedelta(name))
                    if old_format:
                        assert int(val) == 100
        wallclocks = np.asarray(wallclocks)
        rows = dict(mangled_info)["user_size"]
        data_volume = rows * rows * 8
        mangled_info.append(("data_processed", data_volume))
        m = wallclocks.mean()
        s = wallclocks.std()
        num_gpus = 8
        mangled_info.append(("wallclock_mean", m))
        mangled_info.append(("wallclock_std", s))
        mangled_info.append(("throughput_mean", (data_volume / num_gpus) / m))
        mangled_info.append(
            ("throughput_std", (data_volume / num_gpus) * s / (m * (m + s)))
        )
        mangled_info.append(("nreps", len(wallclocks)))
        date, _ = re.match(".*(202[0-9]{5})([0-9]{4}).*", str(filename)).groups()
        date = pd.to_datetime(f"{date}")
        mangled_info.append(("timestamp", date))
        mangled_info = dict(mangled_info)
        assert mangled_info["protocol"] == fileinfo["protocol"]
        mangled_info.update(fileinfo)
        mangled_info.update(version_info)
        series.append(pd.Series(mangled_info))
        # for bw in bandwidths:
        #     series.append(pd.Series(mangled_info | dict(bw)))
    return series


def get_merge_metadata(df):
    candidates = [
        "backend",
        "merge_type",
        "rows_per_chunk",
        "devices",
        "rmm_pool",
        "frac_match",
        "nreps",
    ]
    meta = {}
    for candidate in candidates:
        try:
            (val,) = set(df[candidate])
            meta[candidate] = val
        except ValueError:
            continue
    for key in meta:
        del df[key]
    return df, meta


def get_transpose_metadata(df):
    candidates = [
        "operation",
        "user_size",
        "user_second_size",
        "compute_shape",
        "compute_chunk_size",
        "devices",
        "worker_threads",
        "nreps",
    ]
    meta = {}
    for candidate in candidates:
        try:
            (val,) = set(df[candidate])
            meta[candidate] = val
        except ValueError:
            continue
    for key in meta:
        del df[key]
    return df, meta


def is_new_data(p, known_dates):
    return p.is_dir() and p.parent.name[:8] not in known_dates


def get_results(
    data_directory: Path,
    csv_name: Path,
    meta_name: Path,
    parser: Callable[[Path], list[pd.Series]],
    extract_metadata: Callable[[pd.DataFrame], tuple[pd.DataFrame, dict]],
) -> tuple[pd.DataFrame, dict]:
    if csv_name.exists():
        existing = pd.read_csv(csv_name)
        existing["timestamp"] = existing.timestamp.astype(np.datetime64)
        known_dates = set(
            map(
                methodcaller("strftime", "%Y%m%d"),
                pd.DatetimeIndex(existing.timestamp).date,
            )
        )
    else:
        known_dates = set()
        existing = None
    if meta_name.exists():
        with open(meta_name, "r") as f:
            meta = json.load(f)
    else:
        meta = None

    df = pd.DataFrame(
        chain.from_iterable(
            map(
                parser,
                sorted(
                    filter(
                        partial(is_new_data, known_dates=known_dates),
                        data_directory.glob("*/*/"),
                    )
                ),
            )
        )
    )
    if not df.empty:
        df, meta = extract_metadata(df)
    if existing is not None:
        df = pd.concat([existing, df]).sort_values("timestamp", kind="mergesort")
    assert meta is not None
    return sanitize_dataframe(df), meta


def create_throughput_chart(
    data: str, protocol: str, nvlink: str, infiniband: str, tcp: str
) -> alt.LayerChart:
    selector = alt.selection(type="point", fields=["ucx_version"], bind="legend")
    base = (
        alt.Chart(data)
        .transform_filter(
            {
                "and": [
                    alt.FieldEqualPredicate(field="protocol", equal=protocol),
                    # CSV read is stringly-typed
                    alt.FieldEqualPredicate(field="nvlink", equal=nvlink),
                    alt.FieldEqualPredicate(field="infiniband", equal=infiniband),
                    alt.FieldEqualPredicate(field="tcp", equal=tcp),
                ]
            }
        )
        .transform_calculate(
            throughput_mean=datum.throughput_mean / 1e9,
            throughput_std=datum.throughput_std / 1e9,
        )
        .encode(x=alt.X("timestamp:T", axis=alt.Axis(title="Date")))
    )
    throughput = base.mark_line().encode(
        alt.Y("throughput_mean:Q", axis=alt.Axis(title="Throughput [GB/s/GPU]")),
        color="ucx_version:N",
        opacity=alt.condition(selector, alt.value(1), alt.value(0.2)),
    )
    throughput_ci = (
        base.mark_area()
        .transform_calculate(
            y=datum.throughput_mean - datum.throughput_std,
            y2=datum.throughput_mean + datum.throughput_std,
        )
        .encode(
            y="y:Q",
            y2="y2:Q",
            color="ucx_version:N",
            opacity=alt.condition(selector, alt.value(0.3), alt.value(0.01)),
        )
    )
    return cast(
        alt.LayerChart, alt.layer(throughput, throughput_ci).add_params(selector)
    )


MERGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
</head>
<title>
dask-cuda local cudf merge performance
</title>
<body>
<h1>
Historical performance of CUDF merge
</h1>
Legend entries are clickable to highlight that set of data,
filled regions show standard deviation confidence intervals for
throughput (using harmonic means and standard deviations,
because it's a rate-based statistics).

<h2>Hardware setup</h2>

Single node DGX-1 with 8 V100 cards. In-node NVLink bisection
bandwidth 150GB/s (per <a
href="https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf">whitepaper</a>).

<h2>Benchmark setup</h2>
{metadata}

{divs}

<script type="text/javascript">
{embeds}
</script>
</body>
</html>
"""

TRANSPOSE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/vega@{vega_version}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite_version}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed_version}"></script>
</head>
<title>
dask-cuda local cupy transpose performance
</title>
<body>
<h1>
Historical performance of cupy transpose
</h1>
Legend entries are clickable to highlight that set of data,
filled regions show standard deviation confidence intervals for
throughput (using harmonic means and standard deviations,
because it's a rate-based statistics).
<h2>Hardware setup</h2>

Single node DGX-1 with 8 V100 cards. In-node NVLink bisection
bandwidth 150GB/s (per <a
href="https://images.nvidia.com/content/pdf/dgx1-v100-system-architecture-whitepaper.pdf">whitepaper</a>).

<h2>Benchmark setup</h2>
{metadata}

{divs}

<script type="text/javascript">
{embeds}
</script>
</body>
</html>
"""


def make_chart(template: str, csv_name: str, metadata: dict, output_file: Path):
    divs = []
    embeds = []
    for i, (protocol, nv, ib, tcp_over, name) in enumerate(
        [
            ("ucx", "True", "True", "False", "UCX NVLink + InfiniBand"),
            ("ucx", "True", "False", "False", "UCX NVLink only"),
            ("ucx", "False", "True", "False", "UCX InfiniBand only"),
            ("ucx", "False", "False", "True", "TCP over UCX"),
            ("tcp", "False", "False", "True", "Standard TCP"),
        ]
    ):
        throughput = create_throughput_chart(csv_name, protocol, nv, ib, tcp_over)
        divs.append(f"<h2>{name}</h2>")
        divs.append("<h3>Throughput/worker</h3>")
        divs.append(f'<div id="vis_throughput{i}"></div>')
        embeds.append(
            f"vegaEmbed('#vis_throughput{i}', {throughput.to_json(indent=None)})"
            ".catch(console.error);"
        )

    table_metadata = "\n".join(
        chain(
            ["<table><th>Name</th><th>Value</th>"],
            (
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in sorted(metadata.items(), key=itemgetter(0))
            ),
            ["</table>"],
        )
    )
    with open(output_file, "w") as f:
        f.write(
            template.format(
                vega_version=alt.VEGA_VERSION,
                vegalite_version=alt.VEGALITE_VERSION,
                vegaembed_version=alt.VEGAEMBED_VERSION,
                metadata=table_metadata,
                divs="\n".join(divs),
                embeds="\n".join(embeds),
            )
        )


def main(
    data_directory: Path = typer.Argument(..., help="Directory storing raw results"),
    output_directory: Path = typer.Argument(..., help="Directory storing outputs"),
    make_charts: bool = typer.Option(True, help="Make HTML pages for charts?"),
):
    merge_df, merge_meta = get_results(
        data_directory,
        output_directory / "single_node_merge_performance.csv",
        output_directory / "single_node_merge_performance-metadata.json",
        parse_merge,
        get_merge_metadata,
    )

    transpose_df, transpose_meta = get_results(
        data_directory,
        output_directory / "single_node_transpose_performance.csv",
        output_directory / "single_node_transpose_performance-metadata.json",
        parse_transpose,
        get_transpose_metadata,
    )
    merge_df.to_csv(output_directory / "single_node_merge_performance.csv", index=False)
    transpose_df.to_csv(
        output_directory / "single_node_transpose_performance.csv", index=False
    )
    with open(
        output_directory / "single_node_merge_performance-metadata.json", "w"
    ) as f:
        json.dump(merge_meta, f)

    with open(
        output_directory / "single_node_transpose_performance-metadata.json", "w"
    ) as f:
        json.dump(transpose_meta, f)

    if make_charts:
        make_chart(
            MERGE_TEMPLATE,
            "single_node_merge_performance.csv",
            merge_meta,
            output_directory / "single-node-merge.html",
        )
        make_chart(
            TRANSPOSE_TEMPLATE,
            "single_node_transpose_performance.csv",
            merge_meta,
            output_directory / "single-node-transpose.html",
        )


if __name__ == "__main__":
    typer.run(main)
