# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess

import click


def get_versions():
    import cupy
    import numpy
    import ucp

    import dask
    import dask_cuda
    import distributed

    import cudf
    import rmm

    ucx_info = subprocess.check_output(["ucx_info", "-v"]).decode().strip()
    revmarker = "revision "
    revloc = ucx_info.find(revmarker)
    if revloc >= 0:
        ucx_revision, *_ = ucx_info[revloc + len(revmarker) :].split("\n")
    else:
        ucx_revision = ucx_info  # keep something so we can get it back later
    return {
        "numpy": numpy.__version__,
        "cupy": cupy.__version__,
        "rmm": rmm.__version__,
        "ucp": ucp.__version__,
        "ucx": ucx_revision,
        "dask": dask.__version__,
        "distributed": distributed.__version__,
        "dask_cuda": dask_cuda.__version__,
        "cudf": cudf.__version__,
    }


@click.command()
@click.argument("output_file", type=str)
def main(output_file):
    with open(output_file, "w") as f:
        json.dump(get_versions(), f)


if __name__ == "__main__":
    main()
