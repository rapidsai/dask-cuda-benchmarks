# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import click

from distributed import Client


def cleanup_lru_cache():
    import gc

    from distributed.worker import cache_loads

    cache_loads.clear()
    gc.collect()


@click.command()
@click.argument("scheduler_file", type=str)
def main(scheduler_file):
    client = Client(scheduler_file=scheduler_file)
    client.run(cleanup_lru_cache)
    client.close()


if __name__ == "__main__":
    main()
