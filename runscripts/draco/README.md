## Run scripts for benchmarking on Draco

These scripts run benchmarks from
[`dask-cuda`](https://github.com/rapidsai/dask-cuda) in a multi-node
setting. These are set up to run on Draco.

Draco is a SLURM-based system that uses pyxis and enroot for
containerisation.

These scripts assume that the containers are already imported and
available as squashfs images at `$HOME/workdir/enroot-images/`.
`$HOME/workdir` should be a symlink something on a parallel filesystem
on draco (it is resolved via `readlink -f`).

Since the main goal is to benchmark performance of different
[UCX](https://github.com/openucx/ucx) versions, the image naming is as
`ucx-py-$UCX_VERSION-$DATE.sqsh`. Where `UCX_VERSION` is one of
`v1.12.x`, `v1.13.x`, `v1.14.x`, `master`; and `DATE` is the date of
the image.

`job.slurm` is the batch submission script, set up to request an
allocation with eight GPUs/node and then run all UCX versions with
images from the date of submission on the requested number of nodes.
In this loop, the run itself is controlled by `job.sh`.

Note that there is a [bug in UCX
v1.13.x](https://github.com/openucx/ucx/issues/8461) that causes
crashes on more than four nodes, so we skip that image if the
requested number of nodes is greater than four.

On node 0, `job.sh` starts the distributed scheduler, a dask cuda
worker (using eight GPUs), and eventually the client scripts; on all
other nodes we just start workers.

`job.sh` runs in the container, and expects to see environment
variables `RUNDIR`, `OUTDIR`, and `SCRATCHDIR` that are mounted in
from the outside (`job.slurm` sets this up).


### Recommended scaling

Up to 16 nodes is reasonable.

### Docker images

The `docker` subdirectory contains a docker file that builds images
suitable for running. `build-images.sh` builds images for each version
of UCX we want to test. You'll need to adapt the container registry
location to somewhere appropriate. A script that can be run on the
draco frontend to import the images is in `pull-images.sh`.

## Extracting data

For now, data are extracted from the output runs through separate
scripts. Assuming one has the `outputs` directory available locally,
then `python merge-outputs.py --charts merge-data.csv
transpose-data.csv` will munge all data and use
[altair](https://altair-viz.github.io) to produce simple HTML pages
that contain plots. You'll need a pre-release version of altair.
