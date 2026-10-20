# jwst-transit-authority (JWST Transit Authority) as a Hugging Face Docker Space.
#
# The Space repo holds ONLY this deployment shim (Dockerfile, entrypoint,
# bootstrap, README); the build clones the three GitHub repos. To deploy new
# code: push to GitHub, bump the matching *_SHA ARG below (that busts the
# clone layer), push the Space files. Factory rebuild is the fallback if a
# build ever reuses a stale clone layer anyway.
#
# Datasets are NOT in the image. Preferred layout: the private HF dataset repo
# mounted READ-ONLY at /srv/hub-data plus a writable bucket volume at /data
# (HF has retired per-Space persistent storage). Fallback, when
# no dataset mount exists: the entrypoint seeds /data by download, which needs
# the secret HF_TOKEN.
# Base image pinned by DIGEST, not :latest. What that fixes under fixed source
# SHAs: the base image, the app env (installed from the lock with --no-deps)
# and the picaso env (an explicit conda lock with md5s plus a --no-deps pip
# lock). The apt packages, `python=3.12` and the pandeia env re-resolve on
# every build, so the image is reproducible, not bit-identical.
FROM condaforge/miniforge3@sha256:8edc74dee5b29511a9c1a8941c77efbf438b47edffb6320ab8675dc06bd618c2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n app python=3.12 \
    && conda create -y -n pandeia python=3.12 \
    && conda clean -afy

# Clone directories follow the workspace layout (VULCAN-JAX for the
# VULCAN3 repository).
#
# RESTART NONCE: 2. Any commit to the Space repo re-runs the platform init
# steps -- the only restart path an `hf auth login` token has. Bump this
# number (leaving the pins alone) to retry a transient platform failure:
# comments do not bust the Docker cache, so every layer hits and the
# container simply restarts.
#
# Every deployed repository is pinned to an EXACT commit: jwst-tool = the
# validated content commit, siblings = the deploy/pins.env commits CI tests
# against.
# Changing an ARG default busts the cache of the layers that use it, so the
# pins are also the cache-bust. The build FAILS on a pin/rev-parse mismatch;
# BUILD_INFO records the enforced pins.
ARG VULCAN3_SHA=92c3b1e45f60c9d52e86d8d854c5da3a2f6fc5ae
ARG VULCAN_FORWARD_SHA=32adc859137d03db9abd43b8c0203cea301c95dd
ARG JWST_TOOL_SHA=cd67563da6aa050c5d92dcdf58234f2cf93eaaca
RUN set -e; \
    for spec in \
      "VULCAN-JAX https://github.com/imalsky/VULCAN3.git ${VULCAN3_SHA}" \
      "vulcan-forward https://github.com/imalsky/vulcan-forward.git ${VULCAN_FORWARD_SHA}" \
      "jwst-transit-authority https://github.com/imalsky/jwst-transit-authority.git ${JWST_TOOL_SHA}"; do \
      set -- $spec; \
      git init -q "/srv/vulcan/$1"; \
      git -C "/srv/vulcan/$1" remote add origin "$2"; \
      git -C "/srv/vulcan/$1" fetch -q --depth 1 origin "$3"; \
      git -C "/srv/vulcan/$1" checkout -q --detach FETCH_HEAD; \
      test "$(git -C "/srv/vulcan/$1" rev-parse HEAD)" = "$3" || \
        { echo "PIN MISMATCH for $1: wanted $3, got $(git -C /srv/vulcan/$1 rev-parse HEAD)"; exit 1; }; \
      echo "$1 $(git -C "/srv/vulcan/$1" rev-parse HEAD)" >> /srv/vulcan/BUILD_INFO; \
    done; \
    cat /srv/vulcan/BUILD_INFO

# Pins live in the jwst-transit-authority repo (deploy/) so the Space shim never drifts
# from the validated versions. The app env installs the FULL resolver-clean
# lock with --no-deps so a later image rebuild reproduces the certified
# versions instead of re-resolving newer transitive dependencies. The lock
# includes streamlit/pandas (gui extra) and huggingface_hub (bootstrap).
RUN /opt/conda/envs/pandeia/bin/pip install --no-cache-dir \
        -r /srv/vulcan/jwst-transit-authority/deploy/requirements-pandeia.txt
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir --no-deps \
        -r /srv/vulcan/jwst-transit-authority/deploy/requirements-app-lock.txt
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir --no-deps \
        -e /srv/vulcan/VULCAN-JAX \
        -e /srv/vulcan/vulcan-forward \
        -e /srv/vulcan/jwst-transit-authority

# The PICASO engine's own env (picaso 4.1 needs numpy >= 2; the app pins numpy
# < 2), entered only by jwst_tool/picaso_worker.py via JWST_TOOL_PICASO_PYTHON.
# Conda installs the compiled stack (photochem, numba, h5py, ...) from the
# explicit lock; pip adds only the pip-only rest, --no-deps, so it can never
# replace a conda build. The build fails unless the stack is the one the
# worker requires. picaso.justdoit is not imported here: it needs the
# reference data, which only exists at run time.
RUN conda create -y -n picaso \
        --file /srv/vulcan/jwst-transit-authority/deploy/picaso-conda-lock.txt \
    && conda clean -afy \
    && /opt/conda/envs/picaso/bin/pip install --no-cache-dir --no-deps \
        -r /srv/vulcan/jwst-transit-authority/deploy/requirements-picaso-lock.txt \
    && /opt/conda/envs/picaso/bin/python -m pip check \
    && /opt/conda/envs/picaso/bin/python -c "import numba, numpy, photochem; \
from importlib.metadata import version as v; \
assert (v('picaso'), v('photochem')) == ('4.1', '0.9.2'), (v('picaso'), v('photochem')); \
assert numpy.__version__.startswith('2.'), numpy.__version__"

COPY entrypoint.sh bootstrap_data.py /srv/app/
# Spaces run the container as uid 1000; the base image may already ship a
# UID-1000 user (miniforge does), so create one only if absent and chown
# numerically. vulcan3 writes its generated rate code under its package
# __pycache__ at runtime, so the whole tree is chowned.
RUN chmod +x /srv/app/entrypoint.sh \
    && (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) \
    && chown -R 1000:1000 /srv/vulcan /srv/app
USER 1000

# BACKEND PIN. The tool ships exactly one: the STScI-supported 2026.7 triple,
# pandeia.engine==2026.7 (requirements-pandeia.txt) + pandeia_data-2026.7-jwst
# + pandeia_psfs-2026.7-jwst. The engine, the reference data and the PSF
# library must all be the SAME release; the worker refuses a mismatched set
# rather than running one, and there is no archival backend to fall back to.
#
# Upgrading the release means, in this order: download both data trees into
# vulcan-jwst-tool/data/ (STScI Box; the PSF transfer truncates and still
# exits 0, so check byte counts), upload them to the dataset repo under
# jwst-data/ (upload_data.sh -- it stages only top-level entries missing from
# the staging dir, so delete a stale entry first), then rebuild the Space.
# Rebuilding before the dataset carries the trees is fail-closed by design:
# the app lists them as required data missing and every noise step stops
# with an error.
ENV VULCAN_FORWARD_DATA=/data/retrieval-data \
    JWST_TOOL_DATA_DIR=/data/jwst-data \
    JWST_TOOL_OUTPUT_DIR=/data/output \
    JWST_TOOL_PANDEIA_PYTHON=/opt/conda/envs/pandeia/bin/python \
    JWST_TOOL_PICASO_PYTHON=/opt/conda/envs/picaso/bin/python \
    HOME=/data/home \
    PATH=/opt/conda/envs/app/bin:$PATH

EXPOSE 7860
CMD ["/srv/app/entrypoint.sh"]
