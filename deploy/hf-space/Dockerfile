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
# (buckets replaced the retired per-Space persistent storage). Fallback, when
# no dataset mount exists: the entrypoint seeds /data by download, which needs
# the secret HF_TOKEN.
# Base image pinned by DIGEST, not :latest -- an upstream rebuild must never
# change the environment under fixed source SHAs. Update deliberately, with
# a notes.md decision record.
FROM condaforge/miniforge3@sha256:8edc74dee5b29511a9c1a8941c77efbf438b47edffb6320ab8675dc06bd618c2

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n app python=3.12 \
    && conda create -y -n pandeia python=3.12 \
    && conda clean -afy

# Clone name VULCAN-JAX is load-bearing for the engine's path contract.
#
# RESTART NONCE: 2. Any commit to the Space repo re-runs the platform init
# steps -- the only restart path an `hf auth login` token has. Bump this
# number (leaving the pins alone) to retry a transient platform failure:
# comments do not bust the Docker cache, so every layer hits and the
# container simply restarts.
#
# Every deployed repository is pinned to an EXACT commit: jwst-tool = the
# validated content commit, siblings = the heads CI validated against.
# Changing an ARG default busts the cache of the layers that use it, so the
# pins are also the cache-bust. The build FAILS on a pin/rev-parse mismatch;
# BUILD_INFO records the enforced pins.
# (History of these decisions: notes.md, Decision records.)
ARG VULCAN_JAX_SHA=b3932f5db7e0cb34258bd1baa19c155edc5495bf
ARG VULCAN_FORWARD_SHA=63a73dea51932a7134c926d642aa9988179694d7
ARG JWST_TOOL_SHA=3221c21171c3a7a9f754cb135f00c808c41b0a26
RUN set -e; \
    for spec in \
      "VULCAN-JAX https://github.com/imalsky/jax-vulcan.git ${VULCAN_JAX_SHA}" \
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

# FastChem for THIS platform (any checked-in binary would be the wrong arch).
RUN rm -f /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem \
    && make -C /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan \
    && test -x /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem

# Only /data persists on a Space. The forward engine's data root is
# env-overridable ($VULCAN_FORWARD_DATA), so there are no repo-relative data
# stubs to fix up: the entrypoint just exports the variable.

COPY entrypoint.sh bootstrap_data.py /srv/app/
# Spaces run the container as uid 1000; the base image may already ship a
# UID-1000 user (miniforge does), so create one only if absent and chown
# numerically. FastChem writes its input/output inside its package dir at
# runtime, so the whole tree is chowned.
RUN chmod +x /srv/app/entrypoint.sh \
    && (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) \
    && chown -R 1000:1000 /srv/vulcan /srv/app
USER 1000

# BACKEND. The tool ships exactly one: the STScI-supported 2026.7 triple,
# pandeia.engine==2026.7 (requirements-pandeia.txt) + pandeia_data-2026.7-jwst
# + pandeia_psfs-2026.7-jwst. The engine, the reference data and the PSF
# library must all be the SAME release; the worker refuses a mismatched set
# rather than running one, and there is no archival backend to fall back to.
#
# Upgrading the release means, in this order: download both data trees into
# vulcan-jwst-tool/data/ (STScI Box; the PSF transfer truncates and still
# exits 0, so check byte counts), upload them to the dataset repo under
# jwst-data/ (upload_data.sh -- DELETE the staging dir first, it skips staging
# when one already exists), then rebuild the Space. Rebuilding before the
# dataset carries the trees is fail-closed by design: the data panel reports
# them missing and every noise step stops with an error.
ENV VULCAN_PROJECT_ROOT=/srv/vulcan \
    VULCAN_FORWARD_DATA=/data/retrieval-data \
    JWST_TOOL_DATA_DIR=/data/jwst-data \
    JWST_TOOL_OUTPUT_DIR=/data/output \
    JWST_TOOL_PANDEIA_PYTHON=/opt/conda/envs/pandeia/bin/python \
    HOME=/data/home \
    PATH=/opt/conda/envs/app/bin:$PATH

EXPOSE 7860
CMD ["/srv/app/entrypoint.sh"]
