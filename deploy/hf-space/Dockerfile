# vulcan-jwst-tool as a Hugging Face Docker Space.
#
# The Space repo holds ONLY this deployment shim (Dockerfile, entrypoint,
# bootstrap, README); the build clones the three GitHub repos, so updating
# code = push to GitHub, then Settings -> Factory rebuild (a plain rebuild
# may reuse the cached clone layer -- factory rebuild is the reliable path).
#
# Datasets are NOT in the image: the entrypoint seeds /data (the Space's
# persistent-storage volume) from a private HF dataset repo on first boot.
# Requires: persistent storage enabled (Small tier) + secret HF_TOKEN.
FROM condaforge/miniforge3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -y -n app python=3.12 \
    && conda create -y -n pandeia python=3.12 \
    && conda clean -afy

# Clone name VULCAN-JAX is load-bearing for the engine's path contract.
# BUILD_INFO records the exact SHAs baked into this image.
# RESTART NONCE: 2 (2026-08-04). Any commit to the Space repo re-runs the
# platform init steps, which is the only restart path an `hf auth login`
# token has -- the /restart and hardware endpoints answer 401. Bump this
# comment (leaving SRC_STAMP alone) to retry a transient platform failure:
# comments do NOT bust the Docker cache, so every layer hits and the
# container simply starts again. Used once for an `hf-mount` failure
# ("Initialization step 'hf-mount' failed with exit code: 0", empty reason,
# no app logs, hardware never allocated) whose two volume sources were both
# verified intact -- if a mount source is ever genuinely missing, fix the
# volume instead, since a restart cannot help.
#
# The SRC_STAMP echo busts the Docker layer cache so a rebuild re-clones:
# bump it to the new jwst-tool commit SHA on EVERY code update (the API
# factory-reboot endpoint refuses OAuth tokens, so cache-busting happens
# here instead; a comment would be stripped by the parser and cache-hit).
RUN echo "SRC_STAMP 4119280" \
    && git clone --depth 1 https://github.com/imalsky/jax-vulcan.git /srv/vulcan/VULCAN-JAX \
    && git clone --depth 1 https://github.com/imalsky/vulcan-forward.git /srv/vulcan/vulcan-forward \
    && git clone --depth 1 https://github.com/imalsky/vulcan-jwst-tool.git /srv/vulcan/vulcan-jwst-tool \
    && for r in VULCAN-JAX vulcan-forward vulcan-jwst-tool; do \
         echo "$r $(git -C /srv/vulcan/$r rev-parse HEAD)" >> /srv/vulcan/BUILD_INFO; done \
    && cat /srv/vulcan/BUILD_INFO

# Pins live in the jwst-tool repo (deploy/) so the Space shim never drifts
# from the validated versions. The app env installs the FULL LOCK with
# --no-deps: the validated environment violates some declared metadata
# (exojax 2.2.3 pins numpy<2, the working env runs numpy 2.0.2), so a
# resolver pass is guaranteed to fail -- replicate, don't resolve. The lock
# includes streamlit/pandas (gui extra) and huggingface_hub (bootstrap).
RUN /opt/conda/envs/pandeia/bin/pip install --no-cache-dir \
        -r /srv/vulcan/vulcan-jwst-tool/deploy/requirements-pandeia.txt
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir --no-deps \
        -r /srv/vulcan/vulcan-jwst-tool/deploy/requirements-app-lock.txt
RUN /opt/conda/envs/app/bin/pip install --no-cache-dir --no-deps \
        -e /srv/vulcan/VULCAN-JAX \
        -e /srv/vulcan/vulcan-forward \
        -e /srv/vulcan/vulcan-jwst-tool

# FastChem for THIS platform (any checked-in binary would be the wrong arch).
RUN rm -f /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem \
    && make -C /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan \
    && test -x /srv/vulcan/VULCAN-JAX/src/vulcan_jax/fastchem_vulcan/fastchem

# Only /data persists on a Space. Since the forward engine became its own
# distribution its data root IS env-overridable ($VULCAN_FORWARD_DATA), so
# there are no repo-relative data stubs to remove: the entrypoint just exports
# the variable. The retrieval application is no longer installed at all -- the
# planner depends on the shared engine, not on that app.

COPY entrypoint.sh bootstrap_data.py /srv/app/
# Spaces run the container as uid 1000; the base image may already ship a
# UID-1000 user (miniforge does), so create one only if absent and chown
# numerically. FastChem writes its input/output inside its package dir at
# runtime, so the whole tree is chowned.
RUN chmod +x /srv/app/entrypoint.sh \
    && (id -u 1000 >/dev/null 2>&1 || useradd -m -u 1000 user) \
    && chown -R 1000:1000 /srv/vulcan /srv/app
USER 1000

# BACKEND (2026-08-04). `current` is the STScI-supported 2026.7 triple:
# pandeia.engine==2026.7 (requirements-pandeia.txt) + pandeia_data-2026.7-jwst
# + pandeia_psfs-2026.7-jwst. The engine, the reference data and the PSF
# library must all be the SAME release; the worker refuses a mismatched set
# rather than running one.
#
# *** DO NOT REBUILD THIS IMAGE UNTIL BOTH 2026.7 TREES ARE IN THE DATASET ***
# The Space seeds /data/jwst-data from the dataset repo. Until
# pandeia_data-2026.7-jwst and pandeia_psfs-2026.7-jwst are uploaded there,
# this pin makes the app demand trees the volume does not have: the data panel
# reports both missing and every noise step stops with an error. That is the
# intended fail-closed behaviour, not a silent fallback, and it is exactly what
# happened on 2026-08-03 when the token was repointed ahead of the data.
#
# Upgrade order (all three, in this order):
#   1. download pandeia_data-2026.7-jwst (~20 MiB) and
#      pandeia_psfs-2026.7-jwst (~4 GiB) into vulcan-jwst-tool/data/
#      (STScI Box; browser, or the scripted recipe in fetch.py's MANUAL --
#      the PSF transfer truncates and still exits 0, so check byte counts)
#   2. upload both to the dataset repo under jwst-data/ (upload_data.sh;
#      DELETE the staging dir first, it skips staging when one already exists)
#   3. rebuild the Space with this pin at `current`
#
# The previous 2026.2 deployment stays selectable as JWST_TOOL_BACKEND=
# archival_2026_2 for reproducibility, as long as its two trees remain in the
# dataset. The committed PandExo parity artifact
# (tests/parity/outputs/REPORT.md) is a gate-evaluated PASS on 2026.7
# (2026-08-04); the old 2026.2 artifact predates the gate and was never
# gate-evaluated.
ENV VULCAN_PROJECT_ROOT=/srv/vulcan \
    VULCAN_FORWARD_DATA=/data/retrieval-data \
    JWST_TOOL_DATA_DIR=/data/jwst-data \
    JWST_TOOL_OUTPUT_DIR=/data/output \
    JWST_TOOL_PANDEIA_PYTHON=/opt/conda/envs/pandeia/bin/python \
    JWST_TOOL_BACKEND=current \
    HOME=/data/home \
    PATH=/opt/conda/envs/app/bin:$PATH

EXPOSE 7860
CMD ["/srv/app/entrypoint.sh"]
