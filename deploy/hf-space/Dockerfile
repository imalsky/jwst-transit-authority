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
RUN echo "SRC_STAMP 1edd1c2" \
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

# BACKEND PIN (2026-08-04). This Space is a consistent 2026.2 deployment:
# requirements-pandeia.txt installs pandeia.engine==2026.2, and the mounted
# dataset carries pandeia_data-2026.2-jwst + pandeia_psfs-2026.2-jwst. On
# 2026-08-03 the `current` token was repointed to the STScI-supported 2026.7
# release, so leaving this at `current` made the app demand 2026.7 trees that
# are not in the dataset -- the data panel reported both as missing and every
# noise step would have stopped with an error (correctly: the worker refuses a
# mismatched engine/refdata/PSF set rather than running one).
#
# `archival_2026_2` selects the matched 2026.2 triple this Space actually has.
# The app labels it ARCHIVAL and says it is not suitable for planning new
# proposals, which is accurate.
#
# TO UPGRADE to 2026.7, all three must move together:
#   1. deploy/requirements-pandeia.txt -> pandeia.engine==2026.7
#   2. download pandeia_data-2026.7-jwst (~20 MiB) and
#      pandeia_psfs-2026.7-jwst (~4 GiB) -- browser only, STScI Box refuses
#      scripted requests -- and upload both to the dataset repo under
#      jwst-data/
#   3. drop this pin back to JWST_TOOL_BACKEND=current
# Doing any one alone leaves a mismatched set, which the worker refuses.
ENV VULCAN_PROJECT_ROOT=/srv/vulcan \
    VULCAN_FORWARD_DATA=/data/retrieval-data \
    JWST_TOOL_DATA_DIR=/data/jwst-data \
    JWST_TOOL_OUTPUT_DIR=/data/output \
    JWST_TOOL_PANDEIA_PYTHON=/opt/conda/envs/pandeia/bin/python \
    JWST_TOOL_BACKEND=archival_2026_2 \
    HOME=/data/home \
    PATH=/opt/conda/envs/app/bin:$PATH

EXPOSE 7860
CMD ["/srv/app/entrypoint.sh"]
