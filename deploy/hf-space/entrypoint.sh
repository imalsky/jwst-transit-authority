#!/usr/bin/env bash
# Space entrypoint. Preferred layout (HF volumes API, 2026): the dataset repo
# is mounted READ-ONLY at /srv/hub-data and a writable bucket volume at /data
# holds caches + anything the engines insist on writing. Fallback: seed /data
# from the dataset repo via bootstrap_data.py (needs HF_TOKEN) when no
# dataset mount exists.
set -euo pipefail

# Writable state root: the bucket volume at /data when present, else
# container disk (ephemeral -- caches lost on restart, everything works).
STATE=/data
if [ ! -d /data ] || ! touch /data/.rwtest 2>/dev/null; then
    STATE=/tmp/state
    echo "[entrypoint] WARNING: no writable /data volume; caches are" \
         "EPHEMERAL (lost on restart/rebuild)"
fi
rm -f /data/.rwtest 2>/dev/null || true
mkdir -p "$STATE/output" "$STATE/home" "$STATE/cwd"
export JWST_TOOL_OUTPUT_DIR="$STATE/output"
export HOME="$STATE/home"
# Persist numba-compiled kernels across restarts/rebuilds: 67/75 of
# picaso's jit kernels declare cache=True, and the compile is a large
# share of the first climate solve's minutes. Stale entries (picaso or
# python upgrades) miss harmlessly via numba's own code hashing.
mkdir -p "$STATE/numba_cache"
export NUMBA_CACHE_DIR="$STATE/numba_cache"

if [ -d /srv/hub-data/jwst-data ]; then
    echo "[entrypoint] dataset volume found at /srv/hub-data (no download)"
    # jwst-data is a pure READ consumer (cdbs/refdata/PSFs/mie): serve it
    # straight from the read-only mount.
    export JWST_TOOL_DATA_DIR=/srv/hub-data/jwst-data
    # the engine's data tree must be WRITABLE: radis creates a tempdir + lock files
    # inside exojax_linelists even for pure cache reads, and h2he line
    # lists download on first use. Sync the mount to the bucket (~360 MB
    # once); cp -au makes later boots a cheap stat pass that also picks up
    # dataset files that landed AFTER an earlier partial sync (the mount
    # live-updates as commits land).
    echo "[entrypoint] syncing retrieval-data to writable storage ..."
    mkdir -p "$STATE/retrieval-data"
    cp -au /srv/hub-data/retrieval-data/. "$STATE/retrieval-data/"
    # cp -a preserves the mount's read-only modes -- restore owner-write
    # (radis mkdirs its tempdir inside exojax_linelists at import).
    chmod -R u+wX "$STATE/retrieval-data"
    # The forward engine takes its data root from the environment now, so no
    # symlink into a checkout is needed (the dataset folder keeps its name --
    # renaming it would mean re-uploading gigabytes).
    export VULCAN_FORWARD_DATA="$STATE/retrieval-data"
    # PICASO provider + climate reference tree (v18): a pure READ consumer
    # -- measured 2026-07-20: a full chemeq + climate solve runs against a
    # chmod a-w tree, picaso never writes into refdata -- so it serves
    # straight from the read-only mount. Absent tree = the provider refuses
    # loudly and the GUI data panel shows the missing pieces; the VULCAN
    # engine is unaffected.
    if [ -d /srv/hub-data/picaso-reference ]; then
        export JWST_TOOL_PICASO_REFDATA=/srv/hub-data/picaso-reference
        echo "[entrypoint] PICASO reference tree found (provider enabled)"
    else
        echo "[entrypoint] NOTE: no picaso-reference in the dataset volume;" \
             "the PICASO provider/climate mode will refuse until it lands"
    fi
else
    if [ ! -d /data ]; then
        echo "ERROR: no dataset volume at /srv/hub-data and no storage at" >&2
        echo "/data. Mount the dataset repo as a volume (Settings ->" >&2
        echo "Storage/Volumes, or HfApi.set_space_volumes) or add a" >&2
        echo "writable volume so bootstrap_data.py can seed it." >&2
        exit 1
    fi
    echo "[entrypoint] no dataset mount -- seeding /data (download path)"
    mkdir -p /data/jwst-data /data/retrieval-data
    python /srv/app/bootstrap_data.py
    export JWST_TOOL_DATA_DIR=/data/jwst-data
    export VULCAN_FORWARD_DATA=/data/retrieval-data
    # bootstrap snapshots the WHOLE dataset repo, so picaso-reference lands
    # too when it exists there (v18.1; OPTIONAL -- the provider refuses
    # loudly without it and everything else is unaffected)
    if [ -d /data/picaso-reference/chemistry/visscher_grid_2121 ]; then
        export JWST_TOOL_PICASO_REFDATA=/data/picaso-reference
        # Refdata present is NOT the same as provider enabled: the PICASO
        # path is additionally gated on JWST_TOOL_ENABLE_UNCERTIFIED_PICASO,
        # which no deploy file sets. Do not restore the old "provider
        # enabled" wording -- it was false on every boot.
        echo "[entrypoint] PICASO reference tree seeded (refdata available;" \
             "the provider stays OFF until" \
             "JWST_TOOL_ENABLE_UNCERTIFIED_PICASO=1)"
    else
        echo "[entrypoint] NOTE: no picaso-reference in the seeded dataset;" \
             "the PICASO provider/climate mode will refuse until uploaded"
    fi
fi

# VULCAN-JAX's legacy IO writes a RELATIVE output/ dir in the process CWD
# (harmless junk, but the CWD must be writable -- the container default is
# root-owned and the forward subprocess inherits CWD from here).
cd "$STATE/cwd"

# Warm the data-status report in the BACKGROUND (v18.1 latency fix): the
# full scan stats thousands of remote-volume files, and without this the
# first visitor pays it behind a spinner. The GUI serves the disk-cached
# report the moment it exists.
(python -c "from jwst_tool import datacheck; datacheck.warm_report_cache()"     >/dev/null 2>&1 &)

# Pre-solve the DEFAULT climate configuration in the background so the
# first visitor gets a cache hit instead of a multi-minute solve (also
# compiles + persists the numba kernels). Idempotent: instant no-op when
# the /data volume already holds it. Log kept for diagnosis, never /dev/null.
# BOTH conditions are required: refdata on disk AND the experimental gate
# open. With only the refdata test (the shipped state until 2026-08-14) this
# launched a solve that canonical_params killed on its first statement.
if [ -n "${JWST_TOOL_PICASO_REFDATA:-}" ] \
   && [ "${JWST_TOOL_ENABLE_UNCERTIFIED_PICASO:-}" = "1" ]; then
    (python -c "from jwst_tool import picaso_climate; picaso_climate.warm_default()"         >"$STATE/output/climate_warm.log" 2>&1 &)
fi

# CORS/XSRF off: required for uploads (T-P tables, noise-floor tables) to
# work behind the Spaces proxy.
exec jwst-tool \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
