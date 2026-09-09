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
# Persist numba-compiled kernels across restarts/rebuilds; radis (under
# exojax) jit-compiles on first use. Stale entries miss harmlessly via
# numba's own code hashing.
mkdir -p "$STATE/numba_cache"
export NUMBA_CACHE_DIR="$STATE/numba_cache"

# The ExoMolOP k-tables are read straight off the read-only dataset mount
# (see the exomolop block below). HDF5 >= 1.10 takes an advisory lock even on
# a read-only open, and a mount that does not implement locking answers with
# "unable to lock file", which would fail every model build. Nothing in this
# container ever WRITES an HDF5 file, so turning the lock off is safe here and
# is the documented override for exactly this case.
export HDF5_USE_FILE_LOCKING=FALSE

if [ -d /srv/hub-data/jwst-data ]; then
    echo "[entrypoint] dataset volume found at /srv/hub-data (no download)"
    # jwst-data is a pure READ consumer (cdbs/refdata/PSFs): serve it
    # straight from the read-only mount.
    export JWST_TOOL_DATA_DIR=/srv/hub-data/jwst-data
    # the engine's data tree must be WRITABLE (exojax writes CIA caches
    # beside its inputs). Sync the mount to the bucket (~360 MB
    # once); cp -au makes later boots a cheap stat pass that also picks up
    # dataset files that landed AFTER an earlier partial sync (the mount
    # live-updates as commits land).
    echo "[entrypoint] syncing retrieval-data to writable storage ..."
    mkdir -p "$STATE/retrieval-data"
    for entry in /srv/hub-data/retrieval-data/*; do
        [ -e "$entry" ] || continue          # unmatched glob
        name="$(basename "$entry")"
        if [ "$name" = "exomolop" ]; then continue; fi
        cp -au "$entry" "$STATE/retrieval-data/"
        # cp -a preserves the mount's read-only modes -- restore owner-write
        # (exojax writes its CIA caches beside the inputs).
        chmod -R u+wX "$STATE/retrieval-data/$name"
    done
    # The ExoMolOP k-tables are the one tree that is NOT copied: 4.3 GB of
    # pure-read HDF5 (h5py opens them "r"; nothing in the engine writes
    # there), so they are symlinked straight off the read-only mount.
    # Copying them would pay for the same bytes twice -- once in the dataset
    # repo, once in the bucket -- and add minutes to every cold boot.
    if [ -d /srv/hub-data/retrieval-data/exomolop ]; then
        rm -rf "$STATE/retrieval-data/exomolop"   # drop an older boot's copy
        ln -s /srv/hub-data/retrieval-data/exomolop \
              "$STATE/retrieval-data/exomolop"
        echo "[entrypoint] exomolop k-tables served from the read-only mount"
    else
        echo "[entrypoint] WARNING: no exomolop/ in the dataset volume --" \
             "the gas opacity has no tables and every model step" \
             "will stop with an error (upload them: deploy/hf-space/" \
             "upload_data.sh, or see 'jwst-tool data')"
    fi
    # The forward engine takes its data root from the environment now, so no
    # symlink into a checkout is needed (the dataset folder keeps its name --
    # renaming it would mean re-uploading gigabytes).
    export VULCAN_FORWARD_DATA="$STATE/retrieval-data"
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
fi

# VULCAN-JAX's legacy IO writes a RELATIVE output/ dir in the process CWD
# (harmless junk, but the CWD must be writable -- the container default is
# root-owned and the forward subprocess inherits CWD from here).
cd "$STATE/cwd"

# Warm the data-status report and the app's imports BEFORE serving. The full
# scan stats thousands of remote-volume files; when it ran in the background a
# visitor arriving during the first minute paid it again behind the GUI (the
# first page load sat on the how-to text for that long). One foreground pass
# costs the boot about a minute and the first visitor nothing; the duration
# is logged so a slow boot can be read off the container log.
_t0=$(date +%s)
python - <<'PY' || echo "[entrypoint] WARNING: warm-up failed; the first visitor pays the data scan"
from jwst_tool import datacheck, forward
datacheck.warm_report_cache(base_mols=forward.MOLECULES, extra_mols=forward.EXTRA_MOLECULES)
import matplotlib; matplotlib.use("Agg", force=True)
from streamlit.testing.v1 import AppTest
import jwst_tool, pathlib
at = AppTest.from_file(str(pathlib.Path(jwst_tool.__file__).with_name("app.py")), default_timeout=1800)
at.run()
assert not at.exception, [e.value for e in at.exception]
PY
echo "[entrypoint] warm-up (data report + one headless app pass) took $(( $(date +%s) - _t0 )) s"

# CORS/XSRF off: required for uploads (T-P tables, noise-floor tables) to
# work behind the Spaces proxy.
exec jwst-tool \
    --server.address=0.0.0.0 \
    --server.port=7860 \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
