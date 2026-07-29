#!/usr/bin/env bash
# Run on the Mac. Stages a symlink-dereferenced copy of the two data trees
# (~7.5 GB; the phoenix grid symlink must be materialized) and uploads them
# to the private HF dataset repo the Space seeds /data from.
#
# Prereqs:
#   pip install -U "huggingface_hub[cli]"
#   hf auth login              (or: huggingface-cli login)
#   dataset repo created at hf.co/new-dataset (private)
#
# Usage:  ./upload_data.sh [staging-dir]     (default ~/Desktop/hf_data_stage)
#   env DATASET_REPO overrides the target (default imalsky/vulcan-jwst-tool-data)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STAGE="${1:-$HOME/Desktop/hf_data_stage}"
REPO="${DATASET_REPO:-imalsky/vulcan-jwst-tool-data}"

if command -v hf >/dev/null 2>&1; then HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then HF=huggingface-cli
else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' on PATH." >&2
    echo "Install with: pip install -U \"huggingface_hub[cli]\"" >&2
    exit 1
fi

if [ ! -e "$ROOT/vulcan-jwst-tool/data/cdbs/grid/phoenix/catalog.fits" ]; then
    echo "ERROR: phoenix grid not resolvable through the cdbs symlink" >&2
    exit 1
fi

# -RL: follow symlinks so the staged copy is self-contained.
if [ ! -e "$STAGE/jwst-data/cdbs/grid/phoenix/catalog.fits" ]; then
    echo "Staging jwst-data (~7 GB copy, needs the disk space) ..."
    mkdir -p "$STAGE"
    cp -RL "$ROOT/vulcan-jwst-tool/data" "$STAGE/jwst-data"
fi
if [ ! -d "$STAGE/retrieval-data/cm24_wasp39b" ]; then
    echo "Staging retrieval-data ..."
    # The engine's line lists + opacity cache still live in the retrieval
    # checkout on the maintainer's machine; the dataset folder keeps the
    # name "retrieval-data" deliberately, because renaming it would mean
    # re-uploading gigabytes. $VULCAN_FORWARD_DATA points at it at boot.
    cp -RL "$ROOT/vulcan-retrieval/data" "$STAGE/retrieval-data"
fi

# PICASO reference tree (v18.1): OPTIONAL science data for the PICASO
# provider + climate mode -- staged from JWST_TOOL_PICASO_REFDATA when set
# (the tool's own selector; ~12 GB: chemistry grid + preweighted CK +
# stellar grids + opacities.db). cp -c uses APFS clonefile (instant, no
# extra disk) and falls back to a plain dereferencing copy elsewhere.
if [ -n "${JWST_TOOL_PICASO_REFDATA:-}" ] \
        && [ -d "$JWST_TOOL_PICASO_REFDATA/chemistry/visscher_grid_2121" ]; then
    if [ ! -d "$STAGE/picaso-reference/chemistry/visscher_grid_2121" ]; then
        echo "Staging picaso-reference (~12 GB; APFS clone when possible) ..."
        cp -Rc "$JWST_TOOL_PICASO_REFDATA" "$STAGE/picaso-reference" 2>/dev/null \
            || cp -RL "$JWST_TOOL_PICASO_REFDATA" "$STAGE/picaso-reference"
        chmod -R u+w "$STAGE/picaso-reference"
    fi
    if [ ! -f "$STAGE/picaso-reference/manifest.json" ]; then
        echo "Generating picaso-reference/manifest.json ..."
        python3 - "$STAGE/picaso-reference" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files, shas = {}, {}
for p in sorted(root.rglob("*")):
    if not p.is_file() or ".cache" in p.parts:
        continue
    rel = str(p.relative_to(root))
    size = p.stat().st_size
    files[rel] = size
    if size <= 50 * 2**20:
        shas[rel] = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
(root / "manifest.json").write_text(json.dumps(
    {"tree": "picaso v4.0 reference", "n_files": len(files),
     "files": files, "sha1_16": shas}, indent=1, sort_keys=True))
print(f"manifest.json: {len(files)} files")
PYEOF
    fi
else
    echo "NOTE: JWST_TOOL_PICASO_REFDATA unset or incomplete -- skipping the"
    echo "      picaso-reference stage (the PICASO provider/climate mode will"
    echo "      show MISSING on the Space until it is uploaded)."
fi

# Resumable uploader (safe to re-run after an interrupted upload). Uploads the
# staging dir's CONTENTS, giving jwst-data/ + retrieval-data/ at the repo root
# -- exactly the layout bootstrap_data.py expects.
echo "Uploading to $REPO (resumable; re-run this script if interrupted) ..."
$HF upload-large-folder "$REPO" --repo-type dataset "$STAGE"

echo "Done. The staging copy at $STAGE can be deleted once the Space boots."
