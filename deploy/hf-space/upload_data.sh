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

# Stage per TOP-LEVEL ENTRY, not per tree. The old check was a single
# sentinel file ("does the phoenix catalog exist in the stage?"), so a stage
# left over from an earlier run made this skip wholesale -- and anything added
# to data/ since then, e.g. a new pandeia release pair, was silently never
# uploaded. The Space then boots without it and every noise step errors.
# Known limit: this adds MISSING entries, it does not refresh changed files
# inside an entry already present. Delete the entry (or the stage) to re-copy.
# -RL: follow symlinks so the staged copy is self-contained.
stage_tree() {                  # stage_tree <src-dir> <dst-dir> <label>
    local src="$1" dst="$2" label="$3" entry base added=0
    mkdir -p "$dst"
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue          # unmatched glob
        base="$(basename "$entry")"
        if [ ! -e "$dst/$base" ]; then
            echo "Staging $label/$base ..."
            cp -RL "$entry" "$dst/"
            added=$((added + 1))
        fi
    done
    if [ "$added" -eq 0 ]; then
        echo "$label: already staged, nothing new."
    fi
    return 0
}

echo "Staging jwst-data (first run copies ~7 GB, needs the disk space) ..."
stage_tree "$ROOT/vulcan-jwst-tool/data" "$STAGE/jwst-data" jwst-data
# The engine's line lists + opacity cache still live in the retrieval
# checkout on the maintainer's machine; the dataset folder keeps the
# name "retrieval-data" deliberately, because renaming it would mean
# re-uploading gigabytes. $VULCAN_FORWARD_DATA points at it at boot.
stage_tree "$ROOT/vulcan-retrieval/data" "$STAGE/retrieval-data" retrieval-data

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
