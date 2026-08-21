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
stage_tree() {                  # stage_tree <src-dir> <dst-dir> <label> [skip]
    local src="$1" dst="$2" label="$3" skip="${4:-}" entry base added=0
    mkdir -p "$dst"
    for entry in "$src"/*; do
        [ -e "$entry" ] || continue          # unmatched glob
        base="$(basename "$entry")"
        if [ -n "$skip" ] && [ "$base" = "$skip" ]; then
            continue                         # staged separately below
        fi
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
stage_tree "$ROOT/vulcan-retrieval/data" "$STAGE/retrieval-data" retrieval-data \
    exomolop

# ExoMolOP k-tables are staged SELECTIVELY (~371 MiB each): only the species
# the planner can actually select, so the dataset repo does not carry
# gigabytes the Space never opens. The engine's local tree holds more of them
# than this tool has a molecule table for. The list is read from the INSTALLED
# tool -- never copy it in here, it would rot the moment a molecule is added.
KTABLE_SRC="$ROOT/vulcan-retrieval/data/exomolop"
if [ -d "$KTABLE_SRC" ]; then
    KTABLE_MOLS="$(python3 - <<'PYEOF'
from jwst_tool import forward
print(" ".join(m for m in forward.MOLECULES + forward.EXTRA_MOLECULES
                if m not in forward._NO_EXOMOLOP_TABLE))
PYEOF
)"
    [ -n "$KTABLE_MOLS" ] || { echo "ERROR: empty k-table molecule list" >&2; exit 1; }
    mkdir -p "$STAGE/retrieval-data/exomolop"
    for mol in $KTABLE_MOLS; do
        src="$KTABLE_SRC/$mol.ktable.h5"
        dst="$STAGE/retrieval-data/exomolop/$mol.ktable.h5"
        if [ ! -e "$src" ]; then
            echo "ERROR: no k-table for $mol at $src -- fetch it first:" >&2
            echo "  python -m vulcan_forward.fetch_exomolop --molecules $mol" >&2
            exit 1
        fi
        # Hardlink (instant, no extra disk); the tables are immutable products.
        [ -e "$dst" ] || ln "$src" "$dst" 2>/dev/null || cp -c "$src" "$dst"
    done
    # Provenance must describe what was STAGED, not the maintainer's whole
    # local tree. Copying it wholesale once made the dataset repo claim 17
    # tables when 11 were uploaded, and a check that trusted it
    # concluded SH/SO were present when they had never been staged -- the
    # Space then failed at run time on the default molecule set.
    KTABLE_MOLS="$KTABLE_MOLS" python3 - "$KTABLE_SRC/provenance.json" \
            "$STAGE/retrieval-data/exomolop/provenance.json" <<'PYEOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
staged = set(os.environ["KTABLE_MOLS"].split())
full = json.load(open(src))
json.dump({k: v for k, v in full.items() if k in staged},
          open(dst, "w"), indent=1, sort_keys=True)
missing = sorted(staged - set(full))
if missing:
    print(f"WARNING: staged tables with no provenance entry: {missing}",
          file=sys.stderr)
PYEOF
    echo "exomolop: staged $(echo "$KTABLE_MOLS" | wc -w | tr -d ' ') tables."
else
    echo "NOTE: no exomolop/ tree at $KTABLE_SRC -- skipping the k-tables."
    echo "      The Space's default opacity_mode will stop with an error."
fi

# Resumable uploader (safe to re-run after an interrupted upload). Uploads the
# staging dir's CONTENTS, giving jwst-data/ + retrieval-data/ at the repo root
# -- exactly the layout bootstrap_data.py expects.
echo "Uploading to $REPO (resumable; re-run this script if interrupted) ..."
$HF upload-large-folder "$REPO" --repo-type dataset "$STAGE"

echo "Done. The staging copy at $STAGE can be deleted once the Space boots."
