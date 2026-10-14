"""Seed /data from the private HF dataset repo, idempotently and loudly.

The entrypoint runs it before the GUI starts on every boot without a dataset
mount. When every marker file is present the download is skipped, so a wake
from sleep costs nothing. The HF hub cache points at ephemeral /tmp, so the
persistent volume holds only the final copy (~10 GB, about 2.2 GB of it the
PICASO correlated-k tables).

A missing PICASO file triggers the download like any marker. When only
PICASO files are absent, a failed download only warns; a PICASO file still
missing afterwards only warns too. VULCAN then boots without PICASO, and
every PICASO run stops with an error.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

DATA = Path("/data")
DATASET_REPO = os.environ.get("DATASET_REPO", "imalsky/vulcan-jwst-tool-data")

# One marker per dataset VULCAN refuses to run without. The Pandeia markers
# are the load-bearing VERSION FILES, not directories: an empty or truncated
# tree must not count as seeded. Only the 2026.7 pair is listed, because
# instruments.py defines exactly one backend ("current" = 2026.7) and raises
# for anything else. Other Pandeia trees may still sit in the dataset repo,
# which is harmless -- an unlisted tree is simply downloaded and ignored.
# Never list a tree no backend can reach: boot then fails closed on data
# nothing could use.
# The per-molecule k-table markers are DERIVED, in ktable_markers() below.
# The PICASO markers (picaso_markers()) are optional: see the docstring.
MARKERS = [
    DATA / "jwst-data" / "cdbs" / "grid" / "phoenix" / "catalog.fits",
    DATA / "jwst-data" / "pandeia_data-2026.7-jwst" / "VERSION_DATA",
    DATA / "jwst-data" / "pandeia_psfs-2026.7-jwst" / "VERSION_PSF",
    DATA / "retrieval-data" / "cm24_wasp39b",
    DATA / "retrieval-data" / "opacity_cache",
]


def ktable_markers() -> list[Path]:
    """One marker per k-table the INSTALLED tool can select.

    DERIVED, never hardcoded. One marker standing for the whole tree lets a
    seeded /data satisfy the check forever, so widening the molecule menu
    leaves the persistent volume a boot behind the dataset repo with no way
    to catch up -- markers present, download skipped, every default run
    stopping on a missing table. A per-molecule list re-seeds on the next
    boot instead.
    """
    from jwst_tool import forward
    return [DATA / "retrieval-data" / "exomolop" / f"{m}.ktable.h5"
            for m in forward.MOLECULES + forward.EXTRA_MOLECULES
            if m not in forward._NO_EXOMOLOP_TABLE]


def picaso_markers() -> list[Path]:
    """One marker per PICASO reference file and correlated-k table, read from
    the installed tool's datacheck, so the list moves with the pinned code
    like ktable_markers()."""
    from jwst_tool import datacheck as dc
    root = DATA / "jwst-data" / f"picaso_refdata-{dc.PICASO_RELEASE}"
    return ([root / f for f in dc.PICASO_REFDATA_FILES]
            + [root / dc.PICASO_CK_DIR / n for n in dc.PICASO_CK_MD5])


def missing(markers: list[Path]) -> list[Path]:
    return [m for m in markers if not m.exists()]


def main() -> int:
    required = MARKERS + ktable_markers()
    picaso = picaso_markers()
    gone = missing(required + picaso)
    if not gone:
        print("[bootstrap] /data already seeded, skipping download")
        return 0

    print(f"[bootstrap] seeding /data from {DATASET_REPO} "
          f"({len(gone)}/{len(required) + len(picaso)} markers absent) ...")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[bootstrap] WARNING: no HF_TOKEN secret set -- this only "
              "works if the dataset repo is public", file=sys.stderr)

    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=DATASET_REPO, repo_type="dataset",
            token=token, local_dir=str(DATA))
    except Exception as e:                                   # noqa: BLE001
        if missing(required):
            raise
        # only PICASO files were absent: VULCAN still boots offline
        print(f"[bootstrap] WARNING: dataset download failed ({e!r}); "
              "booting without the PICASO data", file=sys.stderr)

    gone = missing(required)
    if gone:
        lines = "\n  ".join(str(m) for m in gone)
        raise RuntimeError(
            "dataset seed finished but these required paths are still "
            f"absent:\n  {lines}\nThe dataset repo layout must be "
            "jwst-data/... + retrieval-data/... at the repo root -- "
            "re-run deploy/hf-space/upload_data.sh from the Mac.")
    gone = missing(picaso)
    if gone:
        print(f"[bootstrap] WARNING: {len(gone)}/{len(picaso)} PICASO files "
              f"absent (first: {gone[0]}) -- VULCAN is unaffected, every "
              "PICASO run will stop with an error (upload them: "
              "deploy/hf-space/upload_data.sh)", file=sys.stderr)
    print("[bootstrap] /data seeded and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
