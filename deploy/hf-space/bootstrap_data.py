"""Seed /data from the private HF dataset repo, idempotently and loudly.

Runs at every container boot (entrypoint), before the GUI starts. If the
marker files are already present the download is skipped, so a wake from
sleep costs nothing. The HF hub cache is pointed at ephemeral /tmp so the
persistent volume only holds the final ~8 GB copy.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/tmp/hf-cache")

DATA = Path("/data")
DATASET_REPO = os.environ.get("DATASET_REPO", "imalsky/vulcan-jwst-tool-data")

# One marker per dataset the tool refuses to run without. The Pandeia markers
# are the load-bearing VERSION FILES, not directories: an empty or truncated
# tree must not count as seeded. Only the 2026.7 pair is listed, because
# instruments.py defines exactly one backend ("current" = 2026.7) and raises
# for anything else. The 2026.2 trees were required here until 2026-08-14,
# from before the archival backend was removed; boot then failed closed on
# 4.33 GB nothing could reach. They may still sit in the dataset repo, which
# is harmless -- an unlisted tree is simply downloaded and ignored.
MARKERS = [
    DATA / "jwst-data" / "cdbs" / "grid" / "phoenix" / "catalog.fits",
    DATA / "jwst-data" / "pandeia_data-2026.7-jwst" / "VERSION_DATA",
    DATA / "jwst-data" / "pandeia_psfs-2026.7-jwst" / "VERSION_PSF",
    DATA / "retrieval-data" / "cm24_wasp39b",
    DATA / "retrieval-data" / "exojax_linelists",
    DATA / "retrieval-data" / "opacity_cache",
]


def missing() -> list[Path]:
    return [m for m in MARKERS if not m.exists()]


def main() -> int:
    gone = missing()
    if not gone:
        print("[bootstrap] /data already seeded, skipping download")
        return 0

    print(f"[bootstrap] seeding /data from {DATASET_REPO} "
          f"({len(gone)}/{len(MARKERS)} markers absent) ...")
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[bootstrap] WARNING: no HF_TOKEN secret set -- this only "
              "works if the dataset repo is public", file=sys.stderr)

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=DATASET_REPO, repo_type="dataset",
        token=token, local_dir=str(DATA))

    gone = missing()
    if gone:
        lines = "\n  ".join(str(m) for m in gone)
        raise RuntimeError(
            "dataset seed finished but these required paths are still "
            f"absent:\n  {lines}\nThe dataset repo layout must be "
            "jwst-data/... + retrieval-data/... at the repo root -- "
            "re-run deploy/hf-space/upload_data.sh from the Mac.")
    print("[bootstrap] /data seeded and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
