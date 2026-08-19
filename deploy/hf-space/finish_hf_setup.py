#!/usr/bin/env python3
"""One-shot Hugging Face setup, run AFTER `hf auth login`.

Does everything the browser/CLI sequence in notes.md's Deployment runbooks section
(Hugging Face Space runbook) steps 2-6 does:
creates the private dataset + Space repos, uploads the Space shim files,
sets the HF_TOKEN secret (+ DATASET_REPO variable if non-default), requests
persistent storage / CPU Upgrade hardware / 1 h sleep (these three need
billing configured at hf.co/settings/billing), and finally uploads the
staged ~7.5 GB data (resumable -- re-run this script if interrupted;
completed steps are skipped or idempotent).

Env overrides: DATASET_REPO, SPACE_REPO, STAGE_DIR.
Loud on every failure, with the remedy.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = Path(os.environ.get("STAGE_DIR", Path.home() / "Desktop" / "hf_data_stage"))
# Pandeia markers are the load-bearing VERSION FILES, not directories: an
# empty or truncated tree must not count as staged. Only the 2026.7 pair,
# matching the single backend instruments.py defines; the same rule as
# bootstrap_data.py's MARKERS.
STAGE_MARKERS = [
    STAGE / "jwst-data" / "cdbs" / "grid" / "phoenix" / "catalog.fits",
    STAGE / "jwst-data" / "pandeia_data-2026.7-jwst" / "VERSION_DATA",
    STAGE / "jwst-data" / "pandeia_psfs-2026.7-jwst" / "VERSION_PSF",
    STAGE / "retrieval-data" / "cm24_wasp39b",
    STAGE / "retrieval-data" / "exojax_linelists",
    STAGE / "retrieval-data" / "opacity_cache",
]


def die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def step(msg: str) -> None:
    print(f"\n=== {msg}")


def main() -> int:
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        die("huggingface_hub not importable -- pip install -U huggingface_hub")

    token = get_token()
    if not token:
        die("not logged in -- run `hf auth login` first (token from "
            "hf.co/settings/tokens), then re-run this script")

    api = HfApi()
    user = api.whoami()["name"]
    dataset = os.environ.get("DATASET_REPO", f"{user}/vulcan-jwst-tool-data")
    space = os.environ.get("SPACE_REPO", f"{user}/jwst-tool")
    print(f"HF user: {user}\ndataset: {dataset}\nspace:   {space}\nstage:   {STAGE}")

    gone = [m for m in STAGE_MARKERS if not m.exists()]
    if gone:
        listing = "\n  ".join(str(m) for m in gone)
        die(f"staged data incomplete -- missing:\n  {listing}\n"
            "Run deploy/hf-space/upload_data.sh once to (re)stage, or set "
            "STAGE_DIR.")

    # Non-fatal preflight: the Space build clones the GitHub repos and reads
    # version pins from vulcan-jwst-tool/deploy/ -- if deploy/ is not pushed
    # yet, the first build fails until a push + Settings -> Factory rebuild.
    step("checking GitHub for the pushed deploy/ directory")
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/imalsky/vulcan-jwst-tool/contents/deploy",
            headers={"User-Agent": "jwst-tool-deploy"})
        with urllib.request.urlopen(req, timeout=15) as r:
            json.load(r)
        print("deploy/ is on GitHub -- the Space build can succeed")
    except Exception as e:  # noqa: BLE001 -- any failure is the same warning
        print(f"WARNING: could not confirm deploy/ on GitHub ({e}).\n"
              "If it is not pushed yet, the first Space build FAILS; push, "
              "then Space Settings -> Factory rebuild.", file=sys.stderr)

    step("creating repos (idempotent)")
    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    print(f"dataset repo exists: {dataset}")
    # HF policy (2026): hosting a DOCKER Space -- even on free cpu-basic
    # hardware -- requires a PRO subscription (~$9/mo). Without it the
    # create call 402s; everything dataset-side is still free, so the data
    # upload below proceeds and the Space steps are skipped until PRO.
    from huggingface_hub.errors import HfHubHTTPError
    space_ok = True
    try:
        api.create_repo(space, repo_type="space", private=True,
                        space_sdk="docker", exist_ok=True)
        print(f"space repo exists: {space}")
    except HfHubHTTPError as e:
        if getattr(e, "response", None) is not None \
                and e.response.status_code == 402:
            space_ok = False
            print("WARNING: Space creation refused (402 Payment Required): "
                  "Docker Spaces need a PRO subscription "
                  "(huggingface.co/pro, ~$9/mo) even on free hardware. "
                  "Skipping every Space step; the data upload still runs. "
                  "Subscribe, then re-run this script to finish.",
                  file=sys.stderr)
        else:
            raise

    if space_ok:
        step("uploading Space shim files (triggers a build)")
        api.upload_folder(
            repo_id=space, repo_type="space", folder_path=str(HERE),
            ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"])
        print(f"pushed {HERE.name}/ contents to the Space repo")

        step("configuring Space secrets/variables")
        api.add_space_secret(space, "HF_TOKEN", token)
        print("HF_TOKEN secret set from your login token.")
        print("NOTE: for least privilege, later create a fine-grained READ "
              "token")
        print("at hf.co/settings/tokens (access to the dataset repo only) and")
        print("replace the secret in Space Settings.")
        if dataset != f"{user}/vulcan-jwst-tool-data":
            api.add_space_variable(space, "DATASET_REPO", dataset)
            print(f"DATASET_REPO variable set to {dataset}")

        step("mounting volumes (dataset read-only + writable cache bucket)")
        # 2026 volumes API (replaces the retired persistent-storage
        # endpoint): the dataset repo mounts read-only at /srv/hub-data so
        # the container needs NO download at boot; a private bucket mounts
        # writable at /data for the model/noise/adjoint caches.
        from huggingface_hub import Volume
        bucket = f"{user}/jwst-tool-cache"
        api.create_bucket(bucket, private=True, exist_ok=True)
        api.set_space_volumes(space, volumes=[
            Volume(type="dataset", source=dataset,
                   mount_path="/srv/hub-data", read_only=True),
            Volume(type="bucket", source=bucket,
                   mount_path="/data", read_only=False)])
        print(f"volumes: {dataset} (ro) -> /srv/hub-data, "
              f"{bucket} (rw) -> /data")

        step("requesting hardware + sleep timer")
        try:
            api.request_space_hardware(space, "cpu-upgrade")
            print("hardware: cpu-upgrade (8 vCPU / 32 GB, "
                  "~$0.03/hr while awake)")
            api.set_space_sleep_time(space, 3600)
            print("sleep timer: 1 h idle")
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: hardware/sleep request failed ({e}) -- an "
                  "OAuth login token lacks the compute scope for this "
                  "endpoint. Set it in the browser instead: Space Settings "
                  "-> Hardware -> CPU Upgrade, sleep 1 h.", file=sys.stderr)

    step("uploading staged data (~7.5 GB, resumable; the long step)")
    api.upload_large_folder(repo_id=dataset, repo_type="dataset",
                            folder_path=str(STAGE))
    print("data upload complete")

    step("done -- what remains")
    if not space_ok:
        print("1. Subscribe to PRO (huggingface.co/pro) -- required for "
              "Docker Spaces.")
        print("2. Re-run this script: the resumable uploader skips "
              "already-uploaded data and the Space steps will complete.")
        return 0
    print(f"1. Watch the build: https://huggingface.co/spaces/{space} "
          "(Build logs tab; 15-30 min).")
    print("2. If the build ran before your GitHub push: Settings -> "
          "Factory rebuild.")
    print("3. Verify in the app: data status panel all OK, then a default "
          "W39b run + one noise panel.")
    print("4. Invite users: Space Settings -> collaborators.")
    print(f"5. Optional: delete the staging copy at {STAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
