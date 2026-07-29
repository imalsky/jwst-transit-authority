# Hugging Face Space setup runbook

One-time setup. Cost: a PRO subscription (~$9/mo -- REQUIRED to host any
Docker Space, even on free hardware; the create call 402s without it) +
$5/mo persistent storage + $0.03/hr CPU Upgrade hardware while awake
(~$14/mo fixed; with a 1 h sleep timer and light use, typically under
$20/mo total). Prices approximate. Dataset repos and the data upload are
free and work without PRO.

## Fast path (scripted)

After the prerequisite in step 0 below, plus an HF account with PRO
(huggingface.co/pro) and billing configured (hf.co/settings/billing):

    pip install -U huggingface_hub
    hf auth login
    python deploy/hf-space/finish_hf_setup.py

The script replaces manual steps 1-6: it creates both private repos,
pushes the Space files, sets the HF_TOKEN secret, requests storage +
CPU Upgrade + the 1 h sleep timer, and runs the resumable ~7.5 GB data
upload (it stages nothing itself -- run `./upload_data.sh` first if the
staging dir is missing; re-run the script if the upload is interrupted).
Then continue at step 7 (verify). The manual steps below are the fallback
and the reference for what the script does.

## 0. Prerequisite

All three repos pushed to GitHub INCLUDING vulcan-jwst-tool's `deploy/`
directory -- the Space build reads the version pins from the cloned repo's
`deploy/requirements-app-lock.txt` and `deploy/requirements-pandeia.txt`
(regenerate the lock with `deploy/make_app_lock.py` after env changes).

## 1. Hugging Face CLI (Mac)

    pip install -U "huggingface_hub[cli]"
    hf auth login

(`huggingface-cli login` on older installs; upload_data.sh accepts either.)

## 2. Create the private dataset repo (browser)

hf.co/new-dataset -> Owner: your account -> Name: `vulcan-jwst-tool-data`
-> Private -> Create. (A different name/owner works too: pass it to the
upload script and set it as a `DATASET_REPO` variable on the Space.)

## 3. Upload the data (Mac; the slow step)

    cd /Users/imalsky/Desktop/Emulators/VULCAN_Project/vulcan-jwst-tool/deploy/hf-space
    ./upload_data.sh

Stages a dereferenced ~7.5 GB copy under ~/Desktop/hf_data_stage, then does
a resumable upload. Re-run the same command if the upload is interrupted.
Delete the staging dir when the Space is confirmed working.

## 4. Create the Space (browser)

Requires PRO (Docker Spaces are PRO-gated). hf.co/new-space -> Name:
`jwst-tool` -> SDK: Docker (blank template) -> Private -> CPU basic for
now -> Create.

## 5. Push the Space files (Mac)

    cd /Users/imalsky/Desktop/Emulators/VULCAN_Project/vulcan-jwst-tool/deploy/hf-space
    hf upload YOUR-HF-USERNAME/jwst-tool . . --repo-type space

This triggers the first image build (15-30 min; watch the Build logs tab).
The first boot will fail loudly until step 6 is done -- expected.

## 6. Configure the Space (browser, Settings tab)

- Volumes (the script does this; manual equivalent via
  `HfApi.set_space_volumes`): mount the dataset repo READ-ONLY at
  `/srv/hub-data` (the entrypoint then needs no download at boot) and a
  private bucket writable at `/data` for the caches. The legacy
  "persistent storage" endpoint is retired; buckets are its replacement.
- Variables and secrets: `HF_TOKEN` is only needed for the no-mount
  fallback (bootstrap download); prefer a fine-grained read token if set.
  If the dataset repo name differs from the default, add a variable
  `DATASET_REPO` = `owner/name`.
- Hardware: CPU Upgrade (8 vCPU / 32 GB, $0.03/hr), sleep time 1 h. NOTE:
  the API hardware endpoint refuses OAuth login tokens (compute scope), so
  set hardware + sleep in the browser unless you log in with a classic
  write token.
- Restart the Space.

## 7. Verify

First boot downloads the ~8 GB seed (minutes, HF-internal network), then
the GUI comes up. Check the in-app data status panel (every dataset row
OK), then run the default W39b forward model end-to-end plus one noise
panel (the noise job exercises the pandeia env and the release-match gate).
Wakes from sleep skip the download (markers present).

## 8. Invite users

Space Settings -> add collaborators by HF username (they need free HF
accounts; only the Space needs sharing, not the dataset repo -- data access
uses your HF_TOKEN). PUBLIC-ACCESS POLICY (updated 2026-07-29): the Space
was made deliberately public on 2026-07-20 (since 0.18.0 the visitor
orientation is the main-page header + the collapsed "How the model works"
section; the former mandatory intro gate was removed in the UX pass). The entrypoint still disables Streamlit's XSRF/CORS protection
(required for the T-P / noise-floor table uploads behind the Spaces
proxy), so two mitigations apply: uploaded tables are parsed by the tool's
own strict validators and stored content-addressed (never executed or
templated), and `jwst_tool.runlimit` caps concurrent heavy subprocesses at
2 per instance so visitors cannot pile solvers onto the hardware. If
either mitigation is ever removed, make the Space private again.

## Updating code later

Push to GitHub, bump the SRC_STAMP line in this directory's Dockerfile to
the new commit SHA (the API factory-rebuild endpoint refuses OAuth tokens,
so the stamp is the cache-buster that forces a fresh clone), and push the
Space files. Caches on /data survive rebuilds and self-invalidate by
version keys. The exact SHAs baked into the running image are recorded at
/srv/vulcan/BUILD_INFO (shown in the build logs).

## Restarting after a platform failure

A Space can build correctly and still fail to start, with a runtime error
naming a platform init step rather than anything in this repo. The one seen
2026-07-29 was:

    Initialization step 'hf-mount' failed with exit code: 0. Reason:
    Container logs: ... No logs available

Empty reason, no application logs, and `get_space_runtime().hardware` is
None, meaning hardware was never allocated: the volume mount failed before
the app ran. Diagnose before acting, because two very different things
produce this:

- A MISSING mount source is real and a restart cannot fix it. Check both:
  `api.dataset_info("imalsky/vulcan-jwst-tool-data")` (the read-only tree)
  and `api.bucket_info("imalsky/jwst-tool-cache")` (the read-write cache).
  If one is gone or renamed, repair the volume configuration with
  `set_space_volumes` instead of restarting.
- Both sources present means the failure is platform-side and transient, so
  a restart is the remedy.

An `hf auth login` token cannot call the lifecycle endpoints: `restart_space`
answers 401 (surfaced confusingly as `RepositoryNotFoundError`, "Repository
Not Found for .../restart"), the same restriction the hardware and
factory-reboot endpoints have. Either click "Restart this Space" in the UI,
or bump the RESTART NONCE comment at the top of the Dockerfile and commit:
any Space commit re-runs the init steps, and because a comment does not bust
the Docker layer cache, every layer hits and the container simply starts
again (leave SRC_STAMP alone -- bumping it would force a pointless re-clone
of unchanged code).

## Operational lessons (2026-07-20 deployment, learned the hard way)

- The repository-commit rate limiter (256/hr) is PER-REPO, far stickier
  than its message suggests, and rejected attempts keep it armed: never
  let an uploader retry-storm it; go silent, then make ONE attempt.
- Kill background uploaders by PID and verify: `pkill` on a wrapper can
  orphan the python child, which then storms the limiter invisibly.
- Small binary files are inlined into commits and REJECTED unless the
  repo's .gitattributes routes their extension through LFS/xet. On any
  new dataset repo, upload a .gitattributes with `*.fits`/`*.pdf` lfs
  rules BEFORE committing data via `upload_folder`.
- `upload-large-folder` is resumable and xet-native, but with multiple
  workers on a home uplink the bulk transfer starves its own commit
  calls (408s); one worker is faster in practice.
- Dataset volume mounts update live as commits land; bucket volumes are
  the writable persistent storage (the legacy per-Space storage API is
  retired).
- radis needs WRITE access inside exojax_linelists even for cache reads
  (scratch tempdir), and `cp -a` from a read-only mount preserves the
  read-only modes: sync + `chmod -R u+wX`.

## Known limitations

- Same as the VM deployment: no concurrent-run queue yet; the "legacy"
  Pandeia 3.0 backend is not deployed.
- The image has never been built (no Docker on the Mac) -- the first Space
  build is the real test; read the Build logs on failure.
- Sleep wake-up takes 1-3 min before the GUI responds.
