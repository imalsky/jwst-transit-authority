# Deploying vulcan-jwst-tool on AWS

One always-on x86_64 VM running the Streamlit GUI in Docker, behind Caddy
(TLS + basic auth). The image carries the code and both Python environments;
all datasets and caches live on the instance disk under `/srv/vulcan-data`,
so cached models/noise persist across image rebuilds.

Everything in this directory was generated from the working Mac setup on
2026-07-19 (package pins captured from the live envs, paths from
`instruments.py` / `forward.config`). The image itself has NOT been
built yet -- there is no Docker on the Mac -- so the first
`server_setup.sh` run on the instance is the build's first real test.

## What runs where

| Piece | Where | Notes |
|---|---|---|
| GUI + JAX forward model | `app` container, conda env `app` | python 3.12, jax 0.9.2, exojax 2.2.3 |
| Pandeia noise worker | same container, conda env `pandeia` | pandeia.engine 2026.7 ("current" backend only; "legacy" 3.0 is not deployed). The engine, `pandeia_data-2026.7-jwst` and `pandeia_psfs-2026.7-jwst` must all be the same release: both trees must be in the uploaded dataset before this image is rebuilt, or every noise step errors (fail-closed by design) |
| Datasets (~8 GB) | `/srv/vulcan-data/{jwst-data,retrieval-data}` | uploaded once from the Mac |
| model/noise/adjoint caches | `/srv/vulcan-data/output` | persistent across rebuilds |
| TLS + auth | `caddy` container | config in `Caddyfile` + `.env` |

## Cost (approximate, on-demand, check current pricing)

- m7i-flex.2xlarge (8 vCPU / 32 GiB): roughly $220-260/mo. c7i-flex.2xlarge
  (16 GiB) is roughly $150-180/mo and workable for 1-2 concurrent runs.
- 150 GB gp3 disk: ~$12/mo. Egress: negligible for this app.
- Cheaper fixed-price alternative: Lightsail 8 vCPU / 32 GB / 640 GB SSD at
  a flat ~$160/mo including transfer (section 2b).
- Stopping the instance when idle stops compute billing (disk still bills);
  the Elastic IP + data survive stop/start.

## 1. Prerequisites (manual, once)

- An AWS account, and either `aws configure` credentials (script path) or
  console access (Lightsail path).
- An EC2 key pair in your region (EC2 console -> Key Pairs -> Create).
- Optional but recommended: a hostname you control (any provider) for real
  HTTPS. Without one, use `SITE_ADDRESS=:80` and an SSH tunnel.

## 2a. Provision with the script

    cd vulcan-jwst-tool/deploy
    KEY_NAME=your-keypair ALLOW_SSH_CIDR=$(curl -s ifconfig.me)/32 ./aws_provision.sh

Note the printed public IP. Allocate the Elastic IP it suggests if you want
a stable address.

## 2b. Or provision via Lightsail (no CLI)

Console -> Lightsail -> Create instance -> Linux, Ubuntu 24.04 -> the
$160/mo 8 vCPU / 32 GB plan. Networking tab: attach a static IP; open ports
80 and 443; restrict 22 to your IP. Then continue identically from step 3
(Lightsail instances are normal Ubuntu; user is `ubuntu`).

## 3. Pack and upload the data (Mac side, ~7.5 GB total)

    cd vulcan-jwst-tool/deploy
    ./make_data_bundles.sh

Then upload (this is the slow step; ~30-60 min on a typical home uplink):

    scp -i YOUR_KEY.pem ~/Desktop/jwst-data.tar ubuntu@SERVER_IP:~
    scp -i YOUR_KEY.pem ~/Desktop/retrieval-data.tar ubuntu@SERVER_IP:~

## 4. Set up the server

    ssh -i YOUR_KEY.pem ubuntu@SERVER_IP
    git clone https://github.com/imalsky/vulcan-jwst-tool.git
    ./vulcan-jwst-tool/deploy/server_setup.sh

The script installs Docker, clones the three repos (clone name `VULCAN-JAX`
is load-bearing), unpacks the data bundles into `/srv/vulcan-data`, builds
the image (15-30 min the first time), and stops to ask for `.env`.

## 5. Configure auth and address

    sudo docker run --rm caddy:2 caddy hash-password --plaintext 'CHOOSE-A-PASSWORD'
    nano ~/vulcan-src/vulcan-jwst-tool/deploy/.env

Set `BASIC_AUTH_HASH` to the printed hash and `SITE_ADDRESS` to your
hostname (after pointing its DNS A record at the server IP), or leave `:80`
for tunnel-only access. Then:

    cd ~/vulcan-src/vulcan-jwst-tool/deploy
    sudo docker compose up -d

## 6. Verify

    sudo docker compose exec app jwst-tool data

Every dataset row should read OK (this is the same detector the GUI status
panel uses; anything missing is named with a remedy). Then open the site
(or `ssh -i YOUR_KEY.pem -L 8080:localhost:80 ubuntu@SERVER_IP` and browse
http://localhost:8080), log in, and run the default W39b forward model
end-to-end including a noise panel -- the first noise job also exercises
the pandeia env + refdata release-match gate.

## Updating code later

    ssh -i YOUR_KEY.pem ubuntu@SERVER_IP
    cd ~/vulcan-src/VULCAN-JAX && git pull --ff-only
    cd ~/vulcan-src/vulcan-retrieval && git pull --ff-only
    cd ~/vulcan-src/vulcan-jwst-tool && git pull --ff-only
    sudo docker build -t jwst-tool:latest -f ~/vulcan-src/vulcan-jwst-tool/deploy/Dockerfile ~/vulcan-src
    cd ~/vulcan-src/vulcan-jwst-tool/deploy && sudo docker compose up -d

Caches self-invalidate correctly across upgrades (`forward._VERSION`,
noise `worker_version`, engine/refdata fingerprints are all in the keys).

## Known limitations of this deployment

- **No run queue yet.** Each browser session can launch a forward run
  (minutes of CPU); nothing caps concurrent runs. Fine for a handful of
  invited users; a semaphore in `app.py` is the first thing to add before
  widening access.
- **"legacy" Pandeia 3.0 backend is not in the image** (it needs the
  separate pinned env + an external refdata tree). `JWST_TOOL_BACKEND` is
  fixed to `current`.
- The env-gated slow FD-closure test and the pytest suite have not been run
  inside the image; the suite (`python -m pytest tests -q`, numpy-only,
  fast) can be run in the container as an extra check:
  `sudo docker compose exec app python -m pytest /srv/vulcan/vulcan-jwst-tool/tests -q`
- Container processes run as root inside the container (single-purpose box);
  acceptable for an invited-user science tool, revisit before fully public.
