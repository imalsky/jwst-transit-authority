---
title: JWST Transit Authority
sdk: docker
app_port: 7860
pinned: false
---

# JWST Transit Authority

JWST observability and information-content forecasts on live VULCAN-JAX
photochemical kinetics: forward transmission/emission spectra (ExoJAX RT),
Pandeia 2026.7 noise (the STScI-supported matched triple), conditional
template S/N, and certified Fisher constraint forecasts.

This Space is a deployment shim: the build clones the three source repos
(jax-vulcan, vulcan-forward, jwst-transit-authority) from GitHub, and the ~8 GB
of reference data (Pandeia refdata + PSFs, synphot CDBS, exojax line lists,
opacity caches) is served from a private dataset repo mounted read-only at
/srv/hub-data (preferred; bootstrap_data.py download-seeding is the
no-mount fallback).

Operational notes:

- Requirements: the dataset repo mounted read-only at `/srv/hub-data` plus a
  writable bucket volume at `/data` (buckets replaced the retired per-Space
  persistent storage); secret `HF_TOKEN` with read access to the dataset repo
  for the no-mount download fallback; CPU Upgrade hardware recommended.
- A forward model run takes minutes of CPU (photochemical kinetics to
  steady state); Fisher forecasts take tens of minutes depending on method.
  Space hardware is slower than the laptop the estimates were measured on
  (measured on the same case: ExoJAX RT ~4.5x, the VULCAN solve ~1.2x), and
  the app scales its displayed estimates by `_RUNTIME_SCALE`. Progress and
  remaining time are shown live while a run is in flight.
- To update the code: push to GitHub, bump the three `*_SHA` ARG defaults in
  the Dockerfile (and `deploy/pins.env` with them), then upload the Space
  files. Changing an ARG default busts the layers that clone the repos, so no
  factory rebuild is needed.
- Setup runbook: the Deployment runbooks section (Hugging Face Space runbook)
  of the main repo's local notes.md.
