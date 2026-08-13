---
title: VULCAN JWST Tool
sdk: docker
app_port: 7860
pinned: false
---

# VULCAN JWST Tool

JWST observability and information-content forecasts on live VULCAN-JAX
photochemical kinetics: forward transmission/emission spectra (ExoJax RT),
Pandeia 2026.7 noise (the STScI-supported matched triple), conditional
template S/N, and certified Fisher constraint forecasts.

This Space is a deployment shim: the build clones the three source repos
(jax-vulcan, vulcan-forward, vulcan-jwst-tool) from GitHub, and the ~8 GB
of reference data (Pandeia refdata + PSFs, synphot CDBS, exojax line lists,
opacity caches) is served from a private dataset repo mounted read-only at
/srv/hub-data (preferred; bootstrap_data.py download-seeding is the
no-mount fallback).

Operational notes:

- Requirements: persistent storage (Small tier), secret `HF_TOKEN` with read
  access to the dataset repo, CPU Upgrade hardware recommended.
- A forward model run takes minutes of CPU (photochemical kinetics to
  steady state); Fisher forecasts take 10-25 min depending on method.
  The in-app status panel reports data availability and progress.
- To update the code: push to GitHub, then Settings -> Factory rebuild
  (a factory rebuild avoids a stale cached clone layer).
- Setup runbook: the Deployment runbooks section (Hugging Face Space runbook)
  of the main repo's local notes.md.
