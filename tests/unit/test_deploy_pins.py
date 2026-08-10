"""deploy/pins.env and the Space Dockerfile must name the same sibling
commits: pins.env is the manifest CI installs from, the Dockerfile ARG
defaults are what the Space image enforces at build time. Drift between
them means CI validates different sibling code than the deployment ships
(2026-08-09 review round 2, finding 6)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PINS = REPO / "deploy" / "pins.env"
DOCKERFILE = REPO / "deploy" / "hf-space" / "Dockerfile"
_SHA = r"([0-9a-f]{40})"


def _pins() -> dict:
    out = {}
    for line in PINS.read_text().splitlines():
        m = re.match(rf"^([A-Z_]+)={_SHA}$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def _dockerfile_args() -> dict:
    out = {}
    for m in re.finditer(rf"^ARG ([A-Z_]+)={_SHA}$",
                         DOCKERFILE.read_text(), re.MULTILINE):
        out[m.group(1)] = m.group(2)
    return out


def test_sibling_pins_are_full_shas_and_agree():
    pins = _pins()
    args = _dockerfile_args()
    assert set(pins) == {"VULCAN_JAX_SHA", "VULCAN_FORWARD_SHA"}, pins
    for name, sha in pins.items():
        assert args.get(name) == sha, (
            f"{name}: pins.env has {sha} but the Dockerfile ARG default is "
            f"{args.get(name)}; update both together (deploy/pins.env is "
            "the manifest)")


def test_dockerfile_pins_the_deployed_tool_commit():
    args = _dockerfile_args()
    assert "JWST_TOOL_SHA" in args, (
        "the Dockerfile must pin the deployed vulcan-jwst-tool commit by "
        "full SHA (no unqualified branch clones)")
