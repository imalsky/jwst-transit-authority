"""deploy/pins.env and the Space Dockerfile must name the same sibling
commits: pins.env is the manifest CI installs from, the Dockerfile ARG
defaults are what the Space image enforces at build time. Drift between
them means CI validates different sibling code than the deployment ships.

CITATION.cff is pinned here for the same reason: it is release metadata that
nothing else checks. `cffconvert --validate` in CI checks the CFF schema, not
that the advertised version is the one the package ships."""
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


def test_citation_version_matches_the_package():
    """CITATION.cff advertises the version a citer will quote.

    It drifted six releases (0.48.2 while the package was 0.48.8) because CI
    validates only the CFF schema. Pin the value, not just the syntax.
    """
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from jwst_tool import __version__

    cff = (REPO / "CITATION.cff").read_text()
    m = re.search(r"^version:\s*(\S+)\s*$", cff, re.MULTILINE)
    assert m, "CITATION.cff has no version field"
    assert m.group(1) == __version__, (
        f"CITATION.cff says version {m.group(1)} but the package is "
        f"{__version__}; bump CITATION.cff in the release commit")
