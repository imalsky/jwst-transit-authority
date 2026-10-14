"""deploy/pins.env and the Space Dockerfile must name the same sibling
commits: pins.env is the manifest CI installs from, the Dockerfile ARG
defaults are what the Space image enforces at build time. Drift between
them means CI validates different sibling code than the deployment ships.

CITATION.cff is pinned here for the same reason: it is release metadata that
nothing else checks. `cffconvert --validate` in CI checks the CFF schema, not
that the advertised version is the one the package ships."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

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
        "the Dockerfile must pin the deployed jwst-transit-authority commit by "
        "full SHA (no unqualified branch clones)")
    # 40 hex is not enough: a SHA from a sibling repo, or one that was rebased
    # away, passes that and deploys code this checkout does not contain.
    sha = args["JWST_TOOL_SHA"]

    def git(*a) -> int:
        return subprocess.run(("git", "-C", str(REPO)) + a,
                              capture_output=True).returncode

    if git("rev-parse", "--git-dir") != 0:
        pytest.skip("not a git checkout")
    if git("cat-file", "-e", f"{sha}^{{commit}}") != 0:
        # CI checks out at depth 1, so the pinned object is normally absent
        pytest.skip(f"{sha[:12]} not present locally (shallow clone)")
    assert git("merge-base", "--is-ancestor", sha, "HEAD") == 0, (
        f"the Dockerfile pins {sha[:12]}, which is not an ancestor of HEAD")


def test_space_build_and_boot_read_only_files_this_repo_ships():
    """The Dockerfile reads the lock files out of the tool clone at
    JWST_TOOL_SHA, so a misspelt or uncommitted one fails only on the Space,
    at build. The PICASO file list and md5s have one home, jwst_tool.datacheck:
    every deploy script that stages, seeds, checks or links PICASO data reads
    them from there, never from a copy that can drift."""
    shim = DOCKERFILE.read_text() + (DOCKERFILE.parent / "entrypoint.sh").read_text()
    refs = set(re.findall(r"jwst-transit-authority/(deploy/[\w.+-]+)", shim))
    assert {"deploy/picaso-conda-lock.txt", "deploy/requirements-picaso-lock.txt"} <= refs
    for ref in sorted(refs):
        assert (REPO / ref).is_file(), f"{ref} is read on the Space but not in the repo"
    assert "JWST_TOOL_PICASO_PYTHON=/opt/conda/envs/picaso/bin/python" in shim

    for name in ("upload_data.sh", "bootstrap_data.py", "finish_hf_setup.py",
                 "entrypoint.sh"):
        text = (DOCKERFILE.parent / name).read_text()
        assert re.search(r"from jwst_tool import datacheck as d", text), name
        assert {"PICASO_REFDATA_FILES", "PICASO_CK_MD5"} <= set(
            re.findall(r"\bd\w*\.(PICASO_\w+)", text)), name
        assert not re.search(r"_1460\.hdf5|\b[0-9a-f]{32}\b|refdata\.md5", text), name


def test_space_boots_vulcan_when_picaso_data_is_missing(tmp_path, monkeypatch,
                                                         capsys):
    """PICASO is optional on the Space: its files may trigger the dataset
    download, but one still missing afterwards only warns. A missing VULCAN
    marker still stops the boot."""
    import importlib.util
    import sys
    import types

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    spec = importlib.util.spec_from_file_location(
        "bootstrap_data", DOCKERFILE.parent / "bootstrap_data.py")
    boot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot)
    required = tmp_path / "VERSION_DATA"
    required.touch()
    calls = []
    monkeypatch.setattr(boot, "DATA", tmp_path)
    monkeypatch.setattr(boot, "MARKERS", [required])
    monkeypatch.setattr(boot, "ktable_markers", lambda: [])
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=lambda **kw: calls.append(kw)))

    assert boot.main() == 0 and len(calls) == 1      # PICASO absent: download, warn
    assert "PICASO files absent" in capsys.readouterr().err

    def offline(**kw):
        raise OSError("Hub unreachable")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=offline))
    assert boot.main() == 0                          # only PICASO absent: boots offline
    assert "booting without the PICASO data" in capsys.readouterr().err
    required.unlink()
    with pytest.raises(OSError, match="unreachable"):
        boot.main()                                   # VULCAN data absent: the error stands
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        snapshot_download=lambda **kw: calls.append(kw)))
    with pytest.raises(RuntimeError, match="required paths"):
        boot.main()


def test_citation_version_matches_the_package():
    """CITATION.cff advertises the version a citer will quote.

    CI validates only the CFF schema, so pin the value, not just the syntax.
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
