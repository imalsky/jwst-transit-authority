"""jwst-tool fetch: spec sanity plus the download/extract mechanics,
exercised without any network (urlopen is monkeypatched)."""
import dataclasses
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from jwst_tool import datacheck, fetch
from jwst_tool import instruments as ins

REPO = Path(__file__).resolve().parents[2]


def test_fetch_specs_are_well_formed():
    specs = fetch.FETCHES + fetch.PICASO_FETCHES
    urls = [f.url for f in specs]
    assert len(urls) == len(set(urls))
    for f in specs:
        assert f.url.startswith("https://")
        assert f.size
        assert callable(f.dest)
        assert bool(f.tar_subtree) == bool(f.marker)   # tarballs need a marker
    # the one default tarball spec is the PHOENIX subtree
    assert [f.tar_subtree for f in fetch.FETCHES if f.tar_subtree] == \
        ["grp/redcat/trds/grid/phoenix"]
    # --picaso: the tag's reference/ tree, then every pinned table, each
    # under the reference root and verified by its own pinned md5
    tree, *tables = fetch.PICASO_FETCHES
    assert tree.tar_subtree == "picaso-4.1/reference" and not tree.md5
    assert tree.check is fetch._picaso_tree_problem
    ck = Path(ins.PICASO_REFDATA) / datacheck.PICASO_CK_DIR
    assert {f.dest(): f.md5 for f in tables} == {
        ck / n: m for n, m in datacheck.PICASO_CK_MD5.items()}
    assert len(tables) == 36
    assert any(f.url.endswith("/H3%2B_1460.hdf5?download=1") for f in tables)
    # the env the recipe's two lock files build is the one the worker requires
    from jwst_tool.picaso_engine import PICASO_VERSIONS
    recipe = datacheck.PICASO_ENV_RECIPE
    assert "--no-deps" in recipe
    conda = (REPO / "deploy" / "picaso-conda-lock.txt").read_text()
    pip = (REPO / "deploy" / "requirements-picaso-lock.txt").read_text()
    assert "--file deploy/picaso-conda-lock.txt" in recipe
    assert "-r deploy/requirements-picaso-lock.txt" in recipe
    assert f"/photochem-{PICASO_VERSIONS['photochem']}-" in conda
    assert f"\npicaso=={PICASO_VERSIONS['picaso']}\n" in pip


def test_fetch_flags_and_picaso_opt_in(monkeypatch, capsys):
    from jwst_tool import cli
    monkeypatch.setattr(fetch, "_present", lambda f: True)     # no network
    monkeypatch.setattr(sys, "argv", ["jwst-tool", "fetch", "--bogus"])
    assert cli.main() == 2
    assert "--bogus" in capsys.readouterr().err
    assert fetch.run_fetch([]) == 0
    assert "PICASO" not in capsys.readouterr().out
    assert fetch.run_fetch(["--picaso"]) == 0
    out = capsys.readouterr().out
    assert out.count("already present: PICASO") == 37
    assert "JWST_TOOL_PICASO_PYTHON" in out


def test_manual_block_names_pieces_and_renders():
    from jwst_tool import instruments as ins

    txt = fetch.MANUAL
    # STScI's installation page, not a release-specific Box link: the page
    # always lists the supported release, so instructions cannot rot.
    assert "outerspace.stsci.edu" in txt
    assert "conda create" in txt
    assert "{refdata}" in txt and "{psf}" in txt
    # the triple must be stated, and the release must come from the backend
    assert "MATCHED TRIPLE" in txt
    assert "{release}" in txt and "{env_suffix}" in txt
    rel = ins.BACKEND_RELEASE
    rendered = fetch.MANUAL.format(refdata="/ref", psf="/psf", release=rel,
                                   env_suffix=rel.replace(".", "_"))
    assert f"pandeia_data-{rel}-jwst" in rendered
    assert f"pandeia_psfs-{rel}-jwst" in rendered
    assert f"pandeia.engine=={rel}" in rendered
    assert "{" not in rendered.replace("%2B", "")   # every field substituted


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_replaces_atomically_and_refuses_truncation(tmp_path,
                                                             monkeypatch):
    # full payload: streamed to <name>.part, atomically replaced
    payload = b"x" * 5000
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        lambda req: _FakeResponse(payload))
    out = tmp_path / "sub" / "file.bin"
    fetch._download("https://example.invalid/f", out, "test")
    assert out.read_bytes() == payload
    assert not out.with_suffix(".bin.part").exists()

    # a body shorter than Content-Length is refused and leaves nothing behind
    short = _FakeResponse(b"abc")
    short.headers = {"Content-Length": "9999"}
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda req: short)
    trunc = tmp_path / "trunc.bin"
    with pytest.raises(RuntimeError):
        fetch._download("https://example.invalid/f", trunc, "test")
    assert not trunc.exists()


def test_pinned_md5_gates_download_and_presence(tmp_path, monkeypatch):
    payload = b"y" * 3000
    good = hashlib.md5(payload).hexdigest()
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        lambda req: _FakeResponse(payload))
    out = tmp_path / "t.hdf5"
    with pytest.raises(RuntimeError, match="md5 mismatch"):
        fetch._download("https://example.invalid/t", out, "t", "0" * 32)
    assert not list(tmp_path.iterdir())            # no file, no .part left
    fetch._download("https://example.invalid/t", out, "t", good)
    spec = fetch.Fetch("t", "https://example.invalid/t", lambda: out, "3 KB",
                       md5=good)
    assert fetch._present(spec)
    # a file on disk with another digest is not "present": it is re-fetched
    assert not fetch._present(dataclasses.replace(spec, md5="0" * 32))


def test_extract_subtree_strips_prefix_and_requires_a_match(tmp_path,
                                                            monkeypatch):
    tar_path = tmp_path / "a.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, data in (("grp/redcat/trds/grid/phoenix/cat.fits", b"A"),
                           ("grp/redcat/trds/grid/phoenix/sub/m.fits", b"B"),
                           ("grp/other/junk.txt", b"C")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "phoenix"
    written = []
    copy = fetch.shutil.copyfileobj
    monkeypatch.setattr(fetch.shutil, "copyfileobj", lambda src, w: (
        written.append(Path(w.name).name), copy(src, w)))
    n = fetch._extract_subtree(tar_path, "grp/redcat/trds/grid/phoenix", dest,
                               marker="cat.fits")
    # the marker is written last, each member to a temp file renamed into
    # place, so a killed extraction never reads complete
    assert written == ["m.fits.part", "cat.fits.part"]
    assert n == 2
    assert not list(dest.rglob("*.part"))
    assert (dest / "cat.fits").read_bytes() == b"A"
    assert (dest / "sub" / "m.fits").read_bytes() == b"B"
    assert not (dest / "junk.txt").exists()
    # a prefix matching no member raises rather than "extracting" nothing
    with pytest.raises(RuntimeError):
        fetch._extract_subtree(tar_path, "grid/phoenix", tmp_path / "d2")

    # killed while writing the marker: no marker under its real name, not
    # even an empty one
    def die_on_marker(src, w):
        if Path(w.name).name == "cat.fits.part":
            raise KeyboardInterrupt
        copy(src, w)
    monkeypatch.setattr(fetch.shutil, "copyfileobj", die_on_marker)
    with pytest.raises(KeyboardInterrupt):
        fetch._extract_subtree(tar_path, "grp/redcat/trds/grid/phoenix",
                               tmp_path / "d3", marker="cat.fits")
    assert (tmp_path / "d3" / "sub" / "m.fits").is_file()
    assert not (tmp_path / "d3" / "cat.fits").exists()


def test_extract_subtree_refuses_a_member_escaping_the_destination(tmp_path):
    """The tarball is a remote file: a member whose path climbs out of the
    subtree must never be written outside ``dest``."""
    tar_path = tmp_path / "evil.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, data in (("pre/good.fits", b"A"), ("pre/../escape", b"B")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    with pytest.raises(RuntimeError, match="escapes"):
        fetch._extract_subtree(tar_path, "pre", tmp_path / "dest")
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("case", ["complete", "wrong_version", "tampered"])
def test_picaso_tree_counts_only_when_datacheck_and_md5_pass(
        tmp_path, monkeypatch, case):
    """The reference tree is present only when datacheck's version-and-files
    check passes AND each file matches its pinned md5; a fetched tree that
    fails either is refused and stays "missing", so the next run re-fetches."""
    files = {rel: rel.encode() for rel in datacheck.PICASO_REFDATA_MD5}
    files["config.json"] = json.dumps(
        {"version": "4.0" if case == "wrong_version" else "4.1"}).encode()
    # pins match the payload, so wrong_version is refused by datacheck alone
    monkeypatch.setattr(datacheck, "PICASO_REFDATA_MD5", {
        rel: hashlib.md5(data).hexdigest() for rel, data in files.items()})
    if case == "tampered":
        files["climate_INPUTS/ck_cx_cont_opacities_661.db"] = b"tampered"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for rel, data in files.items():
            info = tarfile.TarInfo(f"picaso-4.1/reference/{rel}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(fetch.urllib.request, "urlopen",
                        lambda req: _FakeResponse(buf.getvalue()))
    ref = tmp_path / "ref"
    tree = dataclasses.replace(fetch.PICASO_FETCHES[0], dest=lambda: ref)
    assert not fetch._present(tree)
    if case == "complete":
        fetch._fetch_one(tree)
        assert fetch._present(tree)
        return
    with pytest.raises(RuntimeError, match={
            "wrong_version": "version 4.0, need 4.1",
            "tampered": "md5 mismatch: climate_INPUTS/ck_cx"}[case]):
        fetch._fetch_one(tree)
    assert (ref / "config.json").is_file() and not fetch._present(tree)
