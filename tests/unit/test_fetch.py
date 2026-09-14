"""jwst-tool fetch: spec sanity plus the download/extract mechanics,
exercised without any network (urlopen is monkeypatched)."""
import io
import tarfile

import pytest

from jwst_tool import fetch


def test_fetch_specs_are_well_formed():
    urls = [f.url for f in fetch.FETCHES]
    assert len(urls) == len(set(urls))
    for f in fetch.FETCHES:
        assert f.url.startswith("https://")
        assert f.size
        assert callable(f.dest)
    # the one tarball spec is the PHOENIX subtree
    assert [f.tar_subtree for f in fetch.FETCHES if f.tar_subtree] == \
        ["grp/redcat/trds/grid/phoenix"]


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


def test_extract_subtree_strips_prefix_and_requires_a_match(tmp_path):
    tar_path = tmp_path / "a.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, data in (("grp/redcat/trds/grid/phoenix/cat.fits", b"A"),
                           ("grp/redcat/trds/grid/phoenix/sub/m.fits", b"B"),
                           ("grp/other/junk.txt", b"C")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    dest = tmp_path / "phoenix"
    n = fetch._extract_subtree(tar_path, "grp/redcat/trds/grid/phoenix", dest)
    assert n == 2
    assert (dest / "cat.fits").read_bytes() == b"A"
    assert (dest / "sub" / "m.fits").read_bytes() == b"B"
    assert not (dest / "junk.txt").exists()
    # a prefix matching no member raises rather than "extracting" nothing
    with pytest.raises(RuntimeError):
        fetch._extract_subtree(tar_path, "grid/phoenix", tmp_path / "d2")


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
