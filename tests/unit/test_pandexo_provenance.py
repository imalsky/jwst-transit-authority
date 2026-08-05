"""`_pandexo_commit` must identify the IMPORTED PandExo tree or return None.

The original bug: `git rev-parse` walks UP from its -C directory, so a
pip-installed package under site-packages reported whatever repository
enclosed the environment (measured: Homebrew's own HEAD came back as "the
PandExo commit"). The fix requires the imported package file to be TRACKED
by the repository that answers, and the repository to ship PandExo's engine;
pip's direct_url.json is accepted only when the imported package lives under
that distribution's install root and the value is a full commit hash.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "parity" / "scripts"


@pytest.fixture(scope="module")
def worker():
    spec = importlib.util.spec_from_file_location(
        "pandexo_worker_probe", SCRIPTS / "pandexo_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def git_env(monkeypatch, tmp_path):
    """Hermetic git: no user/system config, identity from env."""
    if shutil.which("git") is None:                     # pragma: no cover
        pytest.skip("git not available")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")


def _git(cwd, *args) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, (args, r.stderr)
    return r.stdout.strip()


def _pandexo_checkout(root: Path) -> tuple[Path, str]:
    """A minimal repo that really versions a pandexo package; returns
    (package dir, HEAD)."""
    pkg = root / "pandexo"
    (pkg / "engine").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "engine" / "__init__.py").write_text("")
    (pkg / "engine" / "justdoit.py").write_text("")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "x")
    return pkg, _git(root, "rev-parse", "HEAD")


def test_unrelated_enclosing_repo_is_not_pandexo(worker, git_env, tmp_path):
    """The reproduced defect: a package directory inside SOME OTHER git repo
    (one with a pyproject.toml, like this one) must not inherit its HEAD."""
    repo = tmp_path / "enclosing"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname = 'x'\n")
    (repo / "pkg" / "__init__.py").write_text("")      # present, NOT tracked
    _git(repo, "init", "-q")
    _git(repo, "add", "pyproject.toml")
    _git(repo, "commit", "-q", "-m", "x")
    assert worker._pandexo_commit(str(repo / "pkg")) is None


def test_tracked_package_without_engine_is_not_pandexo(worker, git_env,
                                                       tmp_path):
    """Tracking the imported __init__.py is necessary but not sufficient:
    the repo must ship PandExo's engine (engine/justdoit.py)."""
    repo = tmp_path / "notpandexo"
    (repo / "pandexo").mkdir(parents=True)
    (repo / "pandexo" / "__init__.py").write_text("")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    assert worker._pandexo_commit(str(repo / "pandexo")) is None


def test_genuine_checkout_reports_head(worker, git_env, tmp_path):
    pkg, head = _pandexo_checkout(tmp_path / "PandExo")
    assert worker._pandexo_commit(str(pkg)) == head


def test_dirty_checkout_is_flagged(worker, git_env, tmp_path):
    pkg, head = _pandexo_checkout(tmp_path / "PandExo")
    (pkg / "engine" / "justdoit.py").write_text("changed")
    assert worker._pandexo_commit(str(pkg)) == head + "-dirty"


def test_worktree_checkout_reports_head(worker, git_env, tmp_path):
    """Worktrees keep .git as a FILE; the old fail-closed isdir(.git) check
    wrongly rejected them."""
    main = tmp_path / "PandExo"
    _pandexo_checkout(main)
    wt = tmp_path / "wt"
    _git(main, "worktree", "add", "-q", str(wt))
    head = _git(wt, "rev-parse", "HEAD")
    assert worker._pandexo_commit(str(wt / "pandexo")) == head


def test_plain_directory_returns_none(worker, git_env, tmp_path):
    d = tmp_path / "plain" / "pandexo"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    assert worker._pandexo_commit(str(d)) is None


def test_direct_url_accepted_only_with_containment_and_full_hash(
        worker, monkeypatch, tmp_path):
    import importlib.metadata as im

    site = tmp_path / "site-packages"
    pkg = site / "pandexo"
    pkg.mkdir(parents=True)
    commit = "a" * 40

    class FakeDist:
        def read_text(self, name):
            return json.dumps({"url": "https://github.com/x/PandExo.git",
                               "vcs_info": {"commit_id": commit}})

        def locate_file(self, p):
            return site / p

    monkeypatch.setattr(im, "distribution", lambda name: FakeDist())
    assert worker._pandexo_commit(str(pkg)) == commit

    # a package OUTSIDE the recorded distribution's install root must not
    # inherit its commit (shadowed second install)
    other = tmp_path / "elsewhere" / "pandexo"
    other.mkdir(parents=True)
    assert worker._pandexo_commit(str(other)) is None

    class ShortDist(FakeDist):
        def read_text(self, name):
            return json.dumps({"vcs_info": {"commit_id": "abc123"}})

    monkeypatch.setattr(im, "distribution", lambda name: ShortDist())
    assert worker._pandexo_commit(str(pkg)) is None
