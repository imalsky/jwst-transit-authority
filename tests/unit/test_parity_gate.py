"""The PandExo parity harness must FAIL CLOSED.

A runner that writes a summary and exits 0 no matter what lets stale or
saturated artifacts look like passing release gates. The gate lives in the
import-safe `validation/parity/scripts/parity_gate.py`, shared by the runner, the
renderers, and these tests -- no exec tricks, no duplicated constants. Each
`_CASES` row breaks one thing in a synthetic full-matrix passing summary and
proves `validate()` reports it (or a legitimate near-miss still passes). The
committed artifact must re-validate as a current pass.
"""
from __future__ import annotations

import copy
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "validation" / "parity" / "scripts"
PLOTS = SCRIPTS / "make_parity_plots.py"

# mutation-path shorthands for _mut/_upd
W39 = ("stars", "w39_like")
OURS = W39 + ("provenance_ours",)
PDX = W39 + ("provenance_pandexo",)
R0 = W39 + ("modes", 0)                  # first mode row, the usual target
BH_R0 = ("stars", "bright_hot", "modes", 0)
_DEL = object()


@pytest.fixture(scope="module")
def gate():
    """The shared gate module (parity_gate.py), imported -- never exec'd."""
    sys.path.insert(0, str(SCRIPTS))
    return vars(importlib.import_module("parity_gate"))


@pytest.fixture(scope="module")
def plots():
    """Load plotting guards without rendering a figure."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("parity_plots_probe", PLOTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ok_row(key, ngroup=100, sat=0.5):
    return {
        "key": key, "status": "OK",
        "sat_frac_ours": sat,
        "npix_pandexo": 1000, "npix_matched": 1000,
        "ngroup_ours": ngroup, "ngroup_pandexo": ngroup,
        "t_int_ours_s": 100.0, "t_int_pandexo_s": 100.0,
        "sigma_ratio_matched": {"median": 1.10},
        "config_ours": {"subarray": "sub", "readout": "RAPID",
                        "filter": "f", "disperser": "d"},
        "config_pandexo": {"subarray": "sub", "readout": "RAPID",
                           "filter": "f", "disperser": "d"},
        "flux_ratio": {"median": 1.001, "max_abs_dev": 0.02},
        "saturation_mask": {
            "n_ours": 1000, "n_pandexo": 1000, "n_matched": 1000,
            "matched_frac": 1.0, "partial_equal_frac": 1.0,
            "full_equal_frac": 1.0, "partial_disagree": 0,
            "full_disagree": 0,
        },
    }


@pytest.fixture
def passing(gate):
    """A full-matrix summary that satisfies every gate check."""
    rel = gate["REQUIRED_PANDEIA_RELEASE"]

    def star_block():
        return {
            "provenance_ours": {
                "worker_version": gate["PRODUCTION_WORKER_VERSION"],
                "engine_version": rel, "refdata_version": rel,
                "psf_version": rel,
            },
            "provenance_pandexo": {
                "pandeia_engine_version": rel, "refdata_version": rel,
                "psf_version": rel, "pandexo_version": "2026.7",
                "pandexo_commit": gate["REQUIRED_PANDEXO_COMMIT"],
            },
            # sorted(): REQUIRED_UNSATURATED_MODES is a set; unsorted
            # iteration makes tests hash-seed-flaky (which mode lands at
            # index 0 varies).
            "modes": [_ok_row(k)
                      for k in sorted(gate["REQUIRED_UNSATURATED_MODES"])],
        }

    return {
        "stars": {name: star_block() for name in gate["STARS"]},
        "config": {
            "transit_duration_s": gate["T_TRANSIT_S"],
            "depth": gate["DEPTH"],
            "sat_limit": gate["SAT_LIMIT"],
            "stars": copy.deepcopy(gate["STARS"]),
        },
    }


def _mut(passing, path, value=_DEL):
    """Deep-copied summary with the leaf at `path` replaced (or deleted)."""
    s = copy.deepcopy(passing)
    d = s
    for k in path[:-1]:
        d = d[k]
    if value is _DEL:
        del d[path[-1]]
    else:
        d[path[-1]] = value
    return s


def _upd(passing, path, **kv):
    """Deep-copied summary with the dict at `path` updated with `kv`."""
    s = copy.deepcopy(passing)
    d = s
    for k in path:
        d = d[k]
    d.update(kv)
    return s


def _drop_last_w39_mode(p):
    s = copy.deepcopy(p)
    s["stars"]["w39_like"]["modes"].pop()
    return s


def _drop_two_stars(p):
    s = copy.deepcopy(p)
    del s["stars"]["bright_hot"], s["stars"]["faint_k"]
    return s


def _append_w39_row(p, row=None):
    s = copy.deepcopy(p)
    modes = s["stars"]["w39_like"]["modes"]
    modes.append(copy.deepcopy(modes[0]) if row is None else row)
    return s


def _w39_config(p, miri, **ours):
    """Edit config_ours on the MIRI LRS row (or the first non-MIRI row); the
    MIRI row also gets PandExo's engine-derived disperser "p750l"."""
    s = copy.deepcopy(p)
    row = next(r for r in s["stars"]["w39_like"]["modes"]
               if (r["key"] == "miri_lrs") == miri)
    if miri:
        row["config_ours"]["disperser"] = None
        row["config_pandexo"]["disperser"] = "p750l"
    row["config_ours"].update(ours)
    return s


def _faint_ngroups(p, *pairs):
    s = copy.deepcopy(p)
    for row, (ours, pandexo) in zip(s["stars"]["faint_k"]["modes"], pairs):
        row.update(ngroup_ours=ours, ngroup_pandexo=pandexo)
    return s


def _saturated_row(key, fw="% full well>80% (360% > 80%)", **overrides):
    row = {"key": key, "status": "SATURATED",
           "ours_reason": "no unsaturated pixels at the shortest ramp",
           "ngroup_ours": 1, "sat_frac_ours": 3.56,
           "ngroup_pandexo": 1,
           "pandexo_warnings": {"% full well high?": fw},
           "saturation_mask": {
               "n_ours": 1000, "n_pandexo": 1000, "n_matched": 1000,
               "matched_frac": 1.0, "partial_equal_frac": 1.0,
               "full_equal_frac": 1.0, "partial_disagree": 0,
               "full_disagree": 0,
           }}
    row.update(overrides)
    return row


def _saturate_bh(p, drop_verdict=False, **kw):
    row = _saturated_row(p["stars"]["bright_hot"]["modes"][0]["key"], **kw)
    if drop_verdict:
        del row["pandexo_warnings"]
    return _mut(p, BH_R0, row)


_ARCHIVAL = dict(refdata_version="2026.2", psf_version="2026.2")

# (mutation of the passing summary, needle). None: it must still pass
# cleanly; otherwise the needle (or every string of a tuple) must appear in
# ONE reported problem.
_CASES = [
    # the clean artifact and the legitimate near-misses that must pass
    (lambda p: p, None),
    # MIRI LRS's engine-derived disperser (ours unset, PandExo "p750l") is a
    # representation difference, not a mismatch
    (lambda p: _w39_config(p, True), None),
    # at ngroup ~500 a 5-group rounding difference is 1.006%: a pure 1% gate
    # would fail the real, matched artifact
    (lambda p: _faint_ngroups(p, (497, 492), (195, 193)), None),
    # exact agreement at 2 groups is not a short-ramp divergence
    (lambda p: _upd(p, BH_R0, ngroup_ours=2, ngroup_pandexo=2), None),
    (lambda p: _saturate_bh(p), None),               # an honest SATURATED row
    # provenance identity is exact: worker version, PSF release, the supported
    # engine release (two sides can agree and still both be archival), and the
    # reviewed, clean PandExo commit (a version string does not identify master)
    (lambda p: _mut(p, OURS + ("worker_version",), 5), "worker_version=5"),
    (lambda p: _mut(p, PDX + ("psf_version",), "2026.2"), "psf release differs"),
    (lambda p: _upd(_upd(p, OURS, engine_version="2026.2", **_ARCHIVAL),
                    PDX, pandeia_engine_version="2026.2", **_ARCHIVAL),
     "not the supported"),
    (lambda p: _mut(p, PDX + ("pandexo_commit",)), "PandExo git commit"),
    (lambda p: _mut(p, PDX + ("pandexo_commit",), "0" * 40), "is not the reviewed"),
    (lambda p: _mut(p, PDX + ("pandexo_commit",),
                    "34e42d81f782c8358bb30fc6e8cdf4b87e488263-dirty"),
     "is not the reviewed"),
    (lambda p: _mut(p, PDX + ("pandexo_version",), "unknown"),
     "PandExo version is unknown"),
    # per-pixel mask agreement is evidence: missing or disagreeing fails
    (lambda p: _mut(p, R0 + ("saturation_mask",)),
     "saturation grids matched fraction"),
    (lambda p: _upd(p, R0 + ("saturation_mask",), full_equal_frac=0.999,
                    full_disagree=1), "full-saturation mask agreement"),
    # row status comes from the measurement, never a label: a saturated row
    # can carry an OK label (one measured at 7.31x Pandeia's per-mode level),
    # and skipping an unknown status lets a 65.9%-pixel-match row through
    (lambda p: _mut(p, R0 + ("sat_frac_ours",), 0.8548803964),
     ("above the", "limit")),
    (lambda p: _mut(p, R0 + ("sat_frac_ours",), 7.3110978), "7.3111"),
    (lambda p: _mut(p, R0, {"key": p["stars"]["w39_like"]["modes"][0]["key"],
                            "status": "ERROR", "ours_error": "boom"}),
     "error row"),
    (lambda p: _mut(p, R0 + ("status",), "SATURATED_ISH"), "unknown status"),
    (lambda p: _mut(p, R0 + ("status",), "SATURATED_ABOVE_LIMIT"),  # sat 0.5
     "label must come from the measurement"),
    # the full star x mode matrix is the experiment: one star losing a mode
    # fails though the coverage set is still met, and deleting stars fails
    (_drop_last_w39_mode,
     lambda p: f"declared mode {p['stars']['w39_like']['modes'][-1]['key']!r} has no row"),
    (_drop_two_stars, "missing from the artifact"),
    (lambda p: _mut(p, ("stars", "mystery"), copy.deepcopy(p["stars"]["w39_like"])),
     "not part of the declared experiment"),
    (_append_w39_row, "appears 2 times"),
    (lambda p: _append_w39_row(p, _ok_row("nirspec_g140m")),
     "'nirspec_g140m' is not part of the declared"),
    # declared config, matched pixel grid, extracted-flux ratio
    (lambda p: _mut(p, ("config", "depth"), 0.02),
     "does not match the declared experiment"),
    (lambda p: _mut(p, R0 + ("npix_matched",), 900), "usable pixels matched"),
    (lambda p: _mut(p, R0 + ("flux_ratio", "median"), 1.05),
     "extracted-flux median ratio"),
    # any other config disagreement or missing value fails, MIRI included
    (lambda p: _w39_config(p, True, readout="FASTR1"), "submitted readout differs"),
    *[(lambda p, f=f: _w39_config(p, False, **{f: None}), f"submitted {f} differs")
      for f in ("subarray", "readout", "filter", "disperser")],
    # a real optimizer divergence fails; on a bright star so do two groups
    (lambda p: _faint_ngroups(p, (497, 492), (195, 193), (500, 400)), "ngroup"),
    (lambda p: _upd(p, R0, ngroup_ours=20, ngroup_pandexo=18), "ngroup 20 vs 18"),
    # 1 vs 2 groups is a ramp-policy divergence, not rounding: at or below
    # LOW_NGROUP_EXACT the counts must be equal (the committed
    # bright_hot/niriss_soss regression)
    (lambda p: _upd(p, BH_R0, ngroup_ours=1, ngroup_pandexo=2), "short ramp"),
    # a 33% integration-time gap fails though the group diff passes; a 7x
    # median sigma ratio is an anomaly, not a noise-model difference (< 2x)
    (lambda p: _upd(p, BH_R0, t_int_ours_s=11.008, t_int_pandexo_s=16.482),
     "per-integration time"),
    (lambda p: _mut(p, BH_R0 + ("sigma_ratio_matched",), {"median": 7.08}),
     "anomaly band"),
    # an OK row missing a gate's input fails that gate, never skips it
    (lambda p: _mut(p, R0 + ("t_int_ours_s",)), "timing gate cannot run"),
    (lambda p: _mut(p, R0 + ("sigma_ratio_matched",), {}), "anomaly gate cannot run"),
    # a SATURATED row carries the measured fraction and PandExo's full-well
    # verdict, agrees with it, and saturates at the same ramp floor
    (lambda p: _saturate_bh(p, fw="All good (42% < 80%)"),
     "saturation-status disagreement"),
    (lambda p: _saturate_bh(p, drop_verdict=True), "no PandExo full-well verdict"),
    (lambda p: _saturate_bh(p, sat_frac_ours=0.5), "must come from the measurement"),
    (lambda p: _saturate_bh(p, ngroup_pandexo=2), "floors the two tools saturate at"),
]


@pytest.mark.parametrize("mutate,needle", _CASES)
def test_validate_fails_closed_on_every_broken_input(gate, passing, mutate,
                                                     needle):
    problems = gate["validate"](mutate(passing))
    if needle is None:
        assert problems == []
        return
    needle = needle(passing) if callable(needle) else needle
    needles = (needle,) if isinstance(needle, str) else needle
    assert any(all(n in p for n in needles) for p in problems), (needle, problems)


def test_validate_artifact_revalidates_and_pins_the_gate_block(gate, passing):
    """The renderers call validate_artifact(); a hand-edited artifact whose
    gate block says PASS must still be refused, as must tampered thresholds
    or a missing gate block."""
    s = copy.deepcopy(passing)
    s["gate"] = gate["gate_block"]([])
    assert gate["validate_artifact"](s) == []

    broken = _mut(s, ("stars", "faint_k"))      # break it, keep "passed": true
    problems = gate["validate_artifact"](broken)
    assert any("does not match a fresh validate" in p for p in problems), \
        problems

    tampered = _mut(s, ("gate", "thresholds", "max_flux_ratio_dev"), 0.5)
    problems = gate["validate_artifact"](tampered)
    assert any("does not match the current gate" in p for p in problems), \
        problems

    problems = gate["validate_artifact"](copy.deepcopy(passing))
    assert any("no gate block" in p for p in problems), problems


def test_gate_identity_is_single_sourced(gate):
    """Worker version and supported release come from production code and the
    deploy pin -- typed once, cross-checked here so they cannot drift."""
    from jwst_tool import instruments as ins
    from jwst_tool import noise

    assert gate["PRODUCTION_WORKER_VERSION"] == noise.WORKER_VERSION
    assert gate["REQUIRED_PANDEIA_RELEASE"] == ins._SUPPORTED_PANDEIA_RELEASE
    req = (Path(__file__).resolve().parents[2] / "deploy"
           / "requirements-pandeia.txt").read_text()
    assert f"pandeia.engine=={ins._SUPPORTED_PANDEIA_RELEASE}" in req, (
        "deploy/requirements-pandeia.txt does not pin the supported release")


def test_committed_artifact_is_a_current_scrubbed_pass(gate):
    """The committed artifact must BE a current pass, and stay one: fresh
    validation AND the persisted gate block agreeing with today's constants
    (a backend bump without a fresh parity run fails here). It must record
    the supported release in every provenance field and carry no
    machine-absolute paths (identity is names/releases/hashes, never
    workstation layout; run_parity scrubs)."""
    import json

    path = SCRIPTS.parents[1] / "parity" / "outputs" / "parity_summary.json"
    if not path.is_file():                              # pragma: no cover
        pytest.skip("committed parity_summary.json not present")
    text = path.read_text()
    summary = json.loads(text)

    problems = gate["validate_artifact"](summary)
    assert not problems, f"the committed artifact no longer passes: {problems}"

    # the two sides name the engine field differently ("engine_version" ours,
    # "pandeia_engine_version" PandExo's); both record all three releases
    want = gate["REQUIRED_PANDEIA_RELEASE"]
    for sname, star in summary["stars"].items():
        for side, engine_key in (("provenance_ours", "engine_version"),
                                 ("provenance_pandexo",
                                  "pandeia_engine_version")):
            prov = star[side]
            for field in (engine_key, "refdata_version", "psf_version"):
                assert str(prov.get(field)) == want, (
                    f"{sname}/{side}: {field}={prov.get(field)!r} is not the "
                    f"supported {want}")

    assert "/Users/" not in text and "/opt/" not in text and \
        "/home/" not in text, "machine-absolute paths leaked into the artifact"


def test_plots_revalidate_instead_of_trusting_the_passed_bit(plots, gate,
                                                             passing):
    """The renderers must refuse a summary with no gate block, re-run the
    gate rather than trust "passed": true, and accept a revalidated pass."""
    with pytest.raises(RuntimeError, match="no gate block"):
        plots.require_passing_summary({"stars": {}})

    s = copy.deepcopy(passing)
    s["gate"] = gate["gate_block"]([])
    broken = _mut(s, ("stars", "faint_k"))      # break it, keep "passed": true
    with pytest.raises(RuntimeError, match="failed its gate"):
        plots.require_passing_summary(broken)

    assert plots.require_passing_summary(s) == \
        gate["REQUIRED_PANDEIA_RELEASE"]
