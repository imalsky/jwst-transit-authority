"""The PandExo parity harness must FAIL CLOSED.

A runner that writes a summary and exits 0 no matter what lets stale or
saturated artifacts look like passing release gates. The gate lives in the
import-safe `validation/parity/scripts/parity_gate.py`, shared by the runner, the
renderers, and these tests -- no exec tricks, no duplicated constants. Each
`_fails` block breaks one thing in a synthetic full-matrix passing summary
and proves `validate()` (or `validate_artifact()`) fails; related fail-closed
branches are grouped into one test each. The committed artifact must
re-validate as a current pass.
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
            # iteration made 4 tests hash-seed-flaky (which mode lands at
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


def _fails(gate, s, needle):
    """validate(s) must report a problem containing `needle`."""
    problems = gate["validate"](s)
    assert any(needle in p for p in problems), (needle, problems)
    return problems


def test_a_clean_artifact_passes(gate, passing):
    assert gate["validate"](passing) == []


def test_provenance_drift_fails(gate, passing):
    """Provenance identity is exact: worker version, PSF release (a mixed
    PSF release changes the extracted flux), the supported engine release
    (two sides can agree with each other and still both be archival), and
    the reviewed, clean PandExo commit (PandExo runs from master, so a
    version string alone does not identify it)."""
    _fails(gate, _mut(passing, OURS + ("worker_version",), 5),
           "worker_version=5")
    _fails(gate, _mut(passing, PDX + ("psf_version",), "2026.2"),
           "psf release differs")
    _fails(gate, _upd(_upd(passing, OURS, engine_version="2026.2",
                           refdata_version="2026.2", psf_version="2026.2"),
                      PDX, pandeia_engine_version="2026.2",
                      refdata_version="2026.2", psf_version="2026.2"),
           "not the supported")
    _fails(gate, _mut(passing, PDX + ("pandexo_commit",)),
           "PandExo git commit")
    for commit in ("0" * 40,                        # unreviewed, and dirty:
                   "34e42d81f782c8358bb30fc6e8cdf4b87e488263-dirty"):
        _fails(gate, _mut(passing, PDX + ("pandexo_commit",), commit),
               "is not the reviewed")
    _fails(gate, _mut(passing, PDX + ("pandexo_version",), "unknown"),
           "PandExo version is unknown")


def test_saturation_mask_evidence_is_required(gate, passing):
    """Per-pixel mask agreement is evidence, not decoration: a missing mask
    and a disagreeing mask must both fail."""
    _fails(gate, _mut(passing, R0 + ("saturation_mask",)),
           "saturation grids matched fraction")
    _fails(gate, _upd(passing, R0 + ("saturation_mask",),
                      full_equal_frac=0.999, full_disagree=1),
           "full-saturation mask agreement")


def test_row_status_gates(gate, passing):
    """Row status must come from the measurement, never a label. Artifacts
    have carried saturated rows labeled OK, one at 7.31x the saturation
    level (Pandeia's per-mode value, not the physical full well), and a
    validator that silently skips any non-OK/ERROR status hides a
    65.9%-pixel-match row behind SATURATED_ABOVE_LIMIT."""
    problems = _fails(gate, _mut(passing, R0 + ("sat_frac_ours",),
                                 0.8548803964), "above the")
    assert any("above the" in p and "limit" in p for p in problems), problems
    _fails(gate, _mut(passing, R0 + ("sat_frac_ours",), 7.3110978), "7.3111")

    key = passing["stars"]["w39_like"]["modes"][0]["key"]
    _fails(gate, _mut(passing, R0, {"key": key, "status": "ERROR",
                                    "ours_error": "boom"}), "error row")
    _fails(gate, _mut(passing, R0 + ("status",), "SATURATED_ISH"),
           "unknown status")
    # SATURATED_ABOVE_LIMIT label while sat_frac (0.5) is BELOW the limit
    _fails(gate, _mut(passing, R0 + ("status",), "SATURATED_ABOVE_LIMIT"),
           "label must come from the measurement")


def test_matrix_membership_gates(gate, passing):
    """The full star x mode matrix is the experiment: nothing missing,
    nothing extra, nothing duplicated. Deleting whole stars must fail: a
    coverage-only check passes with two of three stars removed. One star
    losing a mode fails even though the coverage set is still satisfied by
    the other stars."""
    s = copy.deepcopy(passing)
    dropped = s["stars"]["w39_like"]["modes"].pop()
    _fails(gate, s, f"declared mode {dropped['key']!r} has no row")

    s = copy.deepcopy(passing)
    del s["stars"]["bright_hot"]
    del s["stars"]["faint_k"]
    _fails(gate, s, "missing from the artifact")

    _fails(gate, _mut(passing, ("stars", "mystery"),
                      copy.deepcopy(passing["stars"]["w39_like"])),
           "not part of the declared experiment")

    s = copy.deepcopy(passing)
    s["stars"]["w39_like"]["modes"].append(
        copy.deepcopy(s["stars"]["w39_like"]["modes"][0]))
    _fails(gate, s, "appears 2 times")

    s = copy.deepcopy(passing)
    s["stars"]["w39_like"]["modes"].append(_ok_row("nirspec_g140m"))
    _fails(gate, s, "'nirspec_g140m' is not part of the declared")


def test_config_and_measurement_drift_fails(gate, passing):
    """The declared experiment config, the matched pixel grid, and the
    extracted-flux ratio are each gated."""
    _fails(gate, _mut(passing, ("config", "depth"), 0.02),
           "does not match the declared experiment")
    _fails(gate, _mut(passing, R0 + ("npix_matched",), 900),      # 90%
           "usable pixels matched")
    _fails(gate, _mut(passing, R0 + ("flux_ratio", "median"), 1.05),
           "extracted-flux median ratio")


def test_config_mismatch_fails_outside_exact_miri_exception(gate, passing):
    """A real disagreement or missing value in any config field fails; the
    one exception is MIRI LRS's engine-derived disperser (ours unset,
    PandExo "p750l"), a representation difference, which must not."""
    s = copy.deepcopy(passing)
    row = next(r for r in s["stars"]["w39_like"]["modes"]
               if r["key"] == "miri_lrs")
    row["config_ours"]["disperser"] = None
    row["config_pandexo"]["disperser"] = "p750l"
    assert not [p for p in gate["validate"](s) if "disperser" in p]

    row["config_ours"]["readout"] = "FASTR1"
    _fails(gate, s, "submitted readout differs")

    for field in ("subarray", "readout", "filter", "disperser"):
        s = copy.deepcopy(passing)
        row = next(r for r in s["stars"]["w39_like"]["modes"]
                   if r["key"] != "miri_lrs")        # non-MIRI: no exception
        row["config_ours"][field] = None
        _fails(gate, s, f"submitted {field} differs")


def test_ngroup_rounding_tolerance_and_real_divergence(gate, passing):
    """At ngroup ~500 a 5-group rounding difference is 1.006%, so a pure 1%
    gate would fail the real, matched artifact; a real optimizer divergence
    still fails, and on a bright star a difference of two groups fails."""
    s = copy.deepcopy(passing)
    star = s["stars"]["faint_k"]
    star["modes"][0].update(ngroup_ours=497, ngroup_pandexo=492)   # 1.006%
    star["modes"][1].update(ngroup_ours=195, ngroup_pandexo=193)   # 1.03%
    assert not [p for p in gate["validate"](s) if "ngroup" in p]

    star["modes"][2].update(ngroup_ours=500, ngroup_pandexo=400)
    _fails(gate, s, "ngroup")

    _fails(gate, _upd(passing, R0, ngroup_ours=20, ngroup_pandexo=18),
           "ngroup 20 vs 18")


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


# --- short ramps, timing, sigma anomaly --------------------------------------

def test_short_ramp_policy(gate, passing):
    """1 vs 2 groups is a ramp-policy divergence, not rounding: whenever
    either side is at or below LOW_NGROUP_EXACT the counts must be equal.
    This is the committed bright_hot/niriss_soss regression. Exact
    agreement at 2 groups still passes."""
    assert gate["validate"](_upd(passing, BH_R0, ngroup_ours=2,
                                 ngroup_pandexo=2)) == []
    _fails(gate, _upd(passing, BH_R0, ngroup_ours=1, ngroup_pandexo=2),
           "short ramp")


def test_timing_and_sigma_anomaly_gates(gate, passing):
    """A 33% per-integration-time gap must fail even if the group diff
    passes its own tolerance; and the documented noise-model envelope tops
    out well under 2x, so a 7x median sigma ratio is an anomaly and must
    fail, not pass as a 'noise-model difference'."""
    _fails(gate, _upd(passing, BH_R0, t_int_ours_s=11.008,
                      t_int_pandexo_s=16.482), "per-integration time")
    _fails(gate, _mut(passing, BH_R0 + ("sigma_ratio_matched",),
                      {"median": 7.08}), "anomaly band")


# --- review round 2: no fail-open paths --------------------------------------

def test_ok_row_missing_gate_inputs_fails_closed(gate, passing):
    """An OK row missing the data a gate needs must FAIL that gate, never
    skip it: per-integration times for the timing gate, the sigma-ratio
    median for the anomaly gate."""
    _fails(gate, _mut(passing, R0 + ("t_int_ours_s",)),
           "timing gate cannot run")
    _fails(gate, _mut(passing, R0 + ("sigma_ratio_matched",), {}),
           "anomaly gate cannot run")


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


def test_saturated_row_claims_are_gated(gate, passing):
    """A SATURATED row must carry the measured fraction and PandExo's
    full-well verdict; a PandExo 'All good' on the same configuration is a
    saturation-status disagreement, and the two tools must saturate at the
    same ramp floor."""
    key = passing["stars"]["bright_hot"]["modes"][0]["key"]

    # honest saturated row passes
    assert gate["validate"](_mut(passing, BH_R0, _saturated_row(key))) == []
    _fails(gate, _mut(passing, BH_R0,
                      _saturated_row(key, fw="All good (42% < 80%)")),
           "saturation-status disagreement")
    no_verdict = _saturated_row(key)
    del no_verdict["pandexo_warnings"]
    _fails(gate, _mut(passing, BH_R0, no_verdict),
           "no PandExo full-well verdict")
    # a saturation claim without the measurement backing it
    _fails(gate, _mut(passing, BH_R0, _saturated_row(key, sat_frac_ours=0.5)),
           "must come from the measurement")
    # the two tools saturate at different ramp floors
    _fails(gate, _mut(passing, BH_R0, _saturated_row(key, ngroup_pandexo=2)),
           "floors the two tools saturate at")
