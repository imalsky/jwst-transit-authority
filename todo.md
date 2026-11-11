# todo

Open, owed or waiting on a decision, in priority order. Evidence is in notes.md, by
section name.

1. [blocker] v63 needs an unreleased vulcan-forward: its floor (0.31.0) and the deploy pins move with that release.
2. [decision] Fixed numbers and defaults: which fixed numbers become settings, and two defaults where the GUI and the
   API disagree.
3. [decision] The finite-difference h-vs-2h tolerance 0.25 has a measured spread but no derivation.
4. [upstream] The eclipse photosphere band reads vulcan-forward internals; it needs a public accessor.
5. [stale] The README example figure was not regenerated at v63.

## 1. The engine release v63 depends on

Evidence: notes "Current state". v63 calls `vulcan_chem.select_network` and passes `network`, `atom_list`,
`vulcan_cfg_name`, `art_pbtm_bar` and `p_ref_bar` in the profile; the engine that reads them is vulcan-forward 0.31.0,
uncommitted in its working tree. `pyproject.toml` already floors 0.31.0; `deploy/pins.env`, the Dockerfile and CI still
pin vulcan-forward 0accd36, on which a default run fails. Order: vulcan-forward's version and push, then the pins
(`deploy/pins.env`, the Dockerfile, CI) and this repo's push. Never deploy the Space from a mixed pair.

Done when the pins name a released, tested vulcan-forward commit.

## 2. Fixed numbers and defaults

Evidence: notes "Decision records" (Fixed numbers and request defaults). Done: the request defaults have one definition
in `model_defs` that `canonical_params`, the sidebar and `share_config.WIDGET_RANGES` read; the result-changing inline
numbers of the forward model and the measurement operator are named constants with unit and reason. Open, the
maintainer's choice:

- Which fixed numbers become settings (the FD steps among them; today all stay fixed).
- The sidebar defaults `jac_method` to "ad", the API to "fd"; the sidebar preselects `EXTRA_MOLECULES_DEFAULT`, the API
  runs no extras. Unifying either moves a shipped default.
- Defaults and fixed numbers with no recorded basis: `KZZ_CONST_DEFAULT` 1e9, `KZZ_KDEEP_DEFAULT` 1e5,
  `LOG_GAMMA_DEFAULT` -1, `LOG_KAPPA_CLOUD_DEFAULT` -1, the Guillot kappa, gamma and cloud FD steps (0.05 dex),
  `binning.LSF_R_FLOOR` 5 (a raise of a native R below it, not a refusal), `certify.EMIS_TAU_WARN` 10,
  `tp_tables.T_CHECK_POINTS` 200, `planets.TIRR_RANGE`.

Done when each is a setting or a fixed number with a recorded basis, by the maintainer's choice.

## 3. The finite-difference tolerance

Evidence: notes "Jacobians and Fisher". `jacobian.FD_CONSISTENCY_TOL` 0.25 separates every recorded certified row from
every recorded refused one, but nothing derives it. The maintainer keeps it, or sets it from a stated error budget.

## 4. A public eclipse-photosphere accessor

Evidence: notes "Convergence certificate and element gate". `spectrum.emission_photosphere_bar` reads
`_Emission._dtau`, `_g_layers`, `lnp_em` and `_rt_geometry._photosphere_lnp`, pinned by
`tests/unit/test_engine_api_contract.py`. The item lives in vulcan-forward: a public return of the tau = 2/3 pressure
(per g-ordinate and band) from `eclipse_flux_tau`. Done when the tool calls it and drops the private names.

## 5. The README example figure

`validation/scripts/fig_readme_example.py` (writes `assets/w39b_so2_forecast.png`) runs the app at its defaults and
was last made at v61; the v63 planet values move the default WASP-39 b transit depth by about 7 %. Run it after a
default GUI run has filled `output/`. Done when the figure is regenerated at v63 and notes "Jacobians and Fisher"
carries its numbers.
