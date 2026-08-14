# Methods

## Forward model

The certified production path obtains a converged vertical composition from
VULCAN-JAX and passes the same pressure, temperature, abundance, gravity, and
radius state to `vulcan-forward`. Transmission uses an inverse-square gravity
profile in the altitude and chord optical-depth calculation. Emission reports
the bottom optical depth so an optically thin lower boundary cannot silently
look valid.

Metallicity scales the configured elemental abundances. C/O is the number
ratio C/O and changes carbon at fixed oxygen. The shipped WASP-39 b baseline is
checked against the loaded VULCAN configuration. Unsupported pressure,
temperature, abundance, profile, boundary-condition, convergence, or opacity
states raise instead of being clipped or substituted.

A molecule-removed spectrum changes only that molecule's RT opacity. It does
not recompute the chemistry, temperature, mean molecular weight, or radius.
The resulting detection statistic is therefore a nested RT-template planning
quantity, not a chemical model comparison.

## Pandeia noise

For native pixel `i`, extracted stellar count rate `F_i`, one-integration
extracted noise `s_i`, in/out integration counts `N_in` and `N_out`, and
`N_tr` independent transits, the box-depth approximation is

```text
var(d_i) = (s_i / F_i)^2 (1/N_in + 1/N_out) / N_tr.
```

Integration counts use complete cycles that fit inside each requested
baseline. In- and out-of-transit durations are independent inputs. Model,
Jacobian, and uncertainty vectors use one count-weighted binning operator:

```text
d_bin = sum(F_i d_i) / sum(F_i)
var(d_bin) = sum(F_i^2 var(d_i)) / sum(F_i)^2.
```

The user-selected floor is evaluated at final-bin wavelengths and applied as

```text
sigma_final = max(sqrt(var_random), floor).
```

It is an effective diagonal approximation. It does not represent a physical
time- or wavelength-correlated covariance and does not average down with more
transits. Partially and fully saturated native-pixel masks are retained in the
worker payload and are release-gated against PandExo.

## Fisher forecasts

Let `J` be the binned model Jacobian for the declared parameter coordinates and
`C` the diagonal variance used by the forecast. With nuisance matrix `A`, the
weighted projection is

```text
P = I - A (A^T C^-1 A)^+ A^T C^-1
F = J^T C^-1 P J.
```

The pseudoinverse is evaluated with a documented rank decision. Marginalized
precision comes from `F^+`; conditional precision uses the relevant diagonal
curvature without marginalizing the other requested rows. Combining modes
adds their Fisher matrices only after parameters are aligned by name.

Detection uses the same weighted nuisance projection on the molecule-removal
template. The asymptotic score is recomputed with the selected floor, so an
unreachable target is reported rather than assigned an arbitrary transit
count.

All displayed Gaussian curves are local Fisher/Cramér-Rao forecasts. They are
not posterior samples. Absolute C/O is the exact positive transform of the
local Gaussian in log C/O.

## Mock observations

A mock observation is one seeded draw from the final diagonal uncertainty. It
is overlaid after the deterministic forecast is complete. The seed and scheme
are exported; the draw cannot alter the Fisher covariance, detection score, or
transit requirement.
