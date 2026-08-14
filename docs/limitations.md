# Limitations

- Forecasts are local Fisher/Cramér-Rao bounds, not sampled posteriors. They
  can be optimistic near boundaries, for broad/non-Gaussian likelihoods, or
  when omitted parameters correlate with the requested ones.
- The noise model is a box-transit propagation of Pandeia extracted noise. It
  omits limb darkening, ingress/egress fitting, visit detrending, stellar
  heterogeneity, persistence, and physical correlated systematics.
- A wavelength-dependent or constant floor is only an effective diagonal
  minimum. It must never be described as a correlated-noise model.
- Detection templates remove one molecule from RT while holding the
  atmospheric state fixed. They do not represent a self-consistent
  molecule-free chemistry.
- File T-P forecasts condition on the supplied profile unless an explicit
  temperature parameter is present. A supplied profile is not evidence that
  it is appropriate for another planet.
- AD chemistry rows have narrower applicability than central finite
  differences and remain subject to per-row closure gates. A rank-deficient
  Fisher problem is reported, not repaired by a hidden prior.
- Short one-group NIR ramps and two-to-five-group MIRI ramps can carry
  calibration or approval concerns even when Pandeia returns numbers. The UI
  and policy report must preserve those warnings.
- Registry defaults are planning choices, not proof of APT schedulability or
  an operational recommendation.
- PICASO chemistry/climate is disabled and uncertified for this release. Its
  dependency requirements conflict with the validated ExoJAX environment and
  its native-RT cross-model artifact is stale and failing.
- Third-party reference data are not redistributed in the collaborator
  bundle. The manifest records source identifiers and checksums; recipients
  must acquire the data from the providers.
