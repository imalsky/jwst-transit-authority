# PandExo unmodified-template policy report

This is a configuration-policy experiment, not the fixed-hardware estimator parity gate. `valid = false` means the executed PandExo row carried an operational warning; it is never presented as an unqualified valid configuration.

| star | mode | status | config match | PandExo default | tool fixed | warnings |
|---|---|---:|---:|---|---|---|
| w39_like | nirspec_prism | WARNING | yes | sub512/nrsrapid/clear/prism | sub512/nrsrapid/clear/prism | All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution. |
| w39_like | nirspec_g395h | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395h | sub2048/nrsrapid/f290lp/g395h | - |
| w39_like | nirspec_g235h | EXECUTED | yes | sub2048/nrsrapid/f170lp/g235h | sub2048/nrsrapid/f170lp/g235h | - |
| w39_like | niriss_soss | WARNING | no | substrip96/nisrapid/clear/gr700xd | substrip256/nisrapid/clear/gr700xd | Optimized NGROUPS (43) exceeds the maximum (30). SET TO NGROUPS=30 |
| w39_like | nircam_f322w2 | WARNING | no | subgrism64/bright1/f322w2/grismr | subgrism64/rapid/f322w2/grismr | Optimized NGROUPS (245) exceeds the maximum (100). SET TO NGROUPS=100; Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| w39_like | nircam_f444w | WARNING | no | subgrism64/bright1/f444w/grismr | subgrism64/rapid/f444w/grismr | Optimized NGROUPS (465) exceeds the maximum (100). SET TO NGROUPS=100; Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| w39_like | miri_lrs | EXECUTED | no | slitlessprism_ip/fastr1/-/p750l | slitlessprism/fastr1/-/- | - |
| w39_like | nirspec_g395m | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395m | sub2048/nrsrapid/f290lp/g395m | - |
| bright_hot | nirspec_prism | WARNING | yes | sub512/nrsrapid/clear/prism | sub512/nrsrapid/clear/prism | All good. Ngroups=1 is a new mode since Cycle 4 and has not been rigorously tested. Proceed with caution.; Full saturation:
 There are 96 pixels saturated at the end of the first group. These pixels cannot be recovered.; % full well>80% (360% > 80%); Optimized NGROUPS below minimum (1). SET TO NGROUPS=1 |
| bright_hot | nirspec_g395h | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395h | sub2048/nrsrapid/f290lp/g395h | - |
| bright_hot | nirspec_g235h | EXECUTED | yes | sub2048/nrsrapid/f170lp/g235h | sub2048/nrsrapid/f170lp/g235h | - |
| bright_hot | niriss_soss | EXECUTED | no | substrip96/nisrapid/clear/gr700xd | substrip256/nisrapid/clear/gr700xd | - |
| bright_hot | nircam_f322w2 | WARNING | no | subgrism64/bright1/f322w2/grismr | subgrism64/rapid/f322w2/grismr | Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| bright_hot | nircam_f444w | WARNING | no | subgrism64/bright1/f444w/grismr | subgrism64/rapid/f444w/grismr | Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| bright_hot | miri_lrs | EXECUTED | no | slitlessprism_ip/fastr1/-/p750l | slitlessprism/fastr1/-/- | - |
| bright_hot | nirspec_g395m | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395m | sub2048/nrsrapid/f290lp/g395m | - |
| faint_k | nirspec_prism | EXECUTED | yes | sub512/nrsrapid/clear/prism | sub512/nrsrapid/clear/prism | - |
| faint_k | nirspec_g395h | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395h | sub2048/nrsrapid/f290lp/g395h | - |
| faint_k | nirspec_g235h | EXECUTED | yes | sub2048/nrsrapid/f170lp/g235h | sub2048/nrsrapid/f170lp/g235h | - |
| faint_k | niriss_soss | WARNING | no | substrip96/nisrapid/clear/gr700xd | substrip256/nisrapid/clear/gr700xd | Optimized NGROUPS (422) exceeds the maximum (30). SET TO NGROUPS=30 |
| faint_k | nircam_f322w2 | WARNING | no | subgrism64/bright1/f322w2/grismr | subgrism64/rapid/f322w2/grismr | Optimized NGROUPS (2053) exceeds the maximum (100). SET TO NGROUPS=100; Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| faint_k | nircam_f444w | WARNING | no | subgrism64/bright1/f444w/grismr | subgrism64/rapid/f444w/grismr | Optimized NGROUPS (3668) exceeds the maximum (100). SET TO NGROUPS=100; Selected BRIGHT1 after RAPID exceeded the 15 GB recommendation. Estimate assumes no target acquisition and a standard 2,100-second initial slew. |
| faint_k | miri_lrs | EXECUTED | no | slitlessprism_ip/fastr1/-/p750l | slitlessprism/fastr1/-/- | - |
| faint_k | nirspec_g395m | EXECUTED | yes | sub2048/nrsrapid/f290lp/g395m | sub2048/nrsrapid/f290lp/g395m | - |

Completeness gate: **PASS**.
