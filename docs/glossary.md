# Glossary

The names in the code that are not lower case, and the quantity behind each. Ruff's `extend-ignore-names` in
`pyproject.toml` allows them.

| Name | Quantity | Unit |
|---|---|---|
| `T`, `T_*`, `*_T`, `T0` | temperature | K |
| `P`, `P_*`, `*_P`, `dP` | pressure, its layer step | dyn cm^-2 or bar, as named |
| `Tirr`, `Tint` | Guillot irradiation and interior temperatures (Guillot 2010) | K |
| `Kzz`, `*Kzz*` | eddy diffusion coefficient | cm^2 s^-1 |
| `lnR0` | radius change at fixed planet mass, as a Fisher row | dimensionless |
| `R`, `R_bin` | resolving power, the analysis bins' | dimensionless |
| `M` | third-body number density in a rate law | cm^-3 |
| `Lz` | the layer the adjoint's loss is read at | index |
| `dL*` | derivative of the adjoint's loss | as named |
| `J`, `J_*`, `Jn`, `Jg` | Jacobian rows: raw, per mode, augmented with nuisances | depth per parameter |
| `F`, `Fw` | Fisher matrix, Jacobi-whitened | inverse parameter squared |
| `V` | eigenvectors of `Fw` | dimensionless |
| `U`, `A`, `An` | nuisance rows, their normal matrix, in correlation form | as in the detection formula |
| `S`, `N`, `K` | a sensitivity row, a count, the widget-key function | as named |

## References

Each entry below was resolved against Crossref. They cover the methods named in this directory.

- Batalha, N. E. & Line, M. R. 2017, AJ 153, 151, [10.3847/1538-3881/aa5faa](https://doi.org/10.3847/1538-3881/aa5faa)
- Batalha, N. E. et al. 2017, PASP 129, 064501, [10.1088/1538-3873/aa65b0](https://doi.org/10.1088/1538-3873/aa65b0) (PandExo)
- Birkmann, S. M. et al. 2022, A&A 661, A83, [10.1051/0004-6361/202142592](https://doi.org/10.1051/0004-6361/202142592)
- Greene, T. P. et al. 2016, ApJ 817, 17, [10.3847/0004-637X/817/1/17](https://doi.org/10.3847/0004-637X/817/1/17)
- Guillot, T. 2010, A&A 520, A27, [10.1051/0004-6361/200913396](https://doi.org/10.1051/0004-6361/200913396)
- Heng, K. & Kitzmann, D. 2017, ApJS 232, 20, [10.3847/1538-4365/aa8907](https://doi.org/10.3847/1538-4365/aa8907)
- Inglis, J. et al. 2024, ApJL, [10.3847/2041-8213/ad725e](https://doi.org/10.3847/2041-8213/ad725e)
- Kawahara, H. et al. 2022, ApJS 258, 31, [10.3847/1538-4365/ac3b4d](https://doi.org/10.3847/1538-4365/ac3b4d) (ExoJAX)
- Lodders, K. 2020, Oxford Research Encyclopedia of Planetary Science, [10.1093/acrefore/9780190647926.013.145](https://doi.org/10.1093/acrefore/9780190647926.013.145)
- Moses, J. I. et al. 2011, ApJ 737, 15, [10.1088/0004-637X/737/1/15](https://doi.org/10.1088/0004-637X/737/1/15)
- Mukherjee, S. et al. 2023, ApJ 942, 71, [10.3847/1538-4357/ac9f48](https://doi.org/10.3847/1538-4357/ac9f48) (PICASO climate)
- Pontoppidan, K. M. et al. 2016, Proc. SPIE, [10.1117/12.2231768](https://doi.org/10.1117/12.2231768) (Pandeia)
- Rustamkulov, Z. et al. 2023, Nature 614, 659, [10.1038/s41586-022-05677-y](https://doi.org/10.1038/s41586-022-05677-y)
- Sing, D. K. et al. 2024, Nature 630, 831, [10.1038/s41586-024-07395-z](https://doi.org/10.1038/s41586-024-07395-z)
- Tsai, S.-M. et al. 2023, Nature 617, 483, [10.1038/s41586-023-05902-2](https://doi.org/10.1038/s41586-023-05902-2)
- Vallisneri, M. 2008, PRD 77, 042001, [10.1103/PhysRevD.77.042001](https://doi.org/10.1103/PhysRevD.77.042001)
- Welbanks, L. et al. 2024, Nature 630, 836, [10.1038/s41586-024-07514-w](https://doi.org/10.1038/s41586-024-07514-w)
- Wogan, N. F. et al. 2025, PSJ, [10.3847/PSJ/ae0e1c](https://doi.org/10.3847/PSJ/ae0e1c) (Photochem)
- Zahnle, K. J. & Marley, M. S. 2014, ApJ 797, 41, [10.1088/0004-637X/797/1/41](https://doi.org/10.1088/0004-637X/797/1/41)

Two citations in the text could not be resolved: "Fortney et al. 2019" (eclipse depth at the tau = 2/3 radius) and "Agundez
2025" (class-B rate uncertainty of 0.65 dex). They stay as the code cites them, unverified.
