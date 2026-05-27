# Common Envelope Evolution — WD Binary Reconstruction

A pipeline for reconstructing the common envelope (CE) evolution of white dwarf (WD) binary systems. Given observed binary parameters (WD mass, companion mass, orbital period), it infers the progenitor donor mass and CE efficiency parameter α via a grid search followed by MCMC sampling.

## Method

1. **SSE grid** — evolves a grid of single-star progenitors using COSMIC/BSE to identify giant-phase snapshots whose core mass matches the observed WD mass.
2. **CE grid** — for each snapshot, computes the CE envelope-binding-energy parameter λ (Claeys et al. 2014 prescription) and inverts the α–λ energy formalism to find the pre-CE orbital separation.
3. **MCMC** — samples the posterior over (M₁, α) using `emcee`, initialised from grid survivors.
4. **Plots** — walker traces, a corner plot, and a COSMIC best-fit evolution plot.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config example.ini
```

### Configuration file

All run parameters are set in an `.ini` file. See `example.ini` for a template.

| Section | Key | Description |
|---------|-----|-------------|
| `[binary]` | `stage_name` | Label used for logging |
| | `log_file` | Path to the output log file |
| | `m_wd` | Observed WD mass (M☉) |
| | `sigma_mwd` | Uncertainty on WD mass (M☉) |
| | `m_bd` | Companion mass (M☉) |
| | `p_obs` | Observed orbital period (days) |
| | `a_f` | *(optional)* Final separation (R☉); derived from `p_obs` if omitted |
| `[paths]` | `output_csv` | Grid results CSV |
| | `chain_csv` | MCMC chain CSV |
| | `walkers_plot` | Walker trace PNG |
| | `corner_plot` | Corner plot PNG |
| | `evolution_plot` | Best-fit COSMIC evolution PNG |
| `[mcmc]` | `n_burnin` | Burn-in steps (default 500) |
| | `n_steps` | Production steps (default 2000) |
| | `t_max_plot` | Max time for evolution plot in Myr (default 14000) |

## Output

| File | Description |
|------|-------------|
| `results/grid_results.csv` | All (M₁, α) grid solutions that pass physical bounds |
| `results/chain.csv` | Flattened MCMC chain: columns `M1`, `alpha` |
| `results/walkers.png` | Walker trace for each parameter |
| `results/corner.png` | Corner plot of the posterior |
| `results/best_fit_evolution.png` | COSMIC evolution for the median posterior solution |

## Module overview

| File | Description |
|------|-------------|
| `main.py` | Entry point: argument parsing, MCMC, and plotting |
| `grid_runner.py` | Orchestrates the CE parameter grid over (M₁, α) |
| `sse_grid.py` | Builds the SSE grid of giant-phase snapshots via COSMIC |
| `lambda_claeys2014.py` | λ prescription from Claeys et al. (2014) |
| `ce_energy_inversion.py` | α–λ energy-balance inversion for the pre-CE separation |
| `logger.py` | Rotating file + console logger |
| `example.ini` | Example configuration file |

## References
- Theoretical uncertainties of the Type Ia supernova rate
J. S. W. Claeys, O. R. Pols, R. G. Izzard, J. Vink and F. W. M. Verbunt A&A, 563 (2014) A83 DOI: https://doi.org/10.1051/0004-6361/201322714
- Monica Zorotovic, MatthiasR Schreiber, Close detached white dwarf + brown dwarf binaries: further evidence for low values of the common envelope efficiency, Monthly Notices of the Royal Astronomical Society, Volume 513, Issue 3, July 2022, Pages 3587–3595, https://doi.org/10.1093/mnras/stac1137
- Post-common-envelope binaries from SDSS - IX: Constraining the common-envelope efficiency
-M. Zorotovic, M. R. Schreiber, B. T. Gänsicke and A. Nebot Gómez-Morán
A&A, 520 (2010) A86
DOI: https://doi.org/10.1051/0004-6361/200913658
