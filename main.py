import argparse
import configparser
from logger import CE_logging
from grid_runner import CEGridRunner
import numpy as np
import emcee
from astropy.constants import c, M_sun
from lambda_claeys2014 import Lambda
from ce_energy_inversion import EnergyInversion
import corner
import matplotlib.pyplot as plt
from cosmic.plotting import evolve_and_plot
from cosmic.evolve import Evolve
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.utils import convert_kstar_evol_type
import pandas as pd
import tqdm as tqdm
from utils import bse_defaults, sse_defaults, period_to_separation
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
})


def arg_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstructing the CE evolution of WD binaries")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, str]:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    flat = {}
    for section in cfg.sections():
        for key, val in cfg.items(section):
            flat[key] = val
    return flat


def period_after_CE(t_cool_myr: float, mwd: float, mbd: float, porb_days: float) -> float:
    """Post-CE period (days) from gravitational radiation only (Schreiber & Gänsicke 2003 Eq. 8)."""
    tcool_sec = t_cool_myr * 1e6 * 3.15576e7
    M1 = mwd * M_sun.value
    M2 = mbd * M_sun.value
    factor = (
        (256 / (5 * c.value ** 5))
        * (6.674e-11) ** (5 / 3)
        * (2 * np.pi) ** (8 / 3)
        * M1 * M2
        * (M1 + M2) ** (-1 / 3)
    )
    pce_83 = (porb_days * 86400.0) ** (8 / 3) + factor * tcool_sec
    return pce_83 ** (3 / 8) / 86400.0


def _iter_sse_matches(mwd, m_bd, a_ce_rsun, t_cool, sse_df, sigma_mwd,
                       t_min_age, t_max_age, r1_min, r1_max, logger,
                       alpha_min=0.01, alpha_max=1.0):
    """
    Generator yielding (row, alpha, period_s, m1_init) for every SSE grid entry
    that satisfies:
      - WD core mass within 3 sigma of mwd
      - progenitor radius at CE onset in [r1_min, r1_max]
      - total system age in [t_min_age, t_max_age]
      - sigma_CE in [alpha_min, alpha_max] (enforced by EnergyInversion)
    """
    for m1_init in sse_df["M1_init"].unique():
        sub = sse_df[sse_df["M1_init"] == m1_init]
        row = sub.iloc[(sub["M1c"] - mwd).abs().argmin()]
        if abs(row["M1c"] - mwd) > 3 * sigma_mwd:
            continue
        if not (r1_min <= row["rad_1"] <= r1_max):
            continue
        total_age = row["tphys"] + t_cool
        if not (t_min_age <= total_age <= t_max_age):
            continue
        lam_calc = Lambda(
            logger, row["kstar_1"], row["lum_1"],
            row["mass_1"], row["rad_1"],
            row["mass_1"] - row["M1c"], row["rad_floor"],
        )
        try:
            lam = lam_calc.compute_lambda()
            energy_inv = EnergyInversion(
                logger,
                row["mass_1"], row["mass_1"] - row["M1c"], row["rad_1"],
                row["M1c"], m_bd, a_ce_rsun, lam,
                row["rad_floor"], row["rad_ceil"],
                alpha_min, alpha_max,
            )
            _, alpha, period_s = energy_inv.solve_for_alpha()
            yield row, alpha, period_s, m1_init
        except (ValueError, Exception):
            continue


def log_prior(theta, logger, sse_df, porb, t_min_age, t_max_age, sigma_mwd,
              mwd_obs, mbd_obs, tcool_obs, sigma_mbd, sigma_tcool,
              r1_min, r1_max, alpha_min, alpha_max) -> float:
    mwd, m2, t_cool = theta
    if mwd <= 0 or m2 <= 0 or t_cool <= 0:
        return -np.inf
    lp_gauss = -0.5 * (
        ((mwd - mwd_obs) / sigma_mwd) ** 2 +
        ((m2 - mbd_obs) / sigma_mbd) ** 2 +
        ((t_cool - tcool_obs) / sigma_tcool) ** 2
    )
    pce = period_after_CE(t_cool, mwd, m2, porb)
    a_ce = period_to_separation(pce, mwd, m2)
    for _ in _iter_sse_matches(mwd, m2, a_ce, t_cool, sse_df, sigma_mwd,
                                t_min_age, t_max_age, r1_min, r1_max, logger,
                                alpha_min, alpha_max):
        return lp_gauss
    return -np.inf


def log_likelihood(theta, p_obs, a_obs, sigma_a_obs, r_wd_obs, sigma_rwd):
    mwd, m_bd, _ = theta
    a_kepler = period_to_separation(p_obs, mwd, m_bd)
    Mch = 1.44
    r_wd_pred = 0.0114 * np.sqrt((Mch / mwd) ** (2 / 3) - (mwd / Mch) ** (2 / 3))
    return -0.5 * (
        ((a_kepler - a_obs) / sigma_a_obs) ** 2 +
        ((r_wd_pred - r_wd_obs) / sigma_rwd) ** 2
    )


def log_prob(theta, logger, sse_df, porb, t_min_age, t_max_age,
             mwd_obs, sigma_mwd, mbd_obs, sigma_mbd, tcool_obs, sigma_tcool,
             r1_min, r1_max, alpha_min, alpha_max,
             p_obs, a_obs, sigma_a_obs, r_wd_obs, sigma_rwd) -> float:
    lp = log_prior(theta, logger, sse_df, porb, t_min_age, t_max_age,
                   sigma_mwd, mwd_obs, mbd_obs, tcool_obs, sigma_mbd, sigma_tcool,
                   r1_min, r1_max, alpha_min, alpha_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, p_obs, a_obs, sigma_a_obs, r_wd_obs, sigma_rwd)


def plot_walkers(chain, labels, path):
    ndim = chain.shape[2]
    fig, axes = plt.subplots(ndim, figsize=(10, 4 * ndim), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(chain[:, :, i], alpha=0.3, lw=0.5, color="k")
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("Step")
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def plot_corner(flat_chain, labels, path):
    fig = corner.corner(
        flat_chain,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.savefig(path, dpi=600)
    plt.close(fig)


def derive_posterior_quantities(flat_chain, sse_df, p_obs, t_min_age, t_max_age,
                                 r1_min, r1_max, logger, sigma_mwd, thin=10,
                                 alpha_min=0.01, alpha_max=1.0):
    records = []
    for mwd, m_bd, t_cool in tqdm.tqdm(flat_chain[::thin],
                                         desc="Deriving posterior quantities",
                                         total=len(flat_chain[::thin])):
        p_ce = period_after_CE(t_cool, mwd, m_bd, p_obs)
        a_ce = period_to_separation(p_ce, mwd, m_bd)
        for row, alpha, period_s, m1_init in _iter_sse_matches(
                mwd, m_bd, a_ce, t_cool, sse_df, sigma_mwd,
                t_min_age, t_max_age, r1_min, r1_max, logger,
                alpha_min, alpha_max):
            if period_s / 86400.0 < 1.0:
                continue
            records.append({
                "mwd": mwd, "m_bd": m_bd, "t_cool": t_cool,
                "p_ce_days": p_ce,
                "p_init_days": period_s / 86400.0,
                "M1_init": m1_init,
                "rad1_ce_rsun": row["rad_1"],
                "alpha": alpha,
                "tphys_myr": row["tphys"],
                "total_age_myr": row["tphys"] + t_cool,
            })
    return pd.DataFrame(records)


def plot_best_fit_evolution(flat_chain, sse_df, p_obs, t_max_plot, t_min_age, t_max_age,
                             r1_min, r1_max, logger, path, sigma_mwd,
                             alpha_min=0.01, alpha_max=1.0):
    mwd_med, mbd_med, tcool_med = np.median(flat_chain, axis=0)
    logger.info(f"Posterior medians: M_wd={mwd_med:.3f} Msun, M_bd={mbd_med:.4f} Msun, t_cool={tcool_med:.1f} Myr")
    pce = period_after_CE(tcool_med, mwd_med, mbd_med, p_obs)
    a_ce = period_to_separation(pce, mwd_med, mbd_med)
    best_row, best_alpha, best_period, best_m1 = None, None, None, None
    best_dm1c = np.inf
    for row, alpha, period_s, m1_init in _iter_sse_matches(
            mwd_med, mbd_med, a_ce, tcool_med, sse_df, sigma_mwd,
            t_min_age, t_max_age, r1_min, r1_max, logger,
            alpha_min, alpha_max):
        dm1c = abs(row["M1c"] - mwd_med)
        if dm1c < best_dm1c:
            best_dm1c = dm1c
            best_row, best_alpha, best_period, best_m1 = row, alpha, period_s, m1_init
    if best_row is None:
        logger.warning("No valid best-fit solution found; skipping evolution plot")
        return
    p_init_days = best_period / 86400.0
    t_total = best_row["tphys"] + tcool_med
    logger.info(
        f"Best-fit: M1_init={best_m1:.3f} Msun, M1c={best_row['M1c']:.3f} Msun, "
        f"alpha={best_alpha:.3f}, P_init={p_init_days:.4f} d, total_age={t_total:.1f} Myr"
    )
    single_binary = InitialBinaryTable.InitialBinaries(
        m1=float(best_m1), m2=float(mbd_med), porb=p_init_days, ecc=0.0,
        tphysf=14000.0, kstar1=1, kstar2=0, metallicity=0.014,
    )
    single_binary["dtp"] = 1.0

    bpp, _, _, _ = Evolve.evolve(
        initialbinarytable=single_binary, BSEDict=bse_defaults(), SSEDict=sse_defaults()
    )
    evol_times = convert_kstar_evol_type(
        bpp[["tphys", "kstar_1", "kstar_2", "mass_1", "mass_2", "sep", "evol_type"]].copy()
    )
    logger.info("Best-fit evolution stages:")
    logger.info(f"  {'t (Myr)':>10}  {'evol_type':<30}  {'kstar_1':<38}  {'kstar_2':<30}  {'M1':>6}  {'M2':>6}  {'sep (Rsun)':>10}")
    for _, ev in evol_times.iterrows():
        logger.info(
            f"  {ev['tphys']:>10.2f}  {ev['evol_type']:<30}  {ev['kstar_1']:<38}  {ev['kstar_2']:<30}  "
            f"{ev['mass_1']:>6.3f}  {ev['mass_2']:>6.4f}  {ev['sep']:>10.3f}"
        )

    figs = evolve_and_plot(
        single_binary, t_min=None, t_max=t_max_plot,
        BSEDict=bse_defaults(), SSEDict=sse_defaults(), sys_obs={},
    )
    figs[0].savefig(path, dpi=600)
    plt.close(figs[0])
    logger.info(f"Evolution panel saved to {path}")


def main() -> None:
    args = arg_parse()
    config = load_config(args.config)
    logger = CE_logging(config["stage_name"], config["log_file"]).setup_logger()
    logger.info("Starting CE reconstruction with configuration:")
    for key, val in config.items():
        logger.info(f"  {key} = {val}")

    m1_grid = np.arange(0.08, 2.25, 0.01)
    kstar_giants = tuple(int(k) for k in config.get("kstar_giants", "3,5,6").split(","))

    p_obs        = float(config["p_obs"])
    m_wd         = float(config["m_wd"])
    sigma_mwd    = float(config["sigma_mwd"])
    m_bd         = float(config["m_bd"])
    sigma_mbd    = float(config["sigma_mbd"])
    t_cool       = float(config["t_cool"])
    sigma_tcool  = float(config["sigma_tcool"])
    a_obs        = float(config["a_obs"])
    sigma_a_obs  = float(config["sigma_a_obs"])
    r_wd_obs     = float(config["r_wd"])
    sigma_rwd    = float(config["sigma_rwd"])

    t_max_plot   = float(config.get("t_max_plot", 14000.0))
    t_max_age    = float(config.get("t_max_age",  10000.0))
    t_min_age    = float(config.get("t_min_age",  0.0))
    r1_ce_min    = float(config.get("r1_ce_min",  0.0))
    r1_ce_max    = float(config.get("r1_ce_max",  1e6))
    alpha_min    = float(config.get("alpha_min",  0.01))
    alpha_max    = float(config.get("alpha_max",  1.0))

    p_ce = period_after_CE(t_cool, m_wd, m_bd, p_obs)
    a_f  = period_to_separation(p_ce, m_wd, m_bd)
    logger.info(
        f"Constraints: P_obs={p_obs} d, M_wd={m_wd} Msun, M_bd={m_bd} Msun, "
        f"a_f={a_f:.4f} Rsun, P_CE={p_ce:.6f} d, "
        f"t_min={t_min_age} Myr, t_max={t_max_age} Myr, kstars={kstar_giants}, "
        f"alpha=[{alpha_min},{alpha_max}]"
    )

    ce_runner = CEGridRunner(logger, m1_grid, m_wd, m_bd, a_f, t_max_age, kstar_giants,
                             alpha_min, alpha_max)
    results_df = ce_runner.run()
    logger.info(f"CE grid runner completed with {len(results_df)} valid solutions")
    results_df.to_csv(config["output_csv"], index=False)
    sse_df = ce_runner.sse_df

    logger.info("Starting MCMC sampling")
    n_burnin = int(config.get("n_burnin", 500))
    n_steps  = int(config.get("n_steps",  2000))
    nwalkers, ndim = 32, 3
    labels = [r"$M_{\text{wd}} \, [M_\odot]$", r"$M_{\text{bd}} \, [M_\odot]$", r"$t_{\text{cool}}$"]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(logger, sse_df, p_obs, t_min_age, t_max_age,
              m_wd, sigma_mwd, m_bd, sigma_mbd, t_cool, sigma_tcool,
              r1_ce_min, r1_ce_max, alpha_min, alpha_max,
              p_obs, a_obs, sigma_a_obs, r_wd_obs, sigma_rwd),
    )
    initial_pos = np.column_stack([
        np.random.normal(m_wd,    sigma_mwd   * 0.5, nwalkers),
        np.random.normal(m_bd,    sigma_mbd   * 0.5, nwalkers),
        np.random.normal(t_cool,  sigma_tcool * 0.5, nwalkers),
    ])

    logger.info(f"Running burn-in ({n_burnin} steps)")
    burnin_state = sampler.run_mcmc(initial_pos, n_burnin, progress=True)
    sampler.reset()
    logger.info(f"Running production ({n_steps} steps)")
    sampler.run_mcmc(burnin_state, n_steps, progress=True)

    flat_chain = sampler.get_chain(flat=True)
    chain      = sampler.get_chain()
    np.savetxt(config["chain_csv"], flat_chain, delimiter=",",
               header="mwd,m_bd,t_cool", comments="")
    logger.info(f"Chain saved to {config['chain_csv']} ({len(flat_chain)} samples)")

    plot_walkers(chain, labels, config["walkers_plot"])
    plot_corner(flat_chain, labels, config["corner_plot"])
    plot_best_fit_evolution(
        flat_chain, sse_df, p_obs, t_max_plot,
        t_min_age, t_max_age, r1_ce_min, r1_ce_max,
        logger, config["evolution_plot"], sigma_mwd,
        alpha_min, alpha_max,
    )

    logger.info("MCMC posterior summary (median  +upper / -lower  at 1 sigma):")
    chain_params = [
        (0, "M_wd",   "Msun", ".4f"),
        (1, "M_bd",   "Msun", ".5f"),
        (2, "t_cool", "Myr",  ".2f"),
    ]
    for idx, name, unit, fmt in chain_params:
        q16, q50, q84 = np.percentile(flat_chain[:, idx], [16, 50, 84])
        logger.info(f"  {name:8s}: {q50:{fmt}}  +{q84-q50:{fmt}} / -{q50-q16:{fmt}}  {unit}")

    derived = derive_posterior_quantities(
        flat_chain, sse_df, p_obs, t_min_age, t_max_age,
        r1_ce_min, r1_ce_max, logger, sigma_mwd,
        alpha_min=alpha_min, alpha_max=alpha_max,
    )
    derived.to_csv(config["derived_csv"], index=False)

    if len(derived):
        logger.info("Derived posterior summary (median  +upper / -lower  at 1 sigma):")
        derived_params = [
            ("p_ce_days",     "d",    ".6f"),
            ("p_init_days",   "d",    ".6f"),
            ("M1_init",       "Msun", ".6f"),
            ("rad1_ce_rsun",  "Rsun", ".6f"),
            ("alpha",         "",     ".6f"),
            ("tphys_myr",     "Myr",  ".6f"),
            ("total_age_myr", "Myr",  ".6f"),
        ]
        for col, unit, fmt in derived_params:
            if col in derived.columns:
                q16, q50, q84 = np.percentile(derived[col], [16, 50, 84])
                logger.info(f"  {col:20s}: {q50:{fmt}}  +{q84-q50:{fmt}} / -{q50-q16:{fmt}}  {unit}")

    logger.info("CE reconstruction completed successfully")


if __name__ == "__main__":
    main()
