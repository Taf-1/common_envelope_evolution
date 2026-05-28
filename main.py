import argparse
import configparser
from logger import CE_logging
from grid_runner import CEGridRunner
import numpy as np
import emcee
from astropy.constants import G, M_sun, R_sun, c
from lambda_claeys2014 import Lambda
from ce_energy_inversion import EnergyInversion
import corner
import matplotlib.pyplot as plt
from cosmic.plotting import evolve_and_plot
from cosmic.sample.initialbinarytable import InitialBinaryTable
import pandas as pd

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

def bse_defaults() -> tuple[dict, dict]:
    bse = {
        "neta":0.5,"bwind":0.0,"hewind":0.5,"windflag":3,"LBV_flag":1,
        "alpha1":[1.0,1.0],"lambdaf":0.0,"ceflag":1,"cekickflag":2,
        "cemergeflag":1,"cehestarflag":0,"qcflag":5,"qcrit_array":[0.0]*16,
        "beta":-1.0,"xi":0.5,"acc2":1.5,"eddfac":10,"eddlimflag":0,
        "epsnov":0.001,"gamma":-2.0,"don_lim":-1,"acc_lim":[-1,-1],
        "smt_periastron_check":0,"fprimc_array":[2.0/21.0]*16,"tflag":1,"ST_tide":1,
        "pts1":0.001,"pts2":0.01,"pts3":0.02,"zsun":0.014,
        "wdflag":1,"ifflag":1,"wd_mass_lim":1,"kickflag":5,"sigma":265.0,
        "bhflag":1,"bhsigmafrac":1.0,"sigmadiv":-20.0,"ecsn":2.25,"ecsn_mlow":1.6,
        "aic":1,"ussn":1,"polar_kick_angle":90.0,
        "natal_kick_array":[[-100.,-100.,-100.,-100.,0.],[-100.,-100.,-100.,-100.,0.]],
        "mm_mu_ns":400.0,"mm_mu_bh":200.0,"remnantflag":4,"fryer_mass_limit":0,
        "mxns":3.0,"fryer_fmix":1.0,"fryer_mcrit_nsbh":5.75,"rembar_massloss":0.5,
        "bhms_coll_flag":0,"bhms_accretion_factor":1.0,"pisn":-2,
        "ppi_co_shift":0.0,"ppi_extra_ml":0.0,"rtmsflag":0,"rejuv_fac":1.0,
        "rejuvflag":0,"maltsev_mode":0,"maltsev_fallback":0.5,"maltsev_pf_prob":0.1,
        "bconst":3000,"ck":1000,"bdecayfac":1,"bhspinflag":0,"bhspinmag":0.0,
        "grflag":1,"htpmb":1,"ST_cr":1,
    }
    sse = {"stellar_engine": "sse"}
    return bse, sse

def period_to_separation(period: float, m1: float, m2: float) -> float:
    return ((G.value * (m1 + m2) * M_sun.value * (period*86400/(2*np.pi))**2)**(1/3)) / R_sun.value

def period_after_CE(t_cool, mwd, mbd, porb):
    """
    Compute post-common-envelope orbital period using Schreiber & Gänsicke (2003) Eq. 8.
    """
    tcool_sec = t_cool * 1e6 * 3.15576e7
    M1 = mwd * M_sun.value
    M2 = mbd * M_sun.value
    Mtot = M1 + M2
    porb_83 = (porb * 86400.0)**(8/3)
    factor = (
        (256 / (5 * c.value**5))
        * G.value**(5/3)
        * (2 * np.pi)**(8/3)
        * M1 * M2
        * Mtot**(-1/3)
    )
    pce_83 = porb_83 + factor * tcool_sec
    return pce_83**(3/8) / 86400.0

def log_prior(theta, logger, sse_df, r_bd, porb, t_max) -> float:
    mwd, m2, t_cool = theta
    if mwd <= 0 or m2 <= 0 or t_cool <= 0:
        return -np.inf
    pce = period_after_CE(t_cool, mwd, m2, porb)
    a_ce = period_to_separation(pce, mwd, m2)
    for m1_init in sse_df["M1_init"].unique():
        sub = sse_df[sse_df["M1_init"] == m1_init]
        row = sub.iloc[(sub["M1c"] - mwd).abs().argmin()]
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
                row["M1c"], m2, a_ce, lam,
                row["rad_floor"], row["rad_ceil"],
            )
            _, alpha, _ = energy_inv.solve_for_alpha()
            if not (0.1 <= alpha <= 1.0): continue
            if row["tphys"] + t_cool > t_max: continue
            return 0.0
        except (ValueError, Exception):
            continue
    return -np.inf


def log_likelihood(theta, mwd_obs, sigma_mwd, mbd_obs, sigma_mbd, tcool_obs, sigma_tcool) -> float:
    mwd, m_bd, t_cool = theta
    return -0.5 * (
        ((mwd - mwd_obs) / sigma_mwd) ** 2 +
        ((m_bd - mbd_obs) / sigma_mbd) ** 2 +
        ((t_cool - tcool_obs) / sigma_tcool) ** 2
    )

def log_prob(theta, logger, sse_df, r_bd, porb, t_max,
             mwd_obs, sigma_mwd, mbd_obs, sigma_mbd, tcool_obs, sigma_tcool) -> float:
    lp = log_prior(theta, logger, sse_df, r_bd, porb, t_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, mwd_obs, sigma_mwd, mbd_obs, sigma_mbd, tcool_obs, sigma_tcool)

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

def derive_posterior_quantities(flat_chain, sse_df, r_bd, p_obs, t_max_age, logger, thin=10):
    records = []
    for mwd, m_bd, t_cool in flat_chain[::thin]:
        pce = period_after_CE(t_cool, mwd, m_bd, p_obs)
        a_ce = period_to_separation(pce, mwd, m_bd)
        for m1_init in sse_df["M1_init"].unique():
            sub = sse_df[sse_df["M1_init"] == m1_init]
            row = sub.iloc[(sub["M1c"] - mwd).abs().argmin()]
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
                    row["M1c"], m_bd, a_ce, lam,
                    row["rad_floor"], row["rad_ceil"],
                )
                _, alpha, period = energy_inv.solve_for_alpha()
                if not (0.09 <= alpha <= 1.0):
                    continue
                if row["tphys"] + t_cool > t_max_age:
                    continue

                records.append({
                    "mwd": mwd, "m_bd": m_bd, "t_cool": t_cool,
                    "p_ce_days": pce,
                    "p_init_days": period / 86400.0,
                    "M1_init": m1_init,
                    "rad1_ce_rsun": row["rad_1"],
                    "alpha": alpha,
                    "tphys_myr": row["tphys"],
                    "total_age_myr": row["tphys"] + t_cool,
                })
            except (ValueError, Exception):
                continue
    return pd.DataFrame(records)

def plot_best_fit_evolution(flat_chain, sse_df, r_bd, p_obs, t_max_plot, t_max_age, logger, path):
    mwd_med, mbd_med, tcool_med = np.median(flat_chain, axis=0)
    logger.info(f"Posterior medians: M_wd={mwd_med:.3f} Msun, M_bd={mbd_med:.4f} Msun, t_cool={tcool_med:.1f} Myr")
    pce = period_after_CE(tcool_med, mwd_med, mbd_med, p_obs)
    a_ce = period_to_separation(pce, mwd_med, mbd_med)
    best_row, best_alpha, best_period, best_m1 = None, None, None, None
    best_dm1c = np.inf
    for m1_init in sse_df["M1_init"].unique():
        sub = sse_df[sse_df["M1_init"] == m1_init]
        row = sub.iloc[(sub["M1c"] - mwd_med).abs().argmin()]
        dm1c = abs(row["M1c"] - mwd_med)
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
                row["M1c"], mbd_med, a_ce, lam,
                row["rad_floor"], row["rad_ceil"],
            )
            _, alpha, period = energy_inv.solve_for_alpha()
            if not (0.1 <= alpha <= 1.0):
                continue
            if row["tphys"] + tcool_med > t_max_age:
                continue
            if dm1c < best_dm1c:
                best_dm1c = dm1c
                best_row, best_alpha, best_period, best_m1 = row, alpha, period, m1_init
        except (ValueError, Exception):
            continue
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
    BSEDict, SSEDict = bse_defaults()
    figs = evolve_and_plot(
        single_binary, t_min=None, t_max=t_max_plot,
        BSEDict=BSEDict, SSEDict=SSEDict, sys_obs={},
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
    m1_grid = np.arange(0.08, 8.0, 0.005)
    logger.info(f"Parameter grid for M1: {m1_grid}")
    p_obs, m_wd, sigma_mwd, m_bd, sigma_mbd, t_cool, sigma_tcool, r_bd = (
        float(config["p_obs"]), float(config["m_wd"]),
        float(config["sigma_mwd"]), float(config["m_bd"]),
        float(config["sigma_mbd"]), float(config["t_cool"]), 
        float(config["sigma_tcool"]), float(config["r_bd"]),
    )
    t_max_plot = float(config.get("t_max_plot", 14000.0))
    t_max_age  = float(config.get("t_max_age",  10000.0))
    p_ce = period_after_CE(t_cool, m_wd, m_bd, p_obs)
    a_f = period_to_separation(p_ce, m_wd, m_bd)
    logger.info(f"Observational constraints: P_obs={p_obs} days, M_wd={m_wd} Msun, M_bd={m_bd} Msun, a_f={a_f:.4f} Rsun")
    ce_runner = CEGridRunner(logger, m1_grid, m_wd, m_bd, a_f, t_max_age)
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
        args=(logger, sse_df, r_bd, p_obs, t_max_age,
            m_wd, sigma_mwd, m_bd, sigma_mbd, t_cool, sigma_tcool),
    )
    initial_pos = np.column_stack([
        np.random.normal(m_wd, sigma_mwd * 0.5, nwalkers),
        np.random.normal(m_bd, sigma_mbd * 0.5, nwalkers),
        np.random.normal(t_cool, sigma_tcool * 0.5, nwalkers),
    ])
    logger.info(f"Running burn-in ({n_burnin} steps)")
    burnin_state = sampler.run_mcmc(initial_pos, n_burnin, progress=True)
    sampler.reset()
    logger.info(f"Running production ({n_steps} steps)")
    sampler.run_mcmc(burnin_state, n_steps, progress=True)
    flat_chain = sampler.get_chain(flat=True)
    chain = sampler.get_chain()
    np.savetxt(config["chain_csv"], flat_chain, delimiter=",", header="mwd,m_bd,t_cool", comments="")
    logger.info(f"Chain saved to {config['chain_csv']} ({len(flat_chain)} samples)")
    plot_walkers(chain, labels, config["walkers_plot"])
    logger.info(f"Walker trace saved to {config['walkers_plot']}")
    plot_corner(flat_chain, labels, config["corner_plot"])
    logger.info(f"Corner plot saved to {config['corner_plot']}")
    plot_best_fit_evolution(flat_chain, sse_df, r_bd, p_obs, t_max_plot, t_max_age, logger, config["evolution_plot"])
    logger.info(f"Best-fit evolution plot saved to {config['evolution_plot']}")
    logger.info(
        f"M_wd range: [{flat_chain[:, 0].min():.3f}, {flat_chain[:, 0].max():.3f}] Msun  "
        f"M_bd range: [{flat_chain[:, 1].min():.4f}, {flat_chain[:, 1].max():.4f}] Msun  "
        f"t_cool range: [{flat_chain[:, 2].min():.1f}, {flat_chain[:, 2].max():.1f}] Myr"
    )
    derived = derive_posterior_quantities(flat_chain, sse_df, r_bd, p_obs, t_max_age, logger)
    derived.to_csv(config["derived_csv"], index=False)
    for col in ["p_ce_days", "p_init_days", "M1_init", "rad1_ce_rsun", "alpha", "tphys_myr", "total_age_myr"]:
        logger.info(f"{col}: [{derived[col].min():.4g}, {derived[col].max():.4g}]")
    logger.info("CE reconstruction completed successfully")

if __name__ == "__main__":
    main()
