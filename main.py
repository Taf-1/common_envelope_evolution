import argparse
import configparser
from logger import CE_logging
from grid_runner import CEGridRunner
import numpy as np
import emcee
from astropy.constants import G, M_sun, R_sun
from lambda_claeys2014 import Lambda
from ce_energy_inversion import EnergyInversion
import corner
import matplotlib.pyplot as plt
from cosmic.plotting import evolve_and_plot
from cosmic.sample.initialbinarytable import InitialBinaryTable

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

def log_prior(theta, m1_min, m1_max) -> float:
    m1, alpha = theta
    if m1_min <= m1 <= m1_max and 0.1 <= alpha <= 1.0:
        return 0.0
    return -np.inf

def log_likelihood(theta, logger, sse_df, m_wd, sigma_mwd, m_bd, a_f) -> float:
    m1, alpha = theta
    nearest_m1 = sse_df["M1_init"].iloc[(sse_df["M1_init"] - m1).abs().argmin()]
    subset = sse_df[sse_df["M1_init"] == nearest_m1]
    row = subset.iloc[(subset["M1c"] - m_wd).abs().argmin()]
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
            row["M1c"], m_bd, a_f, lam, alpha,
            row["rad_floor"], row["rad_ceil"],
        )
        _, _ = energy_inv.solve_for_ai()
    except (ValueError, Exception):
        return -np.inf
    return -0.5 * ((row["M1c"] - m_wd) / sigma_mwd) ** 2

def log_prob(theta, logger, sse_df, m_wd, sigma_mwd, m_bd, a_f, m1_min, m1_max) -> float:
    lp = log_prior(theta, m1_min, m1_max)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, logger, sse_df, m_wd, sigma_mwd, m_bd, a_f)
    return lp + ll if np.isfinite(ll) else -np.inf

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

def plot_corner(flat_chain, labels, m_wd, sigma_mwd, path):
    fig = corner.corner(
        flat_chain,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
    )
    fig.savefig(path, dpi=600)
    plt.close(fig)

def plot_best_fit_evolution(flat_chain, sse_df, m_wd, m_bd, a_f, logger, t_max, path):
    m1_med, alpha_med = np.median(flat_chain, axis=0)
    logger.info(f"Best-fit (median): M1={m1_med:.3f} Msun, alpha={alpha_med:.3f}")
    nearest_m1 = sse_df["M1_init"].iloc[(sse_df["M1_init"] - m1_med).abs().argmin()]
    subset = sse_df[sse_df["M1_init"] == nearest_m1]
    row = subset.iloc[(subset["M1c"] - m_wd).abs().argmin()]
    lam_calc = Lambda(
        logger, row["kstar_1"], row["lum_1"],
        row["mass_1"], row["rad_1"],
        row["mass_1"] - row["M1c"], row["rad_floor"],
    )
    lam = lam_calc.compute_lambda()
    energy_inv = EnergyInversion(
        logger,
        row["mass_1"], row["mass_1"] - row["M1c"], row["rad_1"],
        row["M1c"], m_bd, a_f, lam, alpha_med,
        row["rad_floor"], row["rad_ceil"],
    )
    _, p_init_s = energy_inv.solve_for_ai()
    p_init_days = p_init_s / 86400.0
    logger.info(f"Best-fit P_orb_init={p_init_days:.4f} days")
    single_binary = InitialBinaryTable.InitialBinaries(
        m1=float(m1_med), m2=float(m_bd), porb=p_init_days, ecc=0.0,
        tphysf=14000.0, kstar1=1, kstar2=0, metallicity=0.014,
    )
    single_binary["dtp"] = 1.0
    BSEDict, SSEDict = bse_defaults()
    fig = evolve_and_plot(
        single_binary, t_min=None, t_max=t_max,
        BSEDict=BSEDict, SSEDict=SSEDict, sys_obs={},
    )
    fig.savefig(path, dpi=600)
    plt.close(fig)

def main() -> None:
    args = arg_parse()
    config = load_config(args.config)
    logger = CE_logging(config["stage_name"], config["log_file"]).setup_logger()
    logger.info("Starting CE reconstruction with configuration:")
    for key, val in config.items():
        logger.info(f"  {key} = {val}")
    m1_grid = np.arange(1.0, 3.0, 0.05)
    logger.info(f"Parameter grid for M1: {m1_grid}")
    p_obs, m_wd, sigma_mwd, m_bd = (
        float(config["p_obs"]), float(config["m_wd"]),
        float(config["sigma_mwd"]), float(config["m_bd"]),
    )
    a_f_raw = config.get("a_f")
    a_f = float(a_f_raw) if a_f_raw is not None else period_to_separation(p_obs, m_wd, m_bd)
    logger.info(f"Observational constraints: P_obs={p_obs} days, M_wd={m_wd} Msun, M_bd={m_bd} Msun, a_f={a_f:.4f} Rsun")
    alpha_grid = np.linspace(0.1, 1.0, 10)
    ce_runner = CEGridRunner(logger, m1_grid, m_wd, m_bd, a_f, alpha_grid)
    results_df = ce_runner.run()
    logger.info(f"CE grid runner completed with {len(results_df)} valid solutions")
    results_df.to_csv(config["output_csv"], index=False)
    sse_df = ce_runner.sse_df
    logger.info("Starting MCMC sampling")
    n_burnin = int(config.get("n_burnin", 500))
    n_steps  = int(config.get("n_steps",  2000))
    nwalkers, ndim = 32, 2
    labels = [r"$M_1 \, [M_\odot]$", r"$\alpha_{\rm CE}$"]
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob,
        args=(logger, sse_df, m_wd, sigma_mwd, m_bd, a_f, min(m1_grid), max(m1_grid)),
    )
    if len(results_df) == 0:
        logger.warning("No grid survivors — falling back to random initialisation")
        initial_pos = np.array([[np.random.uniform(min(m1_grid), max(m1_grid)),
                                  np.random.uniform(0.1, 1.0)] for _ in range(nwalkers)])
    else:
        survivors = results_df[["M1_init", "alpha"]].values
        idx = np.random.choice(len(survivors), size=nwalkers, replace=True)
        initial_pos = survivors[idx] + np.random.randn(nwalkers, 2) * np.array([0.05, 0.05])
        initial_pos[:, 0] = np.clip(initial_pos[:, 0], min(m1_grid), max(m1_grid))
        initial_pos[:, 1] = np.clip(initial_pos[:, 1], 0.1, 1.0)
    logger.info(f"Running burn-in ({n_burnin} steps)")
    sampler.run_mcmc(initial_pos, n_burnin, progress=True)
    sampler.reset()
    logger.info(f"Running production ({n_steps} steps)")
    sampler.run_mcmc(sampler.get_last_sample(), n_steps, progress=True)
    flat_chain = sampler.get_chain(flat=True)
    chain = sampler.get_chain()
    np.savetxt(config["chain_csv"], flat_chain, delimiter=",", header="M1,alpha", comments="")
    logger.info(f"Chain saved to {config['chain_csv']} ({len(flat_chain)} samples)")
    plot_walkers(chain, labels, config["walkers_plot"])
    logger.info(f"Walker trace saved to {config['walkers_plot']}")
    plot_corner(flat_chain, labels, m_wd, sigma_mwd, config["corner_plot"])
    logger.info(f"Corner plot saved to {config['corner_plot']}")
    t_max = float(config.get("t_max_plot", 14000.0))
    plot_best_fit_evolution(flat_chain, sse_df, m_wd, m_bd, a_f, logger, t_max, config["evolution_plot"])
    logger.info(f"Best-fit evolution plot saved to {config['evolution_plot']}")

if __name__ == "__main__":
    main()
