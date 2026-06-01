import numpy as np
from astropy.constants import G, M_sun, R_sun


def bse_defaults() -> dict:
    return {
        "neta": 0.5, "bwind": 0.0, "hewind": 0.5, "windflag": 3, "LBV_flag": 1,
        "alpha1": [1.0, 1.0], "lambdaf": 0.0, "ceflag": 1, "cekickflag": 2,
        "cemergeflag": 1, "cehestarflag": 0, "qcflag": 5, "qcrit_array": [0.0] * 16,
        "beta": -1.0, "xi": 0.5, "acc2": 1.5, "eddfac": 10, "eddlimflag": 0,
        "epsnov": 0.001, "gamma": -2.0, "don_lim": -1, "acc_lim": [-1, -1],
        "smt_periastron_check": 0, "fprimc_array": [2.0 / 21.0] * 16, "tflag": 1, "ST_tide": 1,
        "pts1": 0.001, "pts2": 0.01, "pts3": 0.02, "zsun": 0.014,
        "wdflag": 1, "ifflag": 1, "wd_mass_lim": 1, "kickflag": 5, "sigma": 265.0,
        "bhflag": 1, "bhsigmafrac": 1.0, "sigmadiv": -20.0, "ecsn": 2.25, "ecsn_mlow": 1.6,
        "aic": 1, "ussn": 1, "polar_kick_angle": 90.0,
        "natal_kick_array": [[-100., -100., -100., -100., 0.], [-100., -100., -100., -100., 0.]],
        "mm_mu_ns": 400.0, "mm_mu_bh": 200.0, "remnantflag": 4, "fryer_mass_limit": 0,
        "mxns": 3.0, "fryer_fmix": 1.0, "fryer_mcrit_nsbh": 5.75, "rembar_massloss": 0.5,
        "bhms_coll_flag": 0, "bhms_accretion_factor": 1.0, "pisn": -2,
        "ppi_co_shift": 0.0, "ppi_extra_ml": 0.0, "rtmsflag": 0, "rejuv_fac": 1.0,
        "rejuvflag": 0, "maltsev_mode": 0, "maltsev_fallback": 0.5, "maltsev_pf_prob": 0.1,
        "bconst": 3000, "ck": 1000, "bdecayfac": 1, "bhspinflag": 0, "bhspinmag": 0.0,
        "grflag": 1, "htpmb": 1, "ST_cr": 1,
    }


def sse_defaults() -> dict:
    return {"stellar_engine": "sse"}


def period_to_separation(period_days: float, m1_msun: float, m2_msun: float) -> float:
    """Kepler III: orbital period (days) → separation (Rsun)."""
    tau = period_days * 86400.0 / (2 * np.pi)
    a_m = (G.value * (m1_msun + m2_msun) * M_sun.value * tau ** 2) ** (1.0 / 3.0)
    return a_m / R_sun.value


def separation_to_period(separation_rsun: float, m1_msun: float, m2_msun: float) -> float:
    """Kepler III: separation (Rsun) → orbital period (seconds)."""
    return 2 * np.pi * np.sqrt(
        (separation_rsun * R_sun.value) ** 3
        / (G.value * (m1_msun + m2_msun) * M_sun.value)
    )
