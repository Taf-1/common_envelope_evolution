from sse_grid import sseGrid
import logging
import pandas as pd
from lambda_claeys2014 import Lambda
from ce_energy_inversion import EnergyInversion
import numpy as np
from astropy.constants import G, M_sun, R_sun

class CEGridRunner:
    def __init__(self, logger: logging.Logger, M1_grid: list, M_wd_obs: float,
                 M2: float, a_f: float, t_max: float = 14000):
        self.logger = logger
        self.M1_grid = M1_grid
        self.M_wd_obs = M_wd_obs
        self.M2 = M2
        self.a_f = a_f
        self.t_max = t_max

    @staticmethod
    def separation_to_period(separation: float, m1: float, m2: float) -> float:
        return 2 * np.pi * np.sqrt((separation* R_sun.value)**3 / (G.value * (m1 + m2) * M_sun.value))

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting CE grid runner")
        sse = sseGrid(self.logger, self.M1_grid, self.M_wd_obs, self.M2)
        self.sse_df = sse.compute_sse_grid()
        results = []
        for _, row in self.sse_df.iterrows():
            q_obs = row["mass_1"] / self.M2
            q_crit = 0.362 + 1.0 / (3.0 * (1.0 - row["M1c"] / row["mass_1"]))
            if q_obs < q_crit:
                self.logger.debug(
                    f"M1={row['M1_init']:.2f} kstar={int(row['kstar_1'])}: "
                    f"q={q_obs:.2f} < q_crit={q_crit:.2f}, stable MT — skipped"
                )
                continue
            lam_calc = Lambda(
                self.logger, row["kstar_1"], row["lum_1"],
                row["mass_1"], row["rad_1"],
                row["mass_1"] - row["M1c"], row["rad_floor"],
            )
            lam = lam_calc.compute_lambda()
            try:
                if row["tphys"] > self.t_max:
                    self.logger.debug(f"M1={row['M1_init']:.2f} t={row['tphys']:.3e} yr: rejected — tphys exceeds t_max")
                    continue
                energy_inv = EnergyInversion(
                    self.logger,
                    row["mass_1"], row["mass_1"] - row["M1c"], row["rad_1"],
                    row["M1c"], self.M2, self.a_f, lam,
                    row["rad_floor"], row["rad_ceil"],
                )
                p_final = self.separation_to_period(self.a_f, row["M1c"], self.M2)
                a_i, alpha, period = energy_inv.solve_for_alpha()
                if period < p_final or period < p_final + 0.1 * p_final:
                    self.logger.debug(f"Initial period: {period:.3e} s: rejected — initial period too short to produce binary system we observe today.")
                    continue
                results.append({
                    "M1_init": row["M1_init"],
                    "alpha": alpha,
                    "tphys_CE": row["tphys"],
                    "kstar_CE": row["kstar_1"],
                    "M1_CE": row["mass_1"],
                    "M1c_CE": row["M1c"],
                    "lam": lam,
                    "a_i_m": a_i,
                    "P_orb_init_s": period,
                    "rad_floor_rsun": row["rad_floor"],
                    "rad_ceil_rsun": row["rad_ceil"],
                })
                self.logger.debug(
                    f"M1={row['M1_init']:.2f} alpha={alpha:.2f}: a_i={a_i:.3e} m  P_orb_init={period:.3e} s"
                )
            except ValueError as e:
                self.logger.debug(
                    f"M1={row['M1_init']:.2f} t={row['tphys']:.3e} Myr: rejected — {e}"
                )
        self.logger.info(f"Grid runner complete: {len(results)} surviving solutions")
        return pd.DataFrame(results)
