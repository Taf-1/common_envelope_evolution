from sse_grid import sseGrid
import logging
import pandas as pd
from lambda_claeys2014 import Lambda
from ce_energy_inversion import EnergyInversion

class CEGridRunner:
    def __init__(self, logger: logging.Logger, M1_grid: list, M_wd_obs: float,
                 M2: float, a_f: float, alpha_grid: list):
        self.logger = logger
        self.M1_grid = M1_grid
        self.M_wd_obs = M_wd_obs
        self.M2 = M2
        self.a_f = a_f
        self.alpha_grid = alpha_grid

    def run(self) -> pd.DataFrame:
        self.logger.info("Starting CE grid runner")
        sse = sseGrid(self.logger, self.M1_grid, self.M_wd_obs, self.M2)
        self.sse_df = sse.compute_sse_grid()
        results = []
        for _, row in self.sse_df.iterrows():
            lam_calc = Lambda(
                self.logger, row["kstar_1"], row["lum_1"],
                row["mass_1"], row["rad_1"],
                row["mass_1"] - row["M1c"], row["rad_floor"],
            )
            lam = lam_calc.compute_lambda()
            for alpha in self.alpha_grid:
                try:
                    energy_inv = EnergyInversion(
                        self.logger,
                        row["mass_1"], row["mass_1"] - row["M1c"], row["rad_1"],
                        row["M1c"], self.M2, self.a_f, lam, alpha,
                        row["rad_floor"], row["rad_ceil"],
                    )
                    a_i, period = energy_inv.solve_for_ai()
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
                    self.logger.info(
                        f"M1={row['M1_init']:.2f} α={alpha:.2f}: a_i={a_i:.3e} m  P_orb_init={period:.3e} s"
                    )
                except ValueError as e:
                    self.logger.debug(
                        f"M1={row['M1_init']:.2f} α={alpha:.2f} t={row['tphys']:.3e} yr: rejected — {e}"
                    )
        self.logger.info(f"Grid runner complete: {len(results)} surviving solutions")
        return pd.DataFrame(results)
