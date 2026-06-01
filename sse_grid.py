from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import logging
import numpy as np
import pandas as pd
from utils import bse_defaults, sse_defaults

# COSMIC kstar legend (Hurley+2000 / COSMIC docs):
#   0  = BD/planet     1  = MS (M≥0.7)   2  = HG
#   3  = GB (FGB/RGB)  4  = CHeB         5  = EAGB
#   6  = TPAGB         10 = He WD        11 = CO WD
# FGB progenitors (kstar 3) → He-core WDs;  AGB (5,6) → CO/ONe WDs.
# Default includes both so the energy balance determines validity per system.
_DEFAULT_KSTAR_GIANTS = (3, 5, 6)


class sseGrid:
    def __init__(self, logger: logging.Logger, M1_grid: list, M_wd_obs: float,
                 M2: float, kstar_giants: tuple = _DEFAULT_KSTAR_GIANTS):
        self.M1_grid = M1_grid
        self.logger = logger
        self.M2 = M2
        self.M_wd_obs = M_wd_obs
        self.kstar_giants = tuple(kstar_giants)

    def _evolve_single(self, M1) -> pd.DataFrame | None:
        self.logger.debug(f"Evolving M1={M1:.2f} Msun as a single star (wide binary) to build SSE grid")
        binary = InitialBinaryTable.InitialBinaries(
            m1=float(M1), m2=self.M2, porb=1e8, ecc=0.0, tphysf=14000.0,
            kstar1=1, kstar2=0, metallicity=0.014,
        )
        binary["dtp"] = 1.0
        self.logger.debug(f"Starting evolution for M1={M1:.2f} Msun")
        _, bcm, _, _ = Evolve.evolve(
            initialbinarytable=binary, BSEDict=bse_defaults(), SSEDict=sse_defaults()
        )
        self.logger.debug(f"Finished evolution for M1={M1:.2f} Msun, processing giant phases")

        ms_rows = bcm[bcm["kstar_1"] == 1]
        if len(ms_rows) == 0:
            self.logger.warning(f"M1={M1}: no main sequence rows in BCM, cannot determine TAMS radius")
            return None
        tams_radius = ms_rows["rad_1"].iloc[-1]

        giants = bcm[bcm["kstar_1"].isin([2, 3, 4, 5, 6])].copy()
        if len(giants) == 0:
            self.logger.warning(f"M1={M1}: no giant phase reached")
            return None

        # He-core mass for HG/FGB (kstar 2,3); CO-core mass for CHeB/AGB (kstar 4,5,6)
        giants["M1c"] = np.where(
            giants["kstar_1"].isin([2, 3]),
            giants["massc_he_layer_1"],
            giants["massc_co_layer_1"],
        )

        cols = ["tphys", "mass_1", "rad_1", "M1c", "kstar_1", "lum_1", "sep"]
        out = giants[cols].copy()
        out = out[np.isfinite(out["M1c"]) & (out["M1c"] > 0)]
        out["M1_init"] = M1
        out["rad_floor"] = tams_radius
        out["rad_ceil"] = out["rad_1"]

        out = out[out["kstar_1"].isin(self.kstar_giants)]
        return out if len(out) > 0 else None

    def compute_sse_grid(self) -> pd.DataFrame:
        frames = []
        self.logger.debug(f"Computing SSE grid for M1 values: {self.M1_grid}")
        for m1 in self.M1_grid:
            result = self._evolve_single(M1=m1)
            if result is not None:
                frames.append(result)
        grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.logger.debug(f"Grid ready: {len(frames)}/{len(self.M1_grid)} M1 values produced giant phases")
        return grid
