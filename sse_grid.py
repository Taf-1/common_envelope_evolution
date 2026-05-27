from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import logging
import numpy as np
import pandas as pd

class sseGrid:
    def __init__(self, logger: logging.Logger, M1_grid: list, M_wd_obs: float, M2: float):
        self.M1_grid = M1_grid
        self.logger = logger
        self.M2 = M2
        self.M_wd_obs = M_wd_obs

    def _evolve_single(self, M1) -> pd.DataFrame | None:
        self.logger.info(f"Evolving M1={M1:.2f} Msun as a single star (wide binary) to build SSE grid")
        binary = InitialBinaryTable.InitialBinaries(
            m1=float(M1), m2=self.M2, porb=1e8, ecc=0.0, tphysf=14000.0,
            kstar1=1, kstar2=0, metallicity=0.014,
        )
        binary["dtp"] = 1.0
        bse_defaults = {
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
        sse_defaults = {"stellar_engine": "sse"}
        self.logger.info(f"Starting evolution for M1={M1:.2f} Msun")    
        _, bcm, _, _ = Evolve.evolve(
            initialbinarytable=binary, BSEDict=bse_defaults, SSEDict=sse_defaults
        )
        self.logger.info(f"Finished evolution for M1={M1:.2f} Msun, processing giant phases")
        ms_rows = bcm[bcm["kstar_1"] == 1]
        if len(ms_rows) == 0:
            self.logger.warning(f"M1={M1}: no main sequence rows in BCM, cannot determine TAMS radius")
            return None
        tams_radius = ms_rows["rad_1"].iloc[-1]
        self.logger.info(f"M1={M1}: TAMS radius = {tams_radius:.4f} Rsun")
        giants = bcm[bcm["kstar_1"].isin([2, 3, 4, 5, 6])].copy()
        if len(giants) == 0:
            self.logger.warning(f"M1={M1}: no giant phase reached")
            return None
        self.logger.info(f"M1={M1}: {len(giants)} giant phase entries found, computing core masses")
        giants["M1c"] = np.where(
            giants["kstar_1"].isin([2, 3]),
            giants["massc_he_layer_1"],
            giants["massc_co_layer_1"],
        )
        self.logger.info(f"M1={M1}: core masses computed, filtering for valid core mass entries")
        cols = ["tphys", "mass_1", "rad_1", "M1c", "kstar_1", "lum_1", "sep"]
        out = giants[cols].copy()
        out = out[np.isfinite(out["M1c"]) & (out["M1c"] > 0)]
        out["M1_init"] = M1
        out["rad_floor"] = tams_radius
        out["rad_ceil"] = out["rad_1"]
        self.logger.info(f"M1={M1}: {len(out)} valid giant phase entries with core mass, filtering for C/O core mass near observed WD mass")
        out = out[out["M1c"].between(self.M_wd_obs * 0.9, self.M_wd_obs * 1.1) & out["kstar_1"].isin([4, 5, 6])]
        return out if len(out) > 0 else None

    def compute_sse_grid(self) -> pd.DataFrame:
        frames = []
        self.logger.info(f"Computing SSE grid for M1 values: {self.M1_grid}")
        for m1 in self.M1_grid:
            result = self._evolve_single(M1=m1)
            if result is not None:
                frames.append(result)
        self.logger.info(f"Finished computing SSE grid, concatenating results")
        grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        self.logger.info(f"Grid ready: {len(frames)}/{len(self.M1_grid)} M1 values produced giant phases")
        return grid
