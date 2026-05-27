import logging
from astropy.constants import G, M_sun, R_sun
import numpy as np

class EnergyInversion:
    def __init__(self, logger: logging.Logger, mass_1: float, menv_1: float, 
                 rad_1: float, m1c: float, m2: float, a_f: float, lam: float, alpha: float,
                 rad_floor: float, rad_ceil: float):
        self.logger = logger
        self.mass_1 = mass_1
        self.menv_1 = menv_1
        self.rad_1 = rad_1
        self.m1c = m1c
        self.m2 = m2
        self.a_f = a_f
        self.lam = lam
        self.alpha = alpha
        self.rad_floor = rad_floor
        self.rad_ceil = rad_ceil

    @staticmethod
    def separation_to_period(separation: float, m1: float, m2: float) -> float:
        return 2 * np.pi * np.sqrt(separation**3 / (G.value * (m1 + m2) * M_sun.value))
    
    @staticmethod
    def eggleton_roche_lobe_radius(q: float) -> float:
        return 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))

    def binding_energy(self) -> float:
        self.logger.info("Calculating binding energy of the envelope")
        return G.value*self.mass_1*M_sun.value*self.menv_1*M_sun.value/(self.rad_1*R_sun.value*self.lam)
    
    def solve_for_ai(self) -> tuple:
        self.logger.info("Solving for initial orbital separation using energy balance")
        E_bind = self.binding_energy()
        denom = self.alpha * G.value * self.m1c*M_sun.value * self.m2*M_sun.value / (self.a_f*R_sun.value) - 2 * E_bind
        if denom <= 0:
            self.logger.error("Denominator in energy inversion is non-positive, check parameters")
            raise ValueError("Invalid parameters leading to non-physical solution")
        a_i = self.alpha * G.value * self.mass_1*M_sun.value * self.m2*M_sun.value / denom
        self.logger.info(f"Calculated initial orbital separation: {a_i:.3e} m")
        period = self.separation_to_period(a_i, self.mass_1, self.m2)
        self.logger.info(f"Corresponding initial orbital period: {period:.3e} s")
        self.logger.info("Checking if initial separation is within physical bounds based on stellar radius")
        q = self.mass_1 / self.m2
        f_rlobe = self.eggleton_roche_lobe_radius(q)
        a_floor = self.rad_floor*R_sun.value / f_rlobe
        a_ceil = self.rad_ceil*R_sun.value / f_rlobe
        if a_i < a_floor or a_i > a_ceil:
            self.logger.error(f"Calculated a_i={a_i:.3e} m is outside physical bounds [{a_floor:.3e}, {a_ceil:.3e}] m based on stellar radius and Roche lobe")
            raise ValueError(f"a_i={a_i:.3e} outside physical bounds [{a_floor:.3e}, {a_ceil:.3e}]")
        return a_i, period