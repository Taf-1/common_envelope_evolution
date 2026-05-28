import logging
from astropy.constants import G, M_sun, R_sun
import numpy as np

class EnergyInversion:
    def __init__(self, logger: logging.Logger, mass_1: float, menv_1: float,
                 rad_1: float, m1c: float, m2: float, a_f: float, lam: float,
                 rad_floor: float, rad_ceil: float):
        self.logger = logger
        self.mass_1 = mass_1
        self.menv_1 = menv_1
        self.rad_1 = rad_1
        self.m1c = m1c
        self.m2 = m2
        self.a_f = a_f
        self.lam = lam
        self.rad_floor = rad_floor
        self.rad_ceil = rad_ceil

    @staticmethod
    def separation_to_period(separation: float, m1: float, m2: float) -> float:
        return 2 * np.pi * np.sqrt(separation**3 / (G.value * (m1 + m2) * M_sun.value))

    @staticmethod
    def eggleton_roche_lobe_radius(q: float) -> float:
        return 0.49 * q**(2/3) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))

    def binding_energy(self) -> float:
        return (G.value * self.mass_1 * M_sun.value * self.menv_1 * M_sun.value
                / (self.rad_1 * R_sun.value * self.lam))

    def solve_for_alpha(self) -> tuple:
        E_bind = self.binding_energy()
        q = self.mass_1 / self.m2
        f_rl = self.eggleton_roche_lobe_radius(q)
        a_i = self.rad_1 * R_sun.value / f_rl  # metres
        E_orb_i = G.value * self.mass_1 * M_sun.value * self.m2 * M_sun.value / (2 * a_i)
        E_orb_f = G.value * self.m1c * M_sun.value * self.m2 * M_sun.value / (2 * self.a_f * R_sun.value)
        delta_E_orb = E_orb_f - E_orb_i
        if delta_E_orb <= 0:
            raise ValueError(f"delta E_orb={delta_E_orb:.3e} J = 0; CE energy balance unphysical")
        alpha = E_bind / delta_E_orb
        period = self.separation_to_period(a_i, self.mass_1, self.m2)
        self.logger.debug(f"a_i={a_i/R_sun.value:.2f} Rsun  α={alpha:.3f}  P_init={period/86400:.4f} d")
        return a_i / R_sun.value, alpha, period
