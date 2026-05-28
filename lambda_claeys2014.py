import numpy as np 
import logging

class Lambda:
    def __init__(self, logger: logging.Logger, kstar_1: int, lumin_1: float, mass_1: float, rad_1: float, menv_1: float, r_zams: float):
        self.logger = logger
        self.lumin_1 = lumin_1
        self.mass_1 = mass_1
        self.rad_1 = rad_1
        self.menv_1 = menv_1
        self.r_zams = r_zams
        self.kstar_1 = kstar_1
    
    @staticmethod
    def compute_agb_lambda(logger: logging.Logger, mass: float, lum: float, kstar: int) -> float:
        logger.debug("Computing AGB lambda using Claeys+2014 prescription")
        lambda_3 = np.minimum(-0.9, 0.58 + 0.75*np.log10(mass)) - 0.08*np.log10(lum)
        if kstar == 4:
            logger.debug("Star is a CHeB, using appropriate formula")
            return 2 * 0.5
        elif kstar == 5:
            logger.debug("Star is an EAGB, using appropriate formula")
            return 2 * np.minimum(0.8, np.minimum(1.25 - 0.15*np.log10(lum), lambda_3))
        else:
            logger.debug("Star is a TPAGB, using appropriate formula")
            return 2 * np.minimum(np.maximum(-3.5 - 0.75*np.log10(mass) + np.log10(lum), lambda_3), 1.0)

    @staticmethod
    def compute_rgb_lambda(logger: logging.Logger, mass: float, menv: float, lum: float, rad: float, r_zams: float, kstar: int) -> float:
        logging.debug("Computing RGB lambda using Claeys+2014 prescription")
        lambda_2 = 0.42 * (r_zams / rad) ** 0.4
        lambda_3 = np.minimum(-0.9, 0.58 + 0.75*np.log10(mass)) - 0.08*np.log10(lum)
        if menv == 0:
            logger.debug("Envelope mass is zero, using lambda2")
            return 2 * lambda_2
        elif kstar == 4:
            logger.debug("Envelope mass is non-zero but <= 1, using minimum of lambda2 and envelope mass")
            return 2 * (lambda_2 + (menv**0.5)*(0.5 - lambda_2))
        elif kstar == 5:
            logger.debug("Envelope mass is non-zero but <= 1, using minimum of lambda2 and envelope mass")
            lambda_1 = np.minimum(0.8, np.minimum(1.25 - 0.15*np.log10(lum), lambda_3))
            return 2 * (lambda_2 + (menv**0.5)*(lambda_1 - lambda_2))
        else:
            logger.debug("Envelope mass is non-zero but <= 1, using minimum of lambda2 and envelope mass")
            lambda_1 = np.minimum(np.maximum(-3.5 - 0.75*np.log10(mass) + np.log10(lum), lambda_3), 1.0)
            return 2 * (lambda_2 + (menv**0.5)*(lambda_1 - lambda_2))

    def compute_lambda(self) -> float:
        self.logger.debug("Computing lambda using Claeys+2014 prescription")
        self.logger.debug("First, checking if envelope mass > 1")
        if self.menv_1 > 1.0:
            self.logger.debug("Envelope mass > 1, using AGB lambda")
            return self.compute_agb_lambda(self.logger, self.mass_1, self.lumin_1, self.kstar_1)
        else:
            self.logger.debug("Envelope mass <= 1, using RGB lambda")
            return self.compute_rgb_lambda(self.logger, self.mass_1, self.menv_1, self.lumin_1, self.rad_1, self.r_zams, self.kstar_1)