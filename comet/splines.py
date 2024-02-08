"""Spline module."""

import numpy as np
from scipy.interpolate import make_interp_spline

class Splines:
    r"""Class for handling splined objects within Comet.
    """

    def __init__(self, ncol=0, id_min_ell6=0, crossover_check=False):
        self.id_min = 0 if ncol == 0 else np.zeros(ncol, dtype=np.int32)
        if ncol > 0:
            self.id_min[-1] = id_min_ell6
        self.col = None if ncol == 0 else np.arange(ncol, dtype=np.int32)
        self.id_max = -1 if ncol == 0 else -1*np.ones(ncol, dtype=np.int32)

    def build(self, x, y, axis=0):
        self.spline = make_interp_spline(x, y, axis=axis)

        # low-k extrapolation
        dly_min = np.log10(np.abs(
            y[self.id_min+2,self.col]/y[self.id_min,self.col]))
        dlx_min = np.atleast_1d(
            np.log10(np.abs(x[self.id_min+2]/x[self.id_min])))
        self.neff_min = dly_min/dlx_min[:,None]
        self.y_min = y[self.id_min,self.col]
        self.x_min = np.atleast_1d(x[self.id_min])
        if self.col is not None:
            self.x_min = self.x_min[:,None]
        self.extrapolation_min = lambda x: self.y_min \
            * np.divide.outer(x,self.x_min)**self.neff_min

        # high-k extrapolation
        dy_max = np.abs(y[self.id_max,self.col]/y[self.id_max-2,self.col])
        dly_max = np.log10(dy_max)
        dlx_max = np.atleast_1d(
            np.log10(np.abs(x[self.id_max]/x[self.id_max-2])))
        self.neff_max = dly_max/dlx_max[:,None]
        self.y_max = y[self.id_max,self.col]
        self.x_max = np.atleast_1d(x[self.id_max])
        if self.col is not None:
            self.x_max = self.x_max[:,None]
        self.extrapolation_max_plaw = lambda x: self.y_max \
            * np.divide.outer(x,self.x_max)**self.neff_max

        if crossover_check:
            self.mask = np.where((dy_max > 2) | (dy_max < 0.5))
            self.slope = (y[self.id_max,self.col] - y[self.id_max-2,self.col]) \
                         / (x[self.id_max] - x[self.id_max-2])[:,None]
            self.intrcpt = y[self.id_max-2,self.col] \
                           - self.slope * x[self.id_max-2]
            self.extrapolation_max_lin = lambda x: np.multiply.outer(x,
                self.slope) + self.intrcpt

            def extrapolation_max(x):
                plaw = self.extrapolation_max_plaw(x)
                lin = self.extrapolation_max_lin(x)
                plaw[:,self.mask[0],self.mask[1]] = \
                    lin[:,self.mask[0],self.mask[1]]
                return plaw

            self.extrapolation_max = extrapolation_max
        else:
            self.extrapolation_max = self.extrapolation_max_plaw

    def eval(self, x):
        spline = np.where(np.less.outer(x, self.x_min),
                          self.extrapolation_min(x),
                          np.where(np.greater.outer(x, self.x_max),
                                   self.extrapolation_max(x),
                                   self.spline(x)))
        return spline
