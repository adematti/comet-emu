"""Spline module."""

import numpy as np
from scipy.interpolate import make_interp_spline

class Splines:
    r"""Class for handling splined objects within Comet.
    """

    def __init__(self, use_Mpc, ncol=0, id_min_ell6=0, crossover_check=False):
        self.use_Mpc = use_Mpc
        self.id_min = 0 if ncol == 0 else np.zeros(ncol, dtype=np.int32)
        self.ncol = ncol
        if self.ncol > 0:
            self.id_min[-1] = id_min_ell6
        self.col = None if ncol == 0 else np.arange(ncol, dtype=np.int32)
        self.id_max = -1 if ncol == 0 else -1*np.ones(ncol, dtype=np.int32)
        self.crossover_check = crossover_check

    def build(self, x, y, h=None, axis=0):
        self.size_last = y.shape[-1]
        self.ids = np.arange(self.size_last, dtype=np.int32)

        if self.use_Mpc:
            self.spline = make_interp_spline(x, y, axis=axis)
        else:
            self.h = h
            self.h3 = self.h**3
            self.spline = make_interp_spline(x, y*self.h3, axis=axis)

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

        if not self.use_Mpc:
            self.y_min *= self.h3
            self.y_max *= self.h3

        if self.crossover_check:
            self.mask = np.where((dy_max > 2) | (dy_max < 0.5))
            self.slope = (y[self.id_max,self.col] - y[self.id_max-2,self.col]) \
                         / (x[self.id_max,None] - x[self.id_max-2,None])
            self.intrcpt = y[self.id_max-2,self.col] \
                           - self.slope * x[self.id_max-2,None]
            self.extrapolation_max_lin = lambda x: np.multiply.outer(x,
                self.slope) + self.intrcpt

            if not self.use_Mpc:
                self.slope *= self.h3
                self.intrcpt *= self.h3

            def extrapolation_max(x):
                plaw = self.extrapolation_max_plaw(x)
                lin = self.extrapolation_max_lin(x)
                plaw[...,self.mask] = lin[...,self.mask]
                return plaw

            self.extrapolation_max = extrapolation_max
        else:
            self.extrapolation_max = self.extrapolation_max_plaw

    def _eval(self, x):
        mask_less = np.less.outer(x, self.x_min)[...,0]
        mask_greater = np.greater.outer(x, self.x_max)[...,0]
        mask_remainder = ~mask_less & ~mask_greater

        res = np.zeros(x.shape+(self.ncol,self.size_last)) if self.ncol > 0 \
            else np.zeros(x.shape+(self.size_last,))

        if self.ncol > 0:
            eval_less = self.extrapolation_min(x[mask_less[...,-1]])
            eval_greater = self.extrapolation_max(x[mask_greater[...,0]])
            eval_remainder = self.spline(x[mask_remainder[...,0]])
            for i in range(self.ncol):
                res[mask_less[...,i],i,...] = eval_less[
                    mask_less[mask_less[...,-1],i],i,...]
                res[mask_greater[...,i],i,...] = eval_greater[
                    mask_greater[mask_greater[...,0],i],i,...]
                res[mask_remainder[...,i],i,...] = eval_remainder[
                    mask_remainder[mask_remainder[...,0],i],i,...]
        else:
            res[mask_less] = self.extrapolation_min(x[mask_less])
            res[mask_greater] = self.extrapolation_max(x[mask_greater])
            res[mask_remainder] = self.spline(x[mask_remainder])

        return res

    def eval(self, x):
        x = np.atleast_1d(x)
        if self.use_Mpc:
            spline = self._eval(x)
        else:
            xh = np.outer(self.h, x)
            spline = self._eval(xh)
            spline = np.moveaxis(spline[self.ids,...,self.ids], 0, -1)
        return spline
