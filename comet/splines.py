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
            self.spline = [make_interp_spline(x, y[...,n], axis=axis) \
                           for n in range(self.size_last)]
        else:
            self.h = np.atleast_1d(h)
            self.h3 = self.h**3
            xh = np.divide.outer(x, self.h)
            yh3 = y*self.h3
            self.spline = [make_interp_spline(xh[...,n], yh3[...,n], axis=axis)\
                           for n in range(self.size_last)]

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
        if self.use_Mpc:
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min[...,0])**self.neff_min[...,n]
        else:
            self.y_min *= self.h3
            self.x_min = self.x_min/self.h
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min[...,n])**self.neff_min[...,n]

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
        if self.use_Mpc:
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.divide.outer(x,self.x_max[...,0])**self.neff_max[...,n]
        else:
            self.y_max *= self.h3
            self.x_max = self.x_max/self.h
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.divide.outer(x,self.x_max[...,n])**self.neff_max[...,n]

        if self.crossover_check:
            self.mask = np.where((dy_max > 2) | (dy_max < 0.5))
            self.slope = (y[self.id_max,self.col] - y[self.id_max-2,self.col]) \
                         / (x[self.id_max,None] - x[self.id_max-2,None])
            self.intrcpt = y[self.id_max-2,self.col] \
                           - self.slope * x[self.id_max-2,None]
            if not self.use_Mpc:
                self.slope *= self.h3*self.h
                self.intrcpt *= self.h3
            self.extrapolation_max_lin = lambda x,n: np.multiply.outer(x,
                self.slope[...,n]) + self.intrcpt[...,n]

    def _eval_extrapolation_min(self, x):
        y = np.array([self.extrapolation_min(x,n) \
                      for n in range(self.size_last)])
        return np.moveaxis(y, 0, -1)

    def _eval_extrapolation_min_varx(self, x):
        y = np.array([self.extrapolation_min(x[...,n],n) \
                      for n in range(self.size_last)])
        return np.moveaxis(y, 0, -1)

    def _eval_spline(self, x):
        y = np.array([self.spline[n](x) for n in range(self.size_last)])
        return np.moveaxis(y, 0, -1)

    def _eval_spline_varx(self, x):
        y = np.array([self.spline[n](x[...,n]) for n in range(self.size_last)])
        return np.moveaxis(y, 0, -1)

    def _eval_extrapolation_max(self, x):
        yplaw = np.array([self.extrapolation_max_plaw(x,n) \
                          for n in range(self.size_last)])
        yplaw = np.moveaxis(yplaw, 0, -1)
        if self.crossover_check:
            ylin = np.array([self.extrapolation_max_lin(x,n) \
                             for n in range(self.size_last)])
            ylin = np.moveaxis(ylin, 0, -1)
            yplaw[(Ellipsis, *self.mask)] = ylin[(Ellipsis, *self.mask)]
        return yplaw

    def _eval_extrapolation_max_varx(self, x):
        yplaw = np.array([self.extrapolation_max_plaw(x[...,n],n) \
                          for n in range(self.size_last)])
        yplaw = np.moveaxis(yplaw, 0, -1)
        if self.crossover_check:
            ylin = np.array([self.extrapolation_max_lin(x[...,n],n) \
                             for n in range(self.size_last)])
            ylin = np.moveaxis(ylin, 0, -1)
            yplaw[(Ellipsis, *self.mask)] = ylin[(Ellipsis, *self.mask)]
        return yplaw

    def eval(self, x):
        spline = np.where(np.less.outer(x, self.x_min),
                          self._eval_extrapolation_min(x),
                          np.where(np.greater.outer(x, self.x_max),
                                   self._eval_extrapolation_max(x),
                                   self._eval_spline(x)))
        return spline

    def eval_varx(self, x):
        n = len(x.shape) - 1
        if self.use_Mpc:
            mask_less = np.moveaxis(np.less.outer(x, self.x_min[:,0]), n, -1)
            mask_greater = np.moveaxis(
                np.greater.outer(x, self.x_max[:,0]), n, -1)
        else:
            mask_less = np.moveaxis(
                np.less.outer(x, self.x_min), n, -1)[...,self.ids,self.ids]
            mask_greater = np.moveaxis(
                np.greater.outer(x, self.x_max), n, -1)[...,self.ids,self.ids]
        spline = np.where(mask_less,
                          self._eval_extrapolation_min_varx(x),
                          np.where(mask_greater,
                                   self._eval_extrapolation_max_varx(x),
                                   self._eval_spline_varx(x)))
        return spline

    # def _eval(self, x):
    #     mask_less = np.less.outer(x, self.x_min)[...,0]
    #     mask_greater = np.greater.outer(x, self.x_max)[...,0]
    #     mask_remainder = ~mask_less & ~mask_greater
    #
    #     res = np.zeros(x.shape+(self.ncol,self.size_last)) if self.ncol > 0 \
    #         else np.zeros(x.shape+(self.size_last,))
    #
    #     if self.ncol > 0:
    #         eval_less = self.extrapolation_min(x[mask_less[...,-1]])
    #         eval_greater = self.extrapolation_max(x[mask_greater[...,0]])
    #         eval_remainder = self.spline(x[mask_remainder[...,0]])
    #         for i in range(self.ncol):
    #             res[mask_less[...,i],i,...] = eval_less[
    #                 mask_less[mask_less[...,-1],i],i,...]
    #             res[mask_greater[...,i],i,...] = eval_greater[
    #                 mask_greater[mask_greater[...,0],i],i,...]
    #             res[mask_remainder[...,i],i,...] = eval_remainder[
    #                 mask_remainder[mask_remainder[...,0],i],i,...]
    #     else:
    #         res[mask_less] = self.extrapolation_min(x[mask_less])
    #         res[mask_greater] = self.extrapolation_max(x[mask_greater])
    #         res[mask_remainder] = self.spline(x[mask_remainder])
    #
    #     return res

    # def eval(self, x):
    #     x = np.atleast_1d(x)
    #     if self.use_Mpc:
    #         spline = self._eval(x)
    #     else:
    #         xh = np.outer(self.h, x)
    #         spline = self._eval(xh)
    #         spline = np.moveaxis(spline[self.ids,...,self.ids], 0, -1)
    #     return spline
