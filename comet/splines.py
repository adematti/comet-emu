"""Spline module."""

import numpy as np
from scipy.interpolate import make_interp_spline

class Splines:
    r"""Class for handling splined objects within Comet.
    """

    def __init__(self, use_Mpc, ncol=0, id_min=0, crossover_check=False):
        self.use_Mpc = use_Mpc
        self.id_min = id_min
        self.id_max = -1
        self.ncol = tuple([ncol]) if not isinstance(ncol, tuple) else ncol
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
        self.y_min = y[self.id_min]
        self.x_min = np.atleast_1d(x[self.id_min])
        ids_nonzero = self.y_min != 0
        dly_min = np.empty_like(self.y_min)
        dly_min[ids_nonzero] = np.log10(
            np.abs(y[self.id_min+2][ids_nonzero]/self.y_min[ids_nonzero]))
        dly_min[~ids_nonzero] = 0.0
        dlx_min = np.log10(np.abs(x[self.id_min+2]/x[self.id_min]))
        self.neff_min = dly_min/dlx_min
        if self.use_Mpc:
            if len(self.ncol) > 1:
                self.x_min = self.x_min[:,np.newaxis]
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min)**self.neff_min[...,n]
            self.extrapolation_min_derivative = lambda x,n: self.y_min[...,n] \
                * np.divide.outer(x,self.x_min)**(self.neff_min[...,n]-1) \
                * self.neff_min[...,n] / self.x_min
        else:
            self.y_min = self.y_min*self.h3
            self.x_min = self.x_min/self.h
            self.extrapolation_min = lambda x,n: self.y_min[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_min[...,n]),self.neff_min[...,n])
            self.extrapolation_min_derivative = lambda x,n: self.y_min[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_min[...,n]),
                    self.neff_min[...,n]-1) \
                * self.neff_min[...,n] / self.x_min[...,n]

        # high-k extrapolation
        self.y_max = y[self.id_max]
        self.x_max = np.atleast_1d(x[self.id_max])
        ids_nonzero = y[self.id_max-2] != 0
        dy_max = np.empty_like(self.y_max)
        dy_max[ids_nonzero] = np.abs(
            self.y_max[ids_nonzero]/y[self.id_max-2][ids_nonzero])
        dy_max[~ids_nonzero] = 1.0
        dly_max = np.log10(dy_max)
        dlx_max = np.log10(np.abs(x[self.id_max]/x[self.id_max-2]))
        self.neff_max = dly_max/dlx_max
        self.neff_max[np.isnan(self.neff_max)] = 0.0
        if self.use_Mpc:
            if len(self.ncol) > 1:
                self.x_max = self.x_max[:,np.newaxis]
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.divide.outer(x,self.x_max)**self.neff_max[...,n]
            self.extrapolation_max_plaw_derivative = lambda x,n: \
                self.y_max[...,n] \
                * np.divide.outer(x,self.x_max)**(self.neff_max[...,n]-1) \
                * self.neff_max[...,n] / self.x_max
        else:
            self.y_max = self.y_max*self.h3
            self.x_max = self.x_max/self.h
            self.extrapolation_max_plaw = lambda x,n: self.y_max[...,n] \
                * np.power.outer(
                    np.divide.outer(x,self.x_max[...,n]), self.neff_max[...,n])
            self.extrapolation_max_plaw_derivative = lambda x,n: \
                self.y_max[...,n] * np.power.outer(
                    np.divide.outer(x,self.x_max[...,n]),
                    self.neff_max[...,n]-1) \
                * self.neff_max[...,n] / self.x_max[...,n]

        if self.crossover_check:
            self.mask = np.where((dy_max > 2) | (dy_max < 0.5))
            self.slope = (y[self.id_max] - y[self.id_max-2]) \
                         / (x[self.id_max] - x[self.id_max-2])
            self.intrcpt = y[self.id_max-2] - self.slope * x[self.id_max-2]
            if not self.use_Mpc:
                self.slope *= self.h3*self.h
                self.intrcpt *= self.h3
            self.extrapolation_max_lin = lambda x,n: np.multiply.outer(x,
                self.slope[...,n]) + self.intrcpt[...,n]
            self.extrapolation_max_lin_derivative = lambda x,n: \
                np.multiply.outer(np.ones_like(x),self.slope[...,n])

    def extrapolation_max(self, x, n):
        y = self.extrapolation_max_plaw(x,n) # nx x nell
        if self.crossover_check and sum(self.mask[-1] == n) > 0:
            ids = (Ellipsis,) + tuple(self.mask[i][self.mask[-1] == n] \
                                      for i in range(len(self.mask)-1))
            y[ids] = self.extrapolation_max_lin(x,n)[ids]
        return y

    def extrapolation_max_derivative(self, x, n):
        y = self.extrapolation_max_plaw_derivative(x,n) # nx x nell
        if self.crossover_check and sum(self.mask[-1] == n) > 0:
            ids = (Ellipsis,) + tuple(self.mask[i][self.mask[-1] == n] \
                                      for i in range(len(self.mask)-1))
            y[ids] = self.extrapolation_max_lin_derivative(x,n)[ids]
        return y

    def _eval_extrapolation_min(self, x):
        y = np.stack([np.squeeze(self.extrapolation_min(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_min_derivative(self, x):
        y = np.stack([np.squeeze(self.extrapolation_min_derivative(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_spline(self, x):
        y = np.stack([np.squeeze(self.spline[n](x)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_spline_derivative(self, x):
        y = np.stack([np.squeeze(self.spline[n].derivative(1)(x)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_max(self, x):
        y = np.stack([np.squeeze(self.extrapolation_max(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def _eval_extrapolation_max_derivative(self, x):
        y = np.stack([np.squeeze(self.extrapolation_max_derivative(x,n)) \
                      for n in range(self.size_last)],
                     axis=-1)
        return np.atleast_2d(y)

    def eval(self, x):
        mask_less = x < self.x_min if self.use_Mpc \
                    else x[:,None] < self.x_min # -> nx x N
        mask_greater = x > self.x_max if self.use_Mpc \
                       else x[:,None] > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape+(self.size_last,))

        y[mask_less] = self._eval_extrapolation_min(x)[mask_less]
        y[mask] = self._eval_spline(x)[mask]
        y[mask_greater] = self._eval_extrapolation_max(x)[mask_greater]
        return y

    def eval_varx(self, x): # last dimension of x matches self.size_last
        n = len(x.shape) - 1
        mask_less = x < self.x_min # -> nx x N
        mask_greater = x > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape[:n] + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape[:n]+(self.size_last,))

        for i in range(self.size_last):
            y[mask_less[...,i],...,i] = \
                self.extrapolation_min(x[mask_less[...,i],i],i)
            y[mask[...,i],...,i] = self.spline[i](x[mask[...,i],i])
            y[mask_greater[...,i],...,i] = \
                self.extrapolation_max(x[mask_greater[...,i],i],i)
        return y

    # this is just a quick fix for compatibility with the bispectrum module
    # derivative should also be applied to the extrapolations
    def derivative(self, n):
        return self.spline[0].derivative(n)

    def eval_derivative(self, x):
        mask_less = x < self.x_min if self.use_Mpc \
                    else x[:,None] < self.x_min # -> nx x N
        mask_greater = x > self.x_max if self.use_Mpc \
                       else x[:,None] > self.x_max
        mask = ~mask_less & ~mask_greater

        y = np.empty(x.shape + (*self.ncol,self.size_last,)) \
            if sum(self.ncol) > 0 else np.empty(x.shape+(self.size_last,))

        y[mask_less] = self._eval_extrapolation_min_derivative(x)[mask_less]
        y[mask] = self._eval_spline_derivative(x)[mask]
        y[mask_greater] = self._eval_extrapolation_max_derivative(x) \
                          [mask_greater]
        return y
