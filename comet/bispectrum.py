"""Bispectrum module."""

import numpy as np

class Bispectrum:

    def __init__(self, real_space, use_Mpc):
        self.real_space = real_space
        self.use_Mpc = use_Mpc
        self.nbar = 1.0 # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc
        self.tri_fixed = None

    def define_nbar(self, nbar):
        self.nbar = np.copy(nbar)

    def set_tri_fixed(self, tri_fixed, kfun):
        self.tri_fixed = tri_fixed
        self.kfun = kfun
        self.generate_index_arrays()
        self.compute_kernels()

    def F2(self, k1, k2, k3):
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu**2

    def K(self, k1, k2, k3):
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return mu**2 - 1.0

    def compute_kernels(self):
        self.F2_fixed = np.zeros([self.tri_fixed.shape[0],3])
        self.K_fixed = np.zeros([self.tri_fixed.shape[0],3])

        for i in range(3):
            self.F2_fixed[:,i] = self.F2(*np.roll(self.tri_fixed, -i, axis=1).T)
            self.K_fixed[:,i] = self.K(*np.roll(self.tri_fixed, -i, axis=1).T)

    def generate_index_arrays(self, round_decimals=2):
        self.tri_fixed_rounded = np.around(self.tri_fixed/self.kfun,
                                           decimals=round_decimals)
        self.tri_fixed_unique = np.unique(self.tri_fixed_rounded)
        self.ki, self.kj = np.meshgrid(self.tri_fixed_unique,
                                       self.tri_fixed_unique)
        ids = np.where(self.ki >= self.kj)
        self.ki = self.ki[ids]
        self.kj = self.kj[ids]

        self.tri_to_id = np.zeros_like(self.tri_fixed, dtype=int)
        self.tri_to_id_sq = np.zeros_like(self.tri_fixed, dtype=int)
        for n in range(self.tri_fixed.shape[0]):
            self.tri_to_id[n,0] = np.where(
                self.tri_fixed_unique == self.tri_fixed_rounded[n,0])[0]
            self.tri_to_id[n,1] = np.where(
                self.tri_fixed_unique == self.tri_fixed_rounded[n,1])[0]
            self.tri_to_id[n,2] = np.where(
                self.tri_fixed_unique == self.tri_fixed_rounded[n,2])[0]
            self.tri_to_id_sq[n,0] = np.where(
                (self.ki == self.tri_fixed_rounded[n,0]) & \
                (self.kj == self.tri_fixed_rounded[n,1]))[0]
            self.tri_to_id_sq[n,1] = np.where(
                (self.ki == self.tri_fixed_rounded[n,1]) & \
                (self.kj == self.tri_fixed_rounded[n,2]))[0]
            self.tri_to_id_sq[n,2] = np.where(
                (self.ki == self.tri_fixed_rounded[n,0]) & \
                (self.kj == self.tri_fixed_rounded[n,2]))[0]

        self.ki = np.searchsorted(self.tri_fixed_unique, self.ki)
        self.kj = np.searchsorted(self.tri_fixed_unique, self.kj)
        self.tri_fixed_unique *= self.kfun

    def Bell_fixed(self, PL_dw, params, ell):
        # currently only real-space without AP
        b1sq = params['b1']**2
        kernel = 2*b1sq * (params['b1']*self.F2_fixed + 0.5*params['b2'] +
                           params['g2']*self.K_fixed)
        P2 = PL_dw[self.ki]*PL_dw[self.kj]

        B_SPT = np.einsum("ij,ij->i", kernel, P2[self.tri_to_id_sq])
        B_stoch = b1sq*params['MB0']/self.nbar \
                  * np.sum(PL_dw[self.tri_to_id],axis=1) \
                  + params['NB0']/self.nbar**2

        Bell = {}
        Bell['ell0'] = B_SPT + B_stoch

        return Bell
