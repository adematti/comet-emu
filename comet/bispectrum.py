"""Bispectrum module."""

import numpy as np

class Bispectrum:

    def __init__(self, real_space, use_Mpc):
        self.real_space = real_space
        self.use_Mpc = use_Mpc
        self.nbar = 1.0 # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc
        self.tri_fixed = None

        if self.real_space:
            self.kernel_names = ['F2', 'K']
        else:
            self.kernel_names = ['F2', 'G2', 'K', 'k31', 'k32',
                                 'dF2_dlnk1', 'dF2_dlnk2', 'dF2_dlnk3',
                                 'dG2_dlnk1', 'dG2_dlnk2', 'dG2_dlnk3',
                                 'dK_dlnk1', 'dK_dlnk2', 'dK_dlnk3',
                                 'dk31_dlnk1', 'dk31_dlnk2', 'dk31_dlnk3',
                                 'dk32_dlnk1', 'dk32_dlnk2', 'dk32_dlnk3']
            self.I_names = ['000', '']

        self.kernels_fixed = {}
        self.I_fixed = {}

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

    def G2(self, k1, k2, k3):
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return 3.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 4.0/7.0 * mu**2

    def K(self, k1, k2, k3):
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return mu**2 - 1.0

    def kernels_real_space(self, k1, k2, k3):
        kernels = {}
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        kernels['F2'] = 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu**2
        kernels['K'] = mu**2 - 1.0
        return kernels

    def kernels_redshift_space(self, k1, k2, k3):
        kernels = {}
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)

        kernels['F2'] = 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu**2
        kernels['G2'] = 3.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 4.0/7.0 * mu**2
        kernels['K'] = mu**2 - 1.0
        kernels['k31'] = k3/k1
        kernels['k32'] = k3/k2

        kernels['dF2_dlnk1'] = -0.5 - k1**2/(2.0*k2**2) - (4.0*k1*mu)/(7.0*k2) \
                               - (k2*mu)/k1 - (4.0*mu**2)/7.0
        kernels['dF2_dlnk2'] = -0.5 - k2**2/(2.0*k1**2) - (k1*mu)/k2 \
                               - (4.0*k2*mu)/(7.0*k1) - (4.0*mu**2)/7.0
        kernels['dF2_dlnk3'] = (k3**2*(7.0*(k1**2 + k2**2) + 8.0*k1*k2*mu)) \
                               / (14.0*k1**2*k2**2)

        kernels['dG2_dlnk1'] = -0.5 - k1**2/(2.0*k2**2) - (8.0*k1*mu)/(7.0*k2) \
                               - (k2*mu)/k1 - (8.0*mu**2)/7.0
        kernels['dG2_dlnk2'] = -0.5 - k2**2/(2.0*k1**2) - (k1*mu)/k2 \
                               - (8.0*k2*mu)/(7.0*k1) - (8.0*mu**2)/7.0
        kernels['dG2_dlnk3'] = (k3**2*(7.0*(k1**2 + k2**2) + 16.0*k1*k2*mu)) \
                               / (14.*k1**2*k2**2)

        kernels['dK_dlnk1'] = (-2.0*mu*(k1 + k2*mu))/k2
        kernels['dK_dlnk2'] = (-2.0*mu*(k2 + k1*mu))/k1
        kernels['dK_dlnk3'] = 2.0*mu*(k1/k2 + k2/k1 + 2.0*mu)

        kernels['dk31_dlnk1'] = -kernels['k31']
        kernels['dk31_dlnk2'] = 0.0
        kernels['dk31_dlnk3'] = kernels['k31']

        kernels['dk32_dlnk1'] = 0.0
        kernels['dk32_dlnk2'] = -kernels['k32']
        kernels['dk32_dlnk3'] = kernels['k32']

    def compute_kernels(self):
        for kk in self.kernel_names:
            self.kernels_fixed[kk] = np.zeros([self.tri_fixed.shape[0],3])

        for i in range(3):
            k123_perm = np.roll(self.tri_fixed, -i, axis=1).T
            if self.real_space:
                kernels = self.kernels_real_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels_fixed[kk][:,i] = kernels[kk]
            else:
                kernels = self.kernels_redshift_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels_fixed[kk][:,i] = kernels[kk]

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
