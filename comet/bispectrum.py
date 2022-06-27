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
            self.I_tuples_ell0 = [(0,0,0), (2,0,0), (1,1,0), (4,0,0), (3,1,0),
                                  (2,2,0), (2,1,1), (6,0,0), (5,1,0), (4,2,0),
                                  (4,1,1), (3,3,0), (3,2,1), (2,2,2), (6,1,1),
                                  (5,2,1), (4,3,1), (4,2,2), (3,3,2), (6,3,1),
                                  (5,4,1), (4,3,3)]
            self.I_tuples_ell2 = [(8,0,0), (7,1,0), (6,2,0), (5,3,0), (8,1,1),
                                  (7,2,1), (6,2,2), (5,3,1), (8,3,1), (7,4,1),
                                  (6,3,3)]
            self.I_tuples_ell4 = [(6,1,1), (10,0,0), (9,1,0), (8,2,0), (8,1,1),
                                  (7,3,0), (10,1,1), (9,2,1), (8,2,2), (7,3,2),
                                  (10,3,1), (9,4,1), (8,3,3)]

        self.kernels = {}
        self.I = {}

    def define_units(self, use_Mpc):
        self.use_Mpc = use_Mpc

    def define_nbar(self, nbar):
        self.nbar = np.copy(nbar)

    def set_tri_fixed(self, tri_fixed, kfun):
        self.tri_fixed = tri_fixed
        self.kfun = kfun
        self.generate_index_arrays()
        self.compute_kernels()
        if not self.real_space:
            self.compute_mu123_integrals()

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
        return kernels

    def mu123_integrals(self, n1, n2, n3, k1, k2, k3):
        if n2 == 0 and n3 == 0:
            I = 1.0/(1.0 + n1)
        elif n2 == 1 and n3 == 0:
            I = -0.5/(2.0 + n1) * (k1**2 + k2**2 - k3**2)/(k1*k2)
        elif n2 == 2 and n3 == 0:
            I = (4*k1**2*k3**2 + (k1**2 - k2**2 + k3**2)**2*n1) \
                / (4.*k1**2*k3**2*(1 + n1)*(3 + n1))
        elif n2 == 1 and n3 == 1:
            I = (-2*k1**2*(k2**2 + k3**2) - (k2**2 - k3**2)**2*n1 \
                + k1**4*(2 + n1))/(4.*k1**2*k2*k3*(3 + 4*n1 + n1**2))
        elif n2 == 3 and n3 == 0:
            I = -0.125*((k1**2 + k2**2 - k3**2)*(k1**4*(-1 + n1) \
                + (k2**2 - k3**2)**2*(-1 + n1) + 2*k1**2*(-(k3**2*(-1 + n1)) \
                + k2**2*(5 + n1))))/(k1**3*k2**3*(2 + n1)*(4 + n1))
        elif n2 == 2 and n3 == 1:
            I = ((k2**2 - k3**2)**3*(-1 + n1) - k1**6*(3 + n1) \
                + k1**2*(k2 - k3)*(k2 + k3)*(-(k3**2*(-5 + n1)) \
                + k2**2*(7 + n1)) + k1**4*(-(k2**2*(3 + n1)) + k3**2*(7 + n1)))\
                / (8.*k1**3*k2**2*k3*(2 + n1)*(4 + n1))
        elif n2 == 3 and n3 == 1:
            I = (-((k2**2 - k3**2)**4*(-2 + n1)*n1) + k1**8*n1*(4 + n1) \
                - 6*k1**4*(-3*k3**4*n1 + 2*k2**2*k3**2*(2 + n1) \
                + k2**4*(4 + n1)) - 2*k1**2*(k2**2 - k3**2)**2*n1 \
                * (-(k3**2*(-5 + n1)) + k2**2*(7 + n1)) + 2*k1**6*(k2**2 \
                * (3 + n1)*(4 + n1) - k3**2*n1*(7 + n1))) \
                / (16.*k1**4*k2**3*k3*(1 + n1)*(3 + n1)*(5 + n1))
        elif n2 == 2 and n3 == 2:
            I = (12*k1**2*(k2**2 - k3**2)**2*(k2**2 + k3**2)*n1 \
                + (k2**2 - k3**2)**4*(-2 + n1)*n1 - 4*k1**6*(k2**2 + k3**2) \
                * (4 + n1) + k1**8*(2 + n1)*(4 + n1) - 2*k1**4*(-2*k2**2*k3**2 \
                * (2 + n1)*(4 + n1) + k2**4*(-4 + n1*(6 + n1)) + k3**4 \
                * (-4 + n1*(6 + n1)))) \
                / (16.*k1**4*k2**2*k3**2*(1 + n1)*(3 + n1)*(5 + n1))
        elif n2 == 4 and n3 == 1:
            I = (-4*(k1**2 + k2**2 - k3**2)**4*(k1**2 - k2**2 + k3**2)
                + (8*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 - k2 + k3) \
                * (k1 + k2 + k3)*(k1**2 + k2**2 - k3**2)**2 \
                * (k1**2 - 5*k2**2 + 5*k3**2))/(4 + n1) \
                + (12*(3*k1**2 + 5*k2**2 - 5*k3**2)*(k1**4 + (k2**2 - k3**2)**2\
                - 2*k1**2*(k2**2 + k3**2))**2) \
                / ((2 + n1)*(4 + n1)))/(128.*k1**5*k2**4*k3*(6 + n1))
        elif n2 == 3 and n3 == 2:
            I = -0.03125*(3*(k1**2 + 5*k2**2 - 5*k3**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2*n1 \
                + (k1**2 + k2**2 - k3**2)*n1*(2 + n1) \
                * (-8*k1**6*k3**2 + 8*k1**2*(k2**2 - k3**2)**2 \
                * (3*k2**2 + 2*k3**2) + (k2**2 - k3**2)**4*(-6 + n1) \
                + k1**8*(6 + n1) - 2*k1**4*(k2 - k3)*(k2 + k3) \
                * (-(k3**2*(4 + n1)) + k2**2*(12 + n1)))) \
                / (k1**5*k2**3*k3**2*n1*(2 + n1)*(4 + n1)*(6 + n1))
        elif n2 == 3 and n3 == 3:
            I = (30*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                + 18*(k1**4 - 5*(k2**2 - k3**2)**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 \
                * (1 + n1) + 2*(k1**4 - (k2**2 - k3**2)**2)*(1 + n1)*(3 + n1) \
                * (-6*k1**6*(k2**2 + k3**2) + 30*k1**2*(k2**2 - k3**2)**2 \
                * (k2**2 + k3**2) + (k2**2 - k3**2)**4*(-10 + n1) + k1**8 \
                * (8 + n1) - 2*k1**4*(k2**2 - k3**2)**2*(11 + n1))) \
                / (128.*k1**6*k2**3*k3**3*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))
        return I

    def compute_kernels(self):
        for kk in self.kernel_names:
            self.kernels[kk] = np.zeros([self.tri_fixed.shape[0],3])

        for i in range(3):
            k123_perm = np.roll(self.tri_fixed, -i, axis=1).T
            if self.real_space:
                kernels = self.kernels_real_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels[kk][:,i] = kernels[kk]
            else:
                kernels = self.kernels_redshift_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels[kk][:,i] = kernels[kk]
                self.kernels['b2'] = 1.0
                self.kernels['db2_dlnk1'] = 0.0
                self.kernels['db2_dlnk2'] = 0.0
                self.kernels['db2_dlnk3'] = 0.0

    def compute_mu123_integrals(self):
        for n123 in self.I_tuples_ell0 + self.I_tuples_ell2 \
                + self.I_tuples_ell4:
            self.I[n123] = np.zeros([self.tri_fixed.shape[0],3])
            if n123[0] != n123[1] and n123[1] != n123[2]:
                n123_odd = (n123[0], n123[2], n123[1])
                self.I[n123_odd] = np.zeros([self.tri_fixed.shape[0],3])
            for i in range(3):
                k123_perm_even = np.roll(self.tri_fixed, -i, axis=1).T
                self.I[n123][:,i] = self.mu123_integrals(*n123, *k123_perm_even)
                if n123[0] != n123[1] and n123[1] != n123[2]:
                    n123_odd = (n123[0], n123[2], n123[1])
                    k123_perm_odd = np.roll(self.tri_fixed[:,[0,2,1]], -i,
                                            axis=1).T
                    self.I[n123_odd][:,i] = self.mu123_integrals(*n123,
                                                                 *k123_perm_odd)
            if (n123[0] != n123[1] and n123[1] == n123[2]) or \
                    (n123[0] == n123[1] and n123[1] != n123[2]):
                for i in range(1,3):
                    n123_perm = tuple(np.roll(np.array(n123), -i))
                    self.I[n123_perm] = np.roll(self.I[n123], -i, axis=1)
            elif n123[0] != n123[1] and n123[1] != n123[2]:
                for i in range(1,3):
                    n123_odd = (n123[0], n123[2], n123[1])
                    n123_perm_even = tuple(np.roll(np.array(n123), -i))
                    n123_perm_odd = tuple(np.roll(np.array(n123_odd), -i))
                    self.I[n123_perm_even] = np.roll(self.I[n123], -i, axis=1)
                    self.I[n123_perm_odd] = np.roll(self.I[n123_odd], -i,
                                                    axis=1)

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

    def join_kernel_mu123_integral(self, K, n123_tuples, neff, coeff,
                                   alpha_tr, alpha_lo):
        K_neff1 = neff[self.tri_to_id]*self.kernels[K]
        K_neff2 = neff[self.tri_to_id[:,[1,2,0]]]*self.kernels[K]
        K_deriv_sum = np.sum([self.kernels['d{}_dlnk{}'.format(K,i+1)]
                              for i in range(3)])
        DeltaB_K = 0.0
        for i, n123 in enumerate(n123_tuples):
            t1 = self.I[n123] * ((1.0 + (alpha_tr-alpha_lo)*sum(n123)) * \
                                       self.kernels[K] \
                                       + (1.0-alpha_tr) * K_deriv_sum \
                                       + (1.0-alpha_tr) * (K_neff1 + K_neff2))
            t2 = self.I[n123[0]+2,n123[1],n123[2]] * (alpha_tr - alpha_lo) \
                 * (self.kernels['d{}_dlnk1'.format(K)] + K_neff1 \
                    - n123[0]*self.kernels[K])
            t3 = self.I[n123[0],n123[1]+2,n123[2]] * (alpha_tr - alpha_lo) \
                 * (self.kernels['d{}_dlnk2'.format(K)] + K_neff2 \
                    - n123[1]*self.kernels[K])
            t4 = self.I[n123[0],n123[1],n123[2]+2] * (alpha_tr - alpha_lo) \
                 * (self.kernels['d{}_dlnk3'.format(K)] \
                    - n123[2]*self.kernels[K])
            DeltaB_K += coeff[i] * (t1 + t2 + t3 + t4)

        return DeltaB_K

    def Bell_fixed(self, PL_dw, params, ell, neff=None):
        if self.real_space:
            b1sq = params['b1']**2
            kernel = 2*b1sq * (params['b1']*self.kernels['F2']
                               + 0.5*params['b2'] +
                               params['g2']*self.kernels['K'])
            kernel_stoch = b1sq/self.nbar
        else:
            b1sq = params['b1']**2
            f2b1 = params['f']**2/params['b1']
            f3b1sq = params['f']**3/params['b1']**2
            params_F2 = [params['b1'], params['f'], params['f'], f2b1]
            params_b2 = [params['b1']*params['b2'], params['f']*params['b2'],
                         params['f']*params['b2'], params['b2']*f2b1]
            params_K = [params['b1']*params['g2'], params['f']*params['g2'],
                        params['f']*params['g2'], params['g2']*f2b1]
            params_G2 = [params['f'], f2b1, f2b1, f3b1sq]
            params_mixed = [params['f'], f2b1, 2*f2b1, f3b1sq, 2*f3b1sq,
                            f3b1sq*params['f']/params['b1']]

            tuples_list_1 = [(0,0,0),(2,0,0),(0,2,0),(2,2,0)]
            tuples_list_2 = [(0,0,2),(2,0,2),(0,2,2),(2,2,2)]
            tuples_list_k31 = [(1,0,1),(3,0,1),(1,2,1),(1,4,1),(3,2,1),(3,4,1)]
            tuples_list_k32 = [(0,1,1),(0,3,1),(2,1,1),(4,1,1),(2,3,1),(4,3,1)]

            kernel_F2 = 2*b1sq * self.join_kernel_mu123_integral(
                'F2', tuples_list_1, neff, params_F2,
                params['alpha_tr'], params['alpha_lo'])
            kernel_b2 = params['b1'] * self.join_kernel_mu123_integral(
                'b2', tuples_list_1, neff, params_b2,
                params['alpha_tr'], params['alpha_lo'])
            kernel_K = 2*params['b1'] * self.join_kernel_mu123_integral(
                'K', tuples_list_1, neff, params_K,
                params['alpha_tr'], params['alpha_lo'])
            kernel_G2 = 2*b1sq * self.join_kernel_mu123_integral(
                'G2', tuples_list_2, neff, params_G2,
                params['alpha_tr'], params['alpha_lo'])
            kernel_k31 = - self.join_kernel_mu123_integral(
                'k31', tuples_list_k31, neff, params_mixed,
                params['alpha_tr'], params['alpha_lo'])
            kernel_k32 = - self.join_kernel_mu123_integral(
                'k32', tuples_list_k32, neff, params_mixed,
                params['alpha_tr'], params['alpha_lo'])

            kernel = kernel_F2 + kernel_b2 + kernel_K + kernel_G2 \
                     + kernel_k31 + kernel_k32
            kernel_stoch = 1.0/self.nbar * (b1sq + params['b1']*params['f'] \
                           * self.I[2,0,0][0,0] + params['f']**2 \
                           * self.I[4,0,0][0,0])

        P2 = PL_dw[self.ki]*PL_dw[self.kj]

        B_SPT = np.einsum("ij,ij->i", kernel, P2[self.tri_to_id_sq])
        B_stoch = params['MB0'] * np.sum(PL_dw[self.tri_to_id],axis=1) \
                  * kernel_stoch + params['NB0']/self.nbar**2

        Bell = {}
        Bell['ell0'] = B_SPT + B_stoch

        return Bell
