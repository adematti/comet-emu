"""Grid module."""

import numpy as np
import ctypes
import numba as nb
import os

nb.config.THREADING_LAYER = 'workqueue'

class Grid:

    def __init__(self, kfun, dk):
        self.kfun = kfun
        self.dk = dk
        self.kbin = None
        self.tri_unique = None
        self.tri_to_id = None
        self.keff_all = None
        self.do_rounding = True
        self.decimals = None
        self.nbin_rounding = None

    def update(self, kfun, dk):
        if self.kfun != kfun or self.dk != dk:
            self.kfun = kfun
            self.dk = dk
            self.kbin = None
            self.tri_unique = None
            self.tri_to_id = None
            self.keff_all = None
            self.do_rounding = True
            self.decimals = None
            self.nbin_rounding = None

    def find_discrete_modes(self, kbin, **kwargs):
        def id_to_mode(ii):
            mode = np.copy(ii)
            mode[ii > self.N/2] -= self.N
            return mode

        do_rounding = kwargs.get('do_rounding', True)
        decimals = kwargs.get('decimals', [1,3])

        if self.kbin is None or not np.all(np.isin(kbin, self.kbin)) \
                or self.do_rounding != do_rounding or self.decimals != decimals:
            self.kbin = kbin
            self.N = 2*int(np.ceil((self.kbin[-1]+self.dk/2)/self.kfun))
            self.do_rounding = do_rounding
            self.decimals = decimals

            ii = np.indices((self.N, self.N, self.N))
            kk = id_to_mode(ii)
            k2 = np.zeros((self.N, self.N, self.N))
            for d in range(3):
                k2 += kk[d]**2
            kmag = np.sqrt(k2)*self.kfun

            self.k_all = None
            self.mu_all = None
            self.weights_all = None
            self.nmodes_all = [0]

            for i in range(self.kbin.size):
                modes = np.where(np.abs(kmag - self.kbin[i]) < self.dk/2)
                kmu = np.zeros([modes[0].shape[0],2])
                for j in range(3):
                    kmu[:,0] += kk[j][modes]**2
                kmu[:,0] = np.sqrt(kmu[:,0])*self.kfun
                kmu[:,1] = np.abs(kk[2][modes]*self.kfun/kmu[:,0])

                if do_rounding:
                    kmu[:,0] = np.around(kmu[:,0]/self.dk,
                                         decimals=decimals[0]) * self.dk
                    kmu[:,1] = np.around(kmu[:,1], decimals=decimals[1])

                kmu, weights = np.unique(kmu, axis=0, return_counts=True)
                self.k_all = np.hstack((self.k_all, kmu[:,0])) \
                    if self.k_all is not None else kmu[:,0]
                self.mu_all = np.hstack((self.mu_all, kmu[:,1])) \
                    if self.mu_all is not None else kmu[:,1]
                self.weights_all = np.hstack((self.weights_all, weights)) \
                    if self.weights_all is not None else weights
                self.nmodes_all.append(self.nmodes_all[-1] + kmu.shape[0])

            self.k = self.k_all
            self.mu = self.mu_all
            self.mu2_all = self.mu_all**2
            self.mu2 = self.mu2_all
            self.weights = self.weights_all
            self.nmodes = self.nmodes_all
            self.keff_all = np.zeros(len(self.nmodes_all)-1)
            self.weights_sum = np.add.reduceat(self.weights, self.nmodes[:-1])
        elif kbin.size != self.kbin.size:
            ids = np.intersect1d(kbin, self.kbin, return_indices=True)[2]
            self.k = None
            self.mu = None
            self.mu2 = None
            self.weights = None
            self.nmodes = [0]

            for i in ids:
                n1 = self.nmodes_all[i]
                n2 = self.nmodes_all[i+1]
                self.k = np.hstack((self.k, self.k_all[n1:n2])) \
                    if self.k is not None else self.k_all[n1:n2]
                self.mu = np.hstack((self.mu, self.mu_all[n1:n2])) \
                    if self.mu is not None else self.mu_all[n1:n2]
                self.mu2 = np.hstack((self.mu2, self.mu2_all[n1:n2])) \
                    if self.mu2 is not None else self.mu2_all[n1:n2]
                self.weights = np.hstack((self.weights,
                                          self.weights_all[n1:n2])) \
                    if self.weights is not None else self.weights_all[n1:n2]
                self.nmodes.append(self.nmodes[-1] + self.nmodes_all[i+1]
                                   - self.nmodes_all[i])
            self.weights_sum = np.add.reduceat(self.weights, self.nmodes[:-1])
        else:
            self.k = self.k_all
            self.mu = self.mu_all
            self.mu2 = self.mu2_all
            self.weights = self.weights_all
            self.nmodes = self.nmodes_all
            self.weights_sum = np.add.reduceat(self.weights, self.nmodes[:-1])

    def find_discrete_triangles(self, tri_unique, tri_to_id, **kwargs):
        def id_to_mode(ii):
            mode = np.copy(ii)
            mode[ii > self.N/2] -= self.N
            return mode

        do_rounding = kwargs.get('do_rounding', True)
        decimals = kwargs.get('decimals', [2,0,0])
        nbin_rounding = kwargs.get('nbin_rounding', 3)

        if self.tri_unique is None \
                or not np.all(np.isin(tri_unique, self.tri_unique)) \
                or not np.all(np.isin(tri_to_id, self.tri_to_id)) \
                or self.do_rounding != do_rounding \
                or self.decimals != decimals \
                or self.nbin_rounding != nbin_rounding:
            print('start finding triangles')
            self.tri_unique = tri_unique
            self.tri_to_id = tri_to_id
            self.N = 2*int(np.ceil((self.tri_unique[-1]+self.dk/2)/self.kfun))
            self.do_rounding = do_rounding
            self.decimals = decimals
            self.nbin_rounding = nbin_rounding

            ii = np.indices((self.N, self.N, self.N))
            kk = id_to_mode(ii)
            ksq = np.zeros((self.N, self.N, self.N))
            for d in range(3):
                ksq += kk[d]**2
            kmag = np.sqrt(ksq)*self.kfun

            ksph_shell = []
            weights_shell = []
            kk_shell = [[],[],[]]
            # dtheta = np.pi/self.nbin_rounding
            dmu = 1./self.nbin_rounding
            dphi = 2*np.pi/self.nbin_rounding

            for i,k in enumerate(self.tri_unique):
                modes = np.where(np.abs(kmag - k) < self.dk/2)
                kmag_shell = np.round(kmag[modes]/self.dk,
                    decimals=self.decimals[0])*self.dk
                #theta_shell = np.round(
                #    np.arccos(kk[2][modes]*self.kfun/kmag_shell)/dtheta,
                #    decimals=self.decimals[1])*dtheta
                mu_shell = np.round(
                    kk[2][modes]*self.kfun/kmag_shell/dmu,
                    decimals=self.decimals[1])*dmu
                phi_shell = np.round(np.arctan2(kk[0][modes],kk[1][modes])/dphi,
                    decimals=self.decimals[2])*dphi
                #t = np.array([kmag_shell, theta_shell, phi_shell]).T
                t = np.array([kmag_shell, mu_shell, phi_shell]).T
                t, ids_shell, weights = np.unique(t, axis=0, return_index=True,
                                                  return_counts=True)
                ksph_shell.append(t)
                weights_shell.append(weights)
                for d in range(3):
                    kk_shell[d].append(kk[d][modes][ids_shell])

            k123_all = []
            mu123_all = []
            weights_all = []
            self.ntri_all = [0]

            i1_old, i2_old = [None]*2
            for i1, i2, i3 in self.tri_to_id:
                if i1_old != i1 or i2_old != i2:
                    kk_shell_12_z = np.add.outer(kk_shell[2][i1],
                                                 kk_shell[2][i2])
                    ksq = kk_shell_12_z**2
                    for d in range(2):
                        ksq += np.add.outer(kk_shell[d][i1], kk_shell[d][i2])**2
                    kmag_12 = np.sqrt(ksq)*self.kfun
                    weights_12 = np.multiply.outer(weights_shell[i1],
                                                   weights_shell[i2])
                    i1_old = i1
                    i2_old = i2
                modes = np.where(np.abs(kmag_12 - self.tri_unique[i3])
                                 < self.dk/2)
                weights = weights_12[modes]
                k1 = ksph_shell[i1][modes[0],0]
                k2 = ksph_shell[i2][modes[1],0]
                k3 = kmag_12[modes]
                # mu1 = np.cos(ksph_shell[i1][modes[0],1])
                # mu2 = np.cos(ksph_shell[i2][modes[1],1])
                # theta3 = np.round(
                #     np.arccos(kk_shell_12_z[modes]*self.kfun/k3)/dtheta,
                #     decimals=self.decimals[1])*dtheta
                # mu3 = np.cos(theta3)
                mu1 = ksph_shell[i1][modes[0],1]
                mu2 = ksph_shell[i2][modes[1],1]
                mu3 = -kk_shell_12_z[modes]*self.kfun/k3
                mu3 = np.round(mu3/dmu, decimals=self.decimals[1])*dmu
                k3 = np.round(k3/self.dk, decimals=self.decimals[0])*self.dk
                kmu = np.array([k1,k2,k3,mu1,mu2,mu3]).T
                kmu[np.where(kmu[:,3] < 0),3:] *= -1.0
                kmu_unique, ids_inv, w = np.unique(kmu, axis=0,
                                                   return_inverse=True,
                                                   return_counts=True)
                ids = np.split(np.argsort(ids_inv), np.cumsum(w[:-1]))
                weights_unique = list(map(lambda x: weights[x].sum(), ids))
                k123_all.append(kmu_unique[:,:3])
                mu123_all.append(kmu_unique[:,3:])
                weights_all.append(weights_unique)
                self.ntri_all.append(self.ntri_all[-1] + kmu_unique.shape[0])

            self.k123_all = np.zeros([self.ntri_all[-1],3])
            self.mu123_all = np.zeros([self.ntri_all[-1],3])
            self.weights_all = np.zeros(self.ntri_all[-1])
            for i in range(self.tri_to_id.shape[0]):
                n1 = self.ntri_all[i]
                n2 = self.ntri_all[i+1]
                self.k123_all[n1:n2] = k123_all[i]
                self.mu123_all[n1:n2] = mu123_all[i]
                self.weights_all[n1:n2] = weights_all[i]

            self.k123 = self.k123_all
            self.mu123 = self.mu123_all
            self.weights = self.weights_all
            self.ntri = self.ntri_all
            self.k123eff_all = np.zeros([len(self.ntri_all)-1,3])
            print('done')
        else:
            self.k123 = self.k123_all
            self.mu123 = self.mu123_all
            self.weights = self.weights_all
            self.ntri = self.ntri_all

    def compute_effective_modes(self, kbin, **kwargs):
        self.find_discrete_modes(kbin, **kwargs)
        if np.all(self.keff_all == 0):
            for i in range(self.keff_all.size):
                n1 = self.nmodes[i]
                n2 = self.nmodes[i+1]
                self.keff_all[i] = np.average(self.k[n1:n2],
                                              weights=self.weights[n1:n2])
                self.keff = self.keff_all
        elif kbin.size != self.kbin.size:
            ids = np.intersect1d(kbin, self.kbin, return_indices=True)[2]
            self.keff = self.keff_all[ids]
        else:
            self.keff = self.keff_all

    def compute_effective_triangles(self, tri_unique, tri_to_id, **kwargs):
        self.find_discrete_triangles(tri_unique, tri_to_id, **kwargs)
        if np.all(self.k123eff_all == 0):
            for i in range(self.k123eff_all.shape[0]):
                n1 = self.ntri[i]
                n2 = self.ntri[i+1]
                self.k123eff_all[i] = np.average(self.k123[n1:n2], axis=0,
                                                 weights=self.weights[n1:n2])
                self.k123eff = self.k123eff_all
        else:
            self.k123eff = self.k123eff_all



class CtypedGrid:

    try:
        lib = ctypes.cdll.LoadLibrary('{}/discreteness/libgrid.so'.format(
            os.path.join(os.path.dirname(__file__))))
        lib.new_double_vector.restype = ctypes.c_void_p
        lib.new_double_vector.argtypes = []
        lib.delete_double_vector.restype = None
        lib.delete_double_vector.argtypes = [ctypes.c_void_p]
        lib.get_double_vector_size.restype = ctypes.c_int
        lib.get_double_vector_size.argtypes = [ctypes.c_void_p]
        lib.push_back_double_vector.restype = None
        lib.push_back_double_vector.argtypes = [ctypes.c_void_p,
                                                ctypes.c_double]

        lib.new_Grid.restype = ctypes.c_void_p
        lib.new_Grid.argtypes = [ctypes.c_int, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double,
                                 ctypes.c_double, ctypes.c_double]
        lib.find_unique_triangles.restype = None
        lib.find_unique_triangles.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_double]
        lib.get_num_triangle_bins.restype = ctypes.c_int
        lib.get_num_triangle_bins.argtypes = [ctypes.c_void_p]
        lib.get_num_fundamental_triangles.restype = ctypes.c_int
        lib.get_num_fundamental_triangles.argtypes = [ctypes.c_void_p]
        lib.get_unique_triangles.restype = None
        lib.get_unique_triangles.argtypes = [ctypes.c_void_p,
            np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
    except:
        print('Warning! "libgrid.so" not found, bispectrum binning options '
              'will not be available.')

    def __init__(self, **kwargs):
        self.kfun = kwargs.get('kfun')
        self.dk = kwargs.get('dk')
        self.kbin = None
        self.num_grid = 0
        self.do_rounding = kwargs.get('do_rounding', True)
        self.decimals = kwargs.get('decimals', [3,3])
        self.shape_limits = kwargs.get('shape_limits', [0.999,1.15])
        self.c_grid = None
        self.kmu123 = None
        if self.do_rounding:
            self.roundk = 10**(-self.decimals[0])*self.dk
            self.roundmu = 10**(-self.decimals[1])
        else:
            self.roundk = 1e-7*self.dk
            self.roundmu = 1e-7

    def update(self, **kwargs):
        if self.kfun != kwargs.get('kfun') or self.dk != kwargs.get('dk') \
                or self.do_rounding != kwargs.get('do_rounding', True) \
                or self.decimals != kwargs.get('decimals', [3,3]) \
                or self.shape_limits != kwargs.get('shape_limits',[0.999,1.15]):
            self.kfun = kwargs.get('kfun')
            self.dk = kwargs.get('dk')
            self.do_rounding = kwargs.get('do_rounding', True)
            self.decimals = kwargs.get('decimals', [3,3])
            self.shape_limits = kwargs.get('shape_limits', [0.999,1.15])
            self.c_grid = None
            self.kmu123 = None
            if self.do_rounding:
                self.roundk = 10**(-self.decimals[0])*self.dk
                self.roundmu = 10**(-self.decimals[1])
            else:
                self.roundk = 1e-7*self.dk
                self.roundmu = 1e-7

    def find_discrete_triangles(self, tri_unique):
        self.tri_unique = tri_unique

        num_grid = 2*int(np.ceil(2*(np.amax(self.tri_unique)+self.dk/2)
                               / self.kfun))
        if num_grid > self.num_grid or self.c_grid is None:
            self.num_grid = num_grid
            self.c_grid = CtypedGrid.lib.new_Grid(self.num_grid, self.kfun,
                                                  self.roundk, self.roundmu,
                                                  self.shape_limits[0],
                                                  self.shape_limits[1])

        self.vec_kbin = CtypedGrid.lib.new_double_vector()
        for i in range(len(self.tri_unique)):
            CtypedGrid.lib.push_back_double_vector(self.vec_kbin,
                                                   self.tri_unique[i])

        CtypedGrid.lib.find_unique_triangles(self.c_grid, self.vec_kbin,
                                             self.dk)

        self.ntri = CtypedGrid.lib.get_num_triangle_bins(self.c_grid)
        self.size = CtypedGrid.lib.get_num_fundamental_triangles(self.c_grid)

        temp = np.empty(self.size*6, dtype=np.float64)
        self.weights = np.empty(self.size, dtype=np.int32)
        self.num_tri_f = np.empty(self.ntri, dtype=np.int32)
        CtypedGrid.lib.get_unique_triangles(self.c_grid, temp,
                                            self.weights, self.num_tri_f)

        self.cum_num_tri_f = np.concatenate(([0],np.cumsum(self.num_tri_f)))
        self.kmu123 = np.empty((self.size,6), dtype=np.float64)
        for i in range(self.ntri):
            self.kmu123[self.cum_num_tri_f[i]:self.cum_num_tri_f[i+1]] = \
                np.array(temp[self.cum_num_tri_f[i]*6:
                              self.cum_num_tri_f[i+1]*6]).reshape(
                                  (self.num_tri_f[i],6))
        # self.kmu123 = [
        #     np.array(self.kmu123[cum_num_tri_f[i]*6:cum_num_tri_f[i+1]*6]).
        #     reshape((self.num_tri_f[i],6)) for i in range(self.ntri)]
        # self.weights = [self.weights[cum_num_tri_f[i]:cum_num_tri_f[i+1]]
        #                 for i in range(self.ntri)]

    @staticmethod
    @nb.njit(parallel=True)
    def _average_k123(kmu123, weights, cum_num_tri_f):
        nconf = cum_num_tri_f.size - 1
        keff = np.zeros((nconf,3))
        for i in nb.prange(nconf):
            n1 = cum_num_tri_f[i]
            n2 = cum_num_tri_f[i+1]
            wsum = np.sum(weights[n1:n2])
            for d in nb.prange(3):
                keff[i,d] = np.sum(kmu123[n1:n2,d]*weights[n1:n2])/wsum
        return keff

    def compute_effective_triangles(self, tri_unique):
        if np.any(tri_unique != self.tri_unique) or self.kmu123 is None:
            self.find_discrete_triangles(tri_unique)
        self.k123eff = self._average_k123(self.kmu123, self.weights,
                                          self.cum_num_tri_f)
