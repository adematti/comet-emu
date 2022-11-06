"""Grid module."""

import numpy as np

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
            self.weights = self.weights_all
            self.nmodes = self.nmodes_all
            self.keff_all = np.zeros(len(self.nmodes_all)-1)
        elif kbin.size != self.kbin.size:
            ids = np.intersect1d(kbin, self.kbin, return_indices=True)[2]
            self.k = None
            self.mu = None
            self.weights = None
            self.nmodes = [0]

            for i in ids:
                n1 = self.nmodes_all[i]
                n2 = self.nmodes_all[i+1]
                self.k = np.hstack((self.k, self.k_all[n1:n2])) \
                    if self.k is not None else self.k_all[n1:n2]
                self.mu = np.hstack((self.mu, self.mu_all[n1:n2])) \
                    if self.mu is not None else self.mu_all[n1:n2]
                self.weights = np.hstack((self.weights,
                                          self.weights_all[n1:n2])) \
                    if self.weights is not None else self.weights_all[n1:n2]
                self.nmodes.append(self.nmodes[-1] + self.nmodes_all[i+1]
                                   - self.nmodes_all[i])
        else:
            self.k = self.k_all
            self.mu = self.mu_all
            self.weights = self.weights_all
            self.nmodes = self.nmodes_all

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
