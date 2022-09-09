"""Grid module."""

import numpy as np

class Grid:

    def __init__(self, kf, dk):

        self.kf = kf
        self.dk = dk
        self.kbin = None
        self.keff_all = None
        self.do_rounding = True
        self.decimals = [1,3]

    def update(self, kf, dk):
        if self.kf != kf or self.dk != dk:
            self.kf = kf
            self.dk = dk
            self.kbin = None
            self.keff_all = None
            self.do_rounding = True
            self.decimals = [1,3]

    def find_discrete_modes(self, kbin, do_rounding=True, decimals=[1,3]):
        def id_to_mode(ii):
            mode = np.copy(ii)
            mode[ii > self.N/2] -= self.N
            return mode

        if self.kbin is None or not np.all(np.isin(kbin, self.kbin)) \
                or self.do_rounding != do_rounding or self.decimals != decimals:
            self.kbin = kbin
            self.N = 2*int(np.ceil((self.kbin[-1]+self.dk/2)/self.kf))
            self.do_rounding = do_rounding
            self.decimals = decimals

            ii = np.indices((self.N, self.N, self.N))
            kk = id_to_mode(ii)
            k2 = np.zeros((self.N, self.N, self.N))
            for d in range(3):
                k2 += kk[d]**2
            kmag = np.sqrt(k2)*self.kf

            self.k_all = None
            self.mu_all = None
            self.weights_all = None
            self.nmodes_all = [0]

            for i in range(self.kbin.size):
                modes = np.where(np.abs(kmag - self.kbin[i]) < self.dk/2)
                kmu = np.zeros([modes[0].shape[0],2])
                for j in range(3):
                    kmu[:,0] += kk[j][modes]**2
                kmu[:,0] = np.sqrt(kmu[:,0])*self.kf
                kmu[:,1] = np.abs(kk[2][modes]*self.kf/kmu[:,0])

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


    def compute_effective_modes(self, kbin, do_rounding=True, decimals=[1,3]):
        self.find_discrete_modes(kbin, do_rounding, decimals)
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
