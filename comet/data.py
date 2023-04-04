"""Data module."""

import numpy as np


class MeasuredData:
    r"""Class for handling data structures (data vectors and covariances)."""

    def __init__(self, **kwargs):
        r"""Class constructor.

        Keywords argument `kwargs` are summarised in the **Parameters** section
        below. An instance of the class can be created without specifying any
        of the keyword arguments, and subsequently updated with the **update**
        method.

        Parameters
        ----------
        bins: numpy.ndarray, optional
            Array containing the sampled wavemodes :math:`k`.
        signal: numpy.ndarray, optional
            Array containing the power spectrum multipoles :math:`P_{\ell}(k)`.
            If more than one multipole is provided, the first and second index
            of the array runs over the different :math:`k` and the different
            multipole :math:`\ell`, respectively.
        cov: numpy.ndarray, optional
            Covariance matrix corresponding to **signal**. The covariance is in
            the form of a 2d array, whose blocks correspond to the auto-
            (along the diagonal) and cross-covariances of the multipoles.
        theory_cov: bool, optional
            Flag to determine if the provided covariance matrix is theoretical
            (**True**) or estimated from simulations (**False**). Defaults to
            **True**.
        n_realizations: int, optional
            If the provided covariance is estimated from numerical simulations,
            this parameter is required, and specifies the total number of
            resamplings (needed for Hartlap corrections).
        """
        if 'stat' in kwargs and \
            kwargs.get('stat') in ['powerspectrum', 'bispectrum']:
                self.stat = kwargs.get('stat')
        if 'bins' in kwargs:
            self.bins = kwargs.get('bins')
            if 'stat' not in kwargs and self.bins.ndim == 1:
                self.stat = 'powerspectrum'
            elif 'stat' not in kwargs and self.bins.shape[1] == 3:
                self.stat = 'bispectrum'
            else:
                print('Warning! Type of statistic not recognised.')
                self.stat = 'unknown'
        if 'signal' in kwargs:
            self.signal = kwargs.get('signal')
            if self.signal.ndim == 1:
                self.signal = self.signal[:,None]
            if 'ell' in kwargs:
                ell = kwargs.get('ell')
                self.ell = [ell] if not isinstance(ell, list) else ell
                self.n_ell = len(self.ell)
            else:
                self.n_ell = self.signal.shape[1]
                self.ell = [2*n for n in range(self.n_ell)]
        if 'cov' in kwargs:
            self.cov = kwargs.get('cov')
            if self.cov.ndim == 1:
                self.cov = np.diag(self.cov)
                self.cov_is_block_diagonal = True
            else:
                self.cov_is_block_diagonal = self.is_block_diagonal(self.cov,
                                                                    self.n_ell)
        if 'bins_mixing_matrix' in kwargs:
            self.bins_mixing_matrix = kwargs.get('bins_mixing_matrix')
            self.bins_mixing_matrix_compressed = np.logspace(
                np.log10(self.bins_mixing_matrix[1][0]),
                np.log10(self.bins_mixing_matrix[1][-1]),
                int(self.bins_mixing_matrix[1][-1]/0.5*100))
        if 'W_mixing_matrix' in kwargs:
            self.W_mixing_matrix = kwargs.get('W_mixing_matrix')
        if 'theory_cov' in kwargs:
            self.theory_cov = kwargs.get('theory_cov')
        else:
            self.theory_cov = True

        if self.stat == 'bispectrum':
            if 'kfun' in kwargs:
                self.kfun = kwargs.get('kfun')
            else:
                try:
                    self.kfun = self.bins[0,0]
                    print('kfun not specified. Using kfun = {}'.format(
                        self.kfun))
                except NameError:
                    self.kfun = 1.0
                    print('Warning. Neither kfun nor bins specified,' +
                          'setting kfun = 1.0')

        if not self.theory_cov:
            if 'n_realizations' in kwargs:
                self.n_realizations = kwargs.get('n_realizations')
            else:
                raise ValueError("For non-analytical covariance matrix, "
                                 "'n_realizations needs' to be specified.")

        self.kmax_is_set = False

    def update(self, **kwargs):
        r"""Update the class with new data.

        Keywords argument `kwargs` are summarised in the **Parameters** section
        below.

        Parameters
        ----------
        bins: numpy.ndarray, optional
            Array containing the sampled wavemodes :math:`k`.
        signal: numpy.ndarray, optional
            Array containing the power spectrum multipoles :math:`P_{\ell}(k)`.
            If more than one multipole is provided, the first and second index
            of the array runs over the different :math:`k` and the different
            multipole :math:`\ell`, respectively.
        cov: numpy.ndarray, optional
            Covariance matrix corresponding to **signal**. The covariance is in
            the form of a 2d array, whose blocks correspond to the auto-
            (along the diagonal) and cross-covariances of the multipoles.
        theory_cov: bool, optional
            Flag to determine if the provided covariance matrix is theoretical
            (**True**) or estimated from simulations (**False**). Defaults to
            **True**.
        n_realizations: int, optional
            If the provided covariance is estimated from numerical simulations,
            this parameter is required, and specifies the total number of
            resamplings (needed for Hartlap corrections).
        """
        if 'stat' in kwargs and \
            kwargs.get('stat') in ['powerspectrum', 'bispectrum']:
                self.stat = kwargs.get('stat')
        if 'bins' in kwargs:
            self.bins = kwargs.get('bins')
            if 'stat' not in kwargs and self.bins.ndim == 1:
                self.stat = 'powerspectrum'
            elif 'stat' not in kwargs and self.bins.shape[1] == 3:
                self.stat = 'bispectrum'
            else:
                print('Warning! Type of statistic not recognised.')
                self.stat = 'unknown'
        if 'signal' in kwargs:
            self.signal = kwargs.get('signal')
            if self.signal.ndim == 1:
                self.signal = self.signal[:,None]
            if 'ell' in kwargs:
                ell = kwargs.get('ell')
                self.ell = [ell] if not isinstance(ell, list) else ell
                self.n_ell = len(self.ell)
            else:
                self.n_ell = self.signal.shape[1]
                self.ell = [2*n for n in range(self.n_ell)]
        if 'cov' in kwargs:
            self.cov = kwargs.get('cov')
            if self.cov.ndim == 1:
                self.cov = np.diag(self.cov)
                self.cov_is_block_diagonal = True
            else:
                self.cov_is_block_diagonal = self.is_block_diagonal(self.cov,
                                                                    self.n_ell)
        if 'bins_mixing_matrix' in kwargs:
            self.bins_mixing_matrix = kwargs.get('bins_mixing_matrix')
            self.bins_mixing_matrix_compressed = np.logspace(
                np.log10(self.bins_mixing_matrix[1][0]),
                np.log10(self.bins_mixing_matrix[1][-1]),
                int(self.bins_mixing_matrix[1][-1]/0.5*100))
        if 'W_mixing_matrix' in kwargs:
            self.W_mixing_matrix = kwargs.get('W_mixing_matrix')
        if 'theory_cov' in kwargs:
            self.theory_cov = kwargs.get('theory_cov')
        if 'kfun' in kwargs:
            self.kfun = kwargs.get('kfun')

        if not self.theory_cov:
            try:
                self.n_realizations
            except:
                if 'n_realizations' in kwargs:
                    self.n_realizations = kwargs.get('n_realizations')
                else:
                    raise ValueError("For non-analytical covariance matrix, "
                                     "'n_realizations' needs to be specified.")

        # update kmax-truncated data containers
        if self.kmax_is_set:
            if len(self.kmax) != self.n_ell:
                kmax_copy = self.kmax.copy()
                self.kmax = []
                for n in range(self.n_ell):
                    self.kmax.append(kmax_copy[n] if n < len(kmax_copy) else 0)
            self.set_kmax(self.kmax)

    def clear_data(self):
        r"""Clear the data.

        Sets to **None** the class attributes.
        """
        self.bins = None
        self.signal = None
        self.cov = None
        self.bins_mixing_matrix = None
        self.W_mixing_matrix = None
        self.kmax_is_set = False

    def is_block_diagonal(self, arr, nblock):
        def is_diagonal(arr):
            return np.all(arr == np.diag(np.diag(arr)))

        nbin_per_block = int(arr.shape[0]/nblock)
        check = np.zeros((nblock,nblock), dtype=bool)
        for i in range(nblock):
            for j in range(nblock):
                check[i,j] = is_diagonal(
                    arr[i*nbin_per_block:(i+1)*nbin_per_block,
                        j*nbin_per_block:(j+1)*nbin_per_block])
        return np.all(check)

    def set_kmax(self, kmax):
        r"""Set the maximum mode used in the computation of the :math:`\chi^2`.

        Sets the class attribute corresponding to the maximum wavemode
        :math:`k_\mathrm{max}` to be used in the calculation of the
        :math:`\chi^2`. In addition, it applies the corresponding scale cuts
        to both data vectors and covariance matrix.

        Parameters
        ----------
        kmax: float or list
            Maximum wavemode :math:`k_\mathrm{max}` for the individual
            multipoles. If a **float** is provided, :math:`k_\mathrm{max}` is
            assumed to be the same for all the multipoles selected by the user.
            On the contrary, each entry of the **list** object is specific
            for a given multipole.
        """
        def isdiagonal(arr):
            return np.all(arr == np.diag(np.diag(arr)))

        if not isinstance(kmax, list):
            self.kmax = [kmax for i in range(self.n_ell)]
        else:
            self.kmax = kmax

        nbin_total = self.bins.shape[0]

        self.nbins = [0 for i in range(self.n_ell)]
        ids_kmax = []
        for ell in range(self.n_ell):
            ids_kmax.append(np.where(
                np.all(self.bins[:,None] < self.kmax[ell], axis=-1))[0])
            self.nbins[ell] = len(ids_kmax[ell])
            # for i in range(nbin_total):
            #     if np.all(self.bins[i] < self.kmax[ell]):
            #         self.nbins[ell] += 1
            #     else:
            #         break

        self.bins_kmax = []
        self.signal_kmax = np.array([])
        for ell in range(self.n_ell):
            self.bins_kmax.append(self.bins[ids_kmax[ell]])
            self.signal_kmax = np.concatenate(
                (self.signal_kmax, self.signal[ids_kmax[ell], ell])) \
                if self.signal_kmax.size else self.signal[ids_kmax[ell],
                                                          ell]

        self.cov_kmax = np.zeros([sum(self.nbins), sum(self.nbins)])
        for ell1 in range(self.n_ell):
            for ell2 in range(self.n_ell):
                self.cov_kmax[
                    sum(self.nbins[:ell1]):sum(self.nbins[:ell1+1]),
                    sum(self.nbins[:ell2]):sum(self.nbins[:ell2+1])] = \
                    self.cov[tuple(
                        np.meshgrid(ell1*nbin_total + ids_kmax[ell1],
                                    ell2*nbin_total + ids_kmax[ell2],
                                    indexing='ij'))]
        self.inverse_cov_kmax = np.linalg.inv(self.cov_kmax)
        if not self.theory_cov:
            self.inverse_cov_kmax *= self.AHfactor(sum(self.nbins))

        if self.stat == 'bispectrum':
            self.inverse_cov_kmax_cholesky = np.linalg.cholesky(
                self.inverse_cov_kmax).T
            if self.cov_is_block_diagonal:
                tri_dtype = {'names':['f{}'.format(i) for i in range(3)],
                             'formats':3 * [self.bins.dtype]}
                self.cholesky_diag = {}
                self.tri_id_ell2_in_ell1 = {}
                self.tri_id_ell1_in_ell2 = {}
                for i in range(self.n_ell):
                    ni1 = sum(self.nbins[:i])
                    ni2 = sum(self.nbins[:i+1])
                    for j in range(i,self.n_ell):
                        nj1 = sum(self.nbins[:j])
                        nj2 = sum(self.nbins[:j+1])
                        id1, id2 = np.sort(np.intersect1d(
                            self.bins_kmax[i].view(tri_dtype),
                            self.bins_kmax[j].view(tri_dtype),
                            return_indices=True)[1:])
                        self.tri_id_ell2_in_ell1[
                            'ell{}ell{}'.format(2*i,2*j)] = id1
                        self.tri_id_ell1_in_ell2[
                            'ell{}ell{}'.format(2*i,2*j)] = id2
                        self.cholesky_diag['ell{}ell{}'.format(2*i,2*j)] = \
                            np.diag(self.inverse_cov_kmax_cholesky[ni1:ni2,
                                nj1:nj2][id1[:,None],id2[None,:]])

        self.SN_kmax = (self.signal_kmax @ self.inverse_cov_kmax @
                        self.signal_kmax)

        self.kmax_is_set = True

    def AHfactor(self, nbins):
        r"""Compute corrections to the inverse covariance matrix.

        Depending on the value of the **theory_cov** attribute, returns the
        correction factor to the inverse covariance matrix (to take into
        account the limited number of independent realizations, in case the
        covariance is estimated from numerical simulations).

        Returns 1 if the class attribute **theory_cov** is set to **True**.
        Otherwise, returns

        .. math:: \frac{N_\mathrm{sim}-N_\mathrm{bins}-2}{N_\mathrm{sim}-1}\,,

        with :math:`N_\mathrm{sim}` and :math:`N_\mathrm{bins}` being the
        number of independent realizations from which the covariance has been
        estimated and the total number of bins at which it is sampled,
        respectively (see `Hartlap 2007
        <https://www.aanda.org/articles/aa/pdf/2007/10/aa6170-06.pdf>`_).

        Parameters
        ----------
        nbins: int
            Number of :math:`k` bins at which the power spectrum multipoles
            are sampled.

        Returns
        -------
        AHfactor: float
            Correcting factor to the inverse covariance matrix.
        """
        return 1.0 if self.theory_cov else (self.n_realizations-nbins-2.0) \
            * 1.0/(self.n_realizations - 1.0)

    def get_signal(self, ell, kmax=None):
        r"""Get power spectrum multipole from data sample.

        Returns the input data power spectrum multipole of order :math:`\ell`
        up to :math:`k_\mathrm{max}`.

        Parameters
        ----------
        ell: int
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`) and
            hexadecapole (:math:`\ell=4`).
        kmax: float, optional
            Maximum wavemode :math:`k_\mathrm{max}` up to which the multipole
            is provided. If **None**, the corresponding class attribute is
            used instead.

        Returns
        -------
        signal_kmax: numpy.ndarray
            Data power spectrum multipole of order :math:`\ell` up to
            :math:`k_\mathrm{max}`.
        """
        if kmax is None and not self.kmax_is_set:
            self.set_kmax(np.amax(self.bins))
        elif (kmax is not None and
              (not self.kmax_is_set or
               (self.kmax != kmax and
                self.kmax != [kmax for i in range(self.n_ell)]))):
            self.set_kmax(kmax)
        n = int(ell/2)
        return self.signal_kmax[sum(self.nbins[:n]):sum(self.nbins[:n+1])]

    def get_std(self, ell, kmax=None):
        r"""Get standard deviation of power spectrum multipole from sample.

        Returns the standard deviation of the input data power spectrum
        multipole of order :math:`\ell` up to :math:`k_\mathrm{max}`.

        Parameters
        ----------
        ell: int
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`) and
            hexadecapole (:math:`\ell=4`).
        kmax: float, optional
            Maximum wavemode  :math:`k_\mathrm{max}` up to which the standard
            deviation is provided. If `None`, the corresponding class
            attribute is used instead.

        Returns
        -------
        std_kmax: numpy.ndarray
            Standard deviation of the data power spectrum multipole of
            order :math:`\ell` up to :math:`k_\mathrm{max}`.
        """
        if kmax is None and not self.kmax_is_set:
            self.set_kmax(np.amax(self.bins))
        elif (kmax is not None and
              (not self.kmax_is_set or
               (self.kmax != kmax and
                self.kmax != [kmax for i in range(self.n_ell)]))):
            self.set_kmax(kmax)
        n = int(ell/2)
        var = np.diag(self.cov_kmax)[sum(self.nbins[:n]):sum(self.nbins[:n+1])]
        return np.sqrt(var)
