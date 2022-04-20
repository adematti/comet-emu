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
        if 'bins' in kwargs:
            self.bins = kwargs.get('bins')
        if 'signal' in kwargs:
            self.signal = kwargs.get('signal')
            self.n_ell = self.signal.shape[1]
        if 'cov' in kwargs:
            self.cov = kwargs.get('cov')
        if 'bins_mixing_matrix' in kwargs:
            self.bins_mixing_matrix = kwargs.get('bins_mixing_matrix')
        if 'W_mixing_matrix' in kwargs:
            self.W_mixing_matrix = kwargs.get('W_mixing_matrix')
        if 'theory_cov' in kwargs:
            self.theory_cov = kwargs.get('theory_cov')
        else:
            self.theory_cov = True

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
        if 'bins' in kwargs:
            self.bins = kwargs.get('bins')
        if 'signal' in kwargs:
            self.signal = kwargs.get('signal')
            self.n_ell = self.signal.shape[1]
        if 'cov' in kwargs:
            self.cov = kwargs.get('cov')
        if 'bins_mixing_matrix' in kwargs:
            self.bins_mixing_matrix = kwargs.get('bins_mixing_matrix')
        if 'W_mixing_matrix' in kwargs:
            self.W_mixing_matrix = kwargs.get('W_mixing_matrix')
        if 'theory_cov' in kwargs:
            self.theory_cov = kwargs.get('theory_cov')
        else:
            self.theory_cov = True

        if not self.theory_cov:
            if 'n_realizations' in kwargs:
                self.n_realizations = kwargs.get('n_realizations')
            else:
                raise ValueError("For non-analytical covariance matrix, "
                                 "'n_realizations' needs to be specified.")

        # update kmax-truncated data containers
        if self.kmax_is_set:
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
        if not isinstance(kmax, list):
            self.kmax = [kmax for i in range(self.n_ell)]
        else:
            self.kmax = kmax

        nbin_total = self.bins.shape[0]

        self.nbins = [0 for i in range(self.n_ell)]
        for ell in range(self.n_ell):
            for i in range(nbin_total):
                if self.bins[i] < self.kmax[ell]:
                    self.nbins[ell] += 1
                else:
                    break

        self.bins_kmax = []
        self.signal_kmax = np.array([])
        for ell in range(self.n_ell):
            self.bins_kmax.append(self.bins[:self.nbins[ell]])
            self.signal_kmax = np.concatenate(
                (self.signal_kmax, self.signal[:self.nbins[ell], ell])) \
                if self.signal_kmax.size else self.signal[:self.nbins[ell],
                                                          ell]

        self.cov_kmax = np.zeros([sum(self.nbins), sum(self.nbins)])
        for ell1 in range(self.n_ell):
            for ell2 in range(self.n_ell):
                self.cov_kmax[
                    sum(self.nbins[:ell1]):sum(self.nbins[:ell1+1]),
                    sum(self.nbins[:ell2]):sum(self.nbins[:ell2+1])] = \
                    self.cov[ell1*nbin_total:ell1*nbin_total+self.nbins[ell1],
                             ell2*nbin_total:ell2*nbin_total+self.nbins[ell2]]
        self.inverse_cov_kmax = np.linalg.inv(self.cov_kmax)
        self.inverse_cov_kmax *= self.AHfactor(sum(self.nbins))

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
