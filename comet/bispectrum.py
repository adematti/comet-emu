"""Bispectrum module."""

import numpy as np
import numba as nb
import pickle
from comet.grid import Grid, CtypedGrid

nb.config.THREADING_LAYER = 'workqueue'

class Bispectrum:
    r"""Main class for the emulator of the bispectrum multipoles.
    """

    def __init__(self, real_space, model, use_Mpc):
        r"""Class constructor

        Parameters
        ----------
        real_space: bool
            Flag that determines if the model bispectrum is computed in real-
            (**False**) or redshift-space (**True**).
        use_Mpc: bool
            Flag that determines if the input and output quantities are
            specified in :math:`\mathrm{Mpc}` (**True**) or
            :math:`h^{-1}\mathrm{Mpc}` (**False**) units. Defaults to **True**.
        """
        self.real_space = real_space
        self.model = model
        self.use_Mpc = use_Mpc
        self.discrete_average = False
        self.use_effective_triangles = False
        self.nbar = 1.0 # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc
        self.tri = None
        self.cnlo_type = 'EggLeeSco'
        self.pow_ctr = 2.0
        self.binning = None
        self.binning_turned_on = False
        self.binning_turned_off = False
        self.last_eval_binned = None
        self.fiducial_Pdw = None
        self.fiducial_Pdw_sq = None
        self.fiducial_Pdw_eff = None
        self.fiducial_cosmology = {}
        self.num_fiducials = 1

        self.kernel_diagrams = {
            'F2':['B0L_b1b1b1', 'B0L_b1b1', 'B0L_b1b1', 'B0L_b1'],
            'b2':['B0L_b1b1b2', 'B0L_b1b2', 'B0L_b1b2', 'B0L_b2'],
            'K':['B0L_b1b1g2', 'B0L_b1g2', 'B0L_b1g2', 'B0L_g2'],
            'G2':['B0L_b1b1', 'B0L_b1', 'B0L_b1', 'B0L_id'],
            'k31':['B0L_b1b1b1', 'B0L_b1b1', 'B0L_b1b1', 'B0L_b1',
                   'B0L_b1', 'B0L_id'],
            'k32':['B0L_b1b1b1', 'B0L_b1b1', 'B0L_b1b1', 'B0L_b1',
                   'B0L_b1', 'B0L_id']
        }
        if 'EFT' in self.model:
            for kk in self.kernel_diagrams:
                temp = np.copy(self.kernel_diagrams[kk])
                for diagram in temp:
                    if diagram != 'B0L_id':
                        self.kernel_diagrams[kk].append(
                            '{}cnloB'.format(diagram))
                    else:
                        self.kernel_diagrams[kk].append('B0L_cnloB')

        if self.real_space:
            self.kernel_names = ['F2', 'b2', 'K']
            self.kernel_mu_tuples = {}
            for kk in self.kernel_names:
                self.kernel_mu_tuples[kk] = [(0,0,0)]
            self.discrete_kernel_mu_tuples = self.kernel_mu_tuples.copy()
            self.n123_tuples_stoch_all = np.array([[0,0,0]])
            self.discrete_stoch_kernel_mu_tuples = \
                {'id':self.n123_tuples_stoch_all}
        else:
            self.kernel_names = ['F2', 'G2', 'b2', 'K', 'k31', 'k32']
            kernel_names_deriv = []
            for kk in self.kernel_names:
                for i in range(3):
                    kk_deriv = 'd{}_dlnk{}'.format(kk, i+1)
                    kernel_names_deriv.append(kk_deriv)
            self.kernel_names += kernel_names_deriv
            if 'EFT' in self.model:
                kernel_names_ctr = []
                for kk in ['F2', 'G2', 'b2', 'K', 'k31', 'k32']:
                    for i in range(3):
                        kk_ctr = 'k{}sq{}'.format(i+1,kk)
                        kernel_names_ctr.append(kk_ctr)
                        for j in range(3):
                            kk_ctr = 'dk{}sq{}_dlnk{}'.format(i+1,kk,j+1)
                            kernel_names_ctr.append(kk_ctr)
                # kernel_names_ctr += ['k1sqb2', 'k2sqb2', 'k3sqb2']
                self.kernel_names += kernel_names_ctr

            self.kernel_mu_tuples = {}
            self.kernel_mu_tuples['F2'] = [(0,0,0), (2,0,0), (0,2,0), (2,2,0)]
            self.kernel_mu_tuples['G2'] = [(0,0,2), (2,0,2), (0,2,2), (2,2,2)]
            self.kernel_mu_tuples['b2'] = self.kernel_mu_tuples['F2']
            self.kernel_mu_tuples['K'] = self.kernel_mu_tuples['F2']
            self.kernel_mu_tuples['k31'] = [
                (1,0,1), (3,0,1), (1,2,1), (1,4,1), (3,2,1), (3,4,1)]
            self.kernel_mu_tuples['k32'] = [
                (0,1,1), (0,3,1), (2,1,1), (4,1,1), (2,3,1), (4,3,1)]
            self.n123_tuples_stoch_all = np.array([
                [0,0,0], [2,0,0], [0,2,0], [0,0,2], [4,0,0], [2,2,0],
                [2,0,2], [6,0,0], [4,2,0], [4,0,2]])
            self._get_mu_tuples_for_discrete_average()

        self.grid = None
        self.kernels = {}
        self.kernels_shell_average = {}
        self.stoch_kernels_shell_average = {}
        self.calibration_mode = False
        self.I = {}
        self.I_stoch = {}
        self.I_stoch_ctr = {}
        self.cov_mixing_kernel = {}

    def _get_mu_tuples_for_discrete_average(self):
        self.discrete_kernel_mu_tuples = {}
        for kk in self.kernel_mu_tuples:
            self.discrete_kernel_mu_tuples[kk] = \
                self.kernel_mu_tuples[kk].copy()
            for i in range(3):
                kk_deriv = 'd{}_dlnk{}'.format(kk, i+1)
                self.discrete_kernel_mu_tuples[kk_deriv] = \
                    self.discrete_kernel_mu_tuples[kk].copy()*2
                num_tup = int(len(self.discrete_kernel_mu_tuples[kk_deriv])/2)
                for j in range(num_tup,2*num_tup):
                    n123 = np.array(self.discrete_kernel_mu_tuples[kk_deriv][j])
                    n123[i] += 2
                    self.discrete_kernel_mu_tuples[kk_deriv][j] = tuple(n123)
                self.discrete_kernel_mu_tuples[kk_deriv] = list(set(
                    self.discrete_kernel_mu_tuples[kk_deriv]
                ))
        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            for kk in self.kernel_mu_tuples:
                for i in range(3):
                    kk_ctr = 'k{}sq{}'.format(i+1, kk)
                    self.discrete_kernel_mu_tuples[kk_ctr] = \
                        self.discrete_kernel_mu_tuples[kk].copy()
                    for j in range(len(self.discrete_kernel_mu_tuples[kk_ctr])):
                        n123 = np.array(
                            self.discrete_kernel_mu_tuples[kk_ctr][j])
                        n123[i] += 2
                        self.discrete_kernel_mu_tuples[kk_ctr][j] = tuple(n123)
                    if self.cnlo_type == 'IvaPhiNis':
                        ntuples = len(self.discrete_kernel_mu_tuples[kk_ctr])
                        for j in range(ntuples):
                            n123 = np.array(
                                self.discrete_kernel_mu_tuples[kk_ctr][j])
                            n123[i] += 2
                            self.discrete_kernel_mu_tuples[kk_ctr].append(
                                tuple(n123))
                    self.discrete_kernel_mu_tuples[kk_ctr] = list(set(
                        self.discrete_kernel_mu_tuples[kk_ctr]
                    ))
                    for j in range(3):
                        kk_ctr_deriv = 'dk{}sq{}_dlnk{}'.format(
                            i+1, kk, j+1)
                        self.discrete_kernel_mu_tuples[kk_ctr_deriv] = \
                            self.discrete_kernel_mu_tuples[kk_ctr].copy()*2
                        num_tup = int(len(
                            self.discrete_kernel_mu_tuples[kk_ctr_deriv])/2)
                        for k in range(num_tup,2*num_tup):
                            n123 = np.array(
                                self.discrete_kernel_mu_tuples[kk_ctr_deriv][k])
                            n123[j] += 2
                            self.discrete_kernel_mu_tuples[kk_ctr_deriv][k] = \
                                tuple(n123)
                        self.discrete_kernel_mu_tuples[kk_ctr_deriv] = list(set(
                            self.discrete_kernel_mu_tuples[kk_ctr_deriv]
                        ))
                if self.cnlo_type == 'IvaPhiNis':
                    kk_k4ctr = 'k1sqk2sq{}'.format(kk)
                    self.discrete_kernel_mu_tuples[kk_k4ctr] = []
                    if kk == 'k31':
                        n123_k4ctr = [(1,0,1), (1,2,1)]
                    elif kk == 'k32':
                        n123_k4ctr = [(0,1,1), (2,1,1)]
                    elif kk in ['F2','b2','K']:
                        n123_k4ctr = [(0,0,0)]
                    else:
                        n123_k4ctr = [(0,0,2)]
                    for i in range(2):
                        for j in range(2):
                            for n123 in n123_k4ctr:
                                n123_ij = np.copy(n123)
                                n123_ij[0] += 2*(i+1)
                                n123_ij[1] += 2*(j+1)
                                self.discrete_kernel_mu_tuples[
                                    kk_k4ctr].append(tuple(n123_ij))
                    for i in range(3):
                        kk_k4ctr_deriv = 'dk1sqk2sq{}_dlnk{}'.format(
                            kk, i+1)
                        self.discrete_kernel_mu_tuples[kk_k4ctr_deriv] = \
                            self.discrete_kernel_mu_tuples[kk_k4ctr].copy()*2
                        num_tup = int(len(
                            self.discrete_kernel_mu_tuples[kk_k4ctr_deriv])/2)
                        for k in range(num_tup,2*num_tup):
                            n123 = np.array(
                                self.discrete_kernel_mu_tuples[
                                    kk_k4ctr_deriv][k])
                            n123[i] += 2
                            self.discrete_kernel_mu_tuples[kk_k4ctr_deriv][k] \
                                = tuple(n123)
                        self.discrete_kernel_mu_tuples[kk_k4ctr_deriv] = \
                            list(set(
                                self.discrete_kernel_mu_tuples[kk_k4ctr_deriv]
                            ))

        # do after adding derivs and counterterms, only for kk and kk_ctr
        for kk in self.kernel_mu_tuples:
            for i in range(len(self.kernel_mu_tuples[kk])):
                for j in range(3):
                    n123 = np.copy(self.discrete_kernel_mu_tuples[kk][i])
                    n123[j] += 2
                    self.discrete_kernel_mu_tuples[kk].append(tuple(n123))
            self.discrete_kernel_mu_tuples[kk] = list(set(
                self.discrete_kernel_mu_tuples[kk]
            ))
        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            for kk in self.kernel_mu_tuples:
                for i in range(3):
                    kk_ctr = 'k{}sq{}'.format(i+1, kk)
                    for j in range(len(self.discrete_kernel_mu_tuples[kk_ctr])):
                        for k in range(3):
                            n123 = np.array(
                                self.discrete_kernel_mu_tuples[kk_ctr][j])
                            n123[k] += 2
                            self.discrete_kernel_mu_tuples[kk_ctr].append(
                                tuple(n123))
                    self.discrete_kernel_mu_tuples[kk_ctr] = list(set(
                        self.discrete_kernel_mu_tuples[kk_ctr]
                    ))
                if self.cnlo_type == 'IvaPhiNis':
                    kk_ctr = 'k1sqk2sq{}'.format(kk)
                    for j in range(len(self.discrete_kernel_mu_tuples[kk_ctr])):
                        for k in range(3):
                            n123 = np.array(
                                self.discrete_kernel_mu_tuples[kk_ctr][j])
                            n123[k] += 2
                            self.discrete_kernel_mu_tuples[kk_ctr].append(
                                tuple(n123))
                    self.discrete_kernel_mu_tuples[kk_ctr] = list(set(
                        self.discrete_kernel_mu_tuples[kk_ctr]
                    ))

        self.discrete_stoch_kernel_mu_tuples = {}
        self.discrete_stoch_kernel_mu_tuples['id'] = self.n123_tuples_stoch_all
        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            self.discrete_stoch_kernel_mu_tuples['ksq'] = []
            for n123 in self.discrete_stoch_kernel_mu_tuples['id']:
                n123_ctr = np.copy(n123)
                n123_ctr[0] += 2
                self.discrete_stoch_kernel_mu_tuples['ksq'].append(n123_ctr)
            self.discrete_stoch_kernel_mu_tuples['ksq'] = np.array(
                self.discrete_stoch_kernel_mu_tuples['ksq']
            )
            self.discrete_stoch_kernel_mu_tuples['dksq_dlnk'] = np.array([
                [2,0,0], [4,0,0], [6,0,0], [8,0,0]
            ])

    def change_RSD_model(self, model):
        self.model = model
        if not self.real_space:
            self.kernel_names = ['F2', 'G2', 'b2', 'K', 'k31', 'k32']
            kernel_names_deriv = []
            for kk in self.kernel_names:
                for i in range(3):
                    kk_deriv = 'd{}_dlnk{}'.format(kk, i+1)
                    kernel_names_deriv.append(kk_deriv)
            self.kernel_names += kernel_names_deriv
            if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                kernel_names_ctr = []
                for kk in ['F2', 'G2', 'b2', 'K', 'k31', 'k32']:
                    for i in range(3):
                        kk_ctr = 'k{}sq{}'.format(i+1,kk)
                        kernel_names_ctr.append(kk_ctr)
                        for j in range(3):
                            kk_ctr = 'dk{}sq{}_dlnk{}'.format(i+1,kk,j+1)
                            kernel_names_ctr.append(kk_ctr)
                # kernel_names_ctr += ['k1sqb2', 'k2sqb2', 'k3sqb2']
                self.kernel_names += kernel_names_ctr
            self._get_mu_tuples_for_discrete_average()

    def change_cnlo_type(self, type):
        if type in ['EggLeeSco','IvaPhiNis']:
            self.cnlo_type = type
            self.tri = None
            if self.cnlo_type == 'IvaPhiNis':
                if not 'k1sqk2sqF2' in self.kernel_names:
                    kernel_names_k4ctr = []
                    for kk in ['F2', 'G2', 'b2', 'K', 'k31', 'k32']:
                        kk_k4ctr = 'k1sqk2sq{}'.format(kk)
                        kernel_names_k4ctr.append(kk_k4ctr)
                        for i in range(3):
                            kk_k4ctr = 'dk1sqk2sq{}_dlnk{}'.format(kk,i+1)
                            kernel_names_k4ctr.append(kk_k4ctr)
                    self.kernel_names += kernel_names_k4ctr
            elif self.cnlo_type == 'EggLeeSco':
                if 'k1sqk2sqF2' in self.kernel_names:
                    temp_kernel_names = self.kernel_names.copy()
                    for kk in temp_kernel_names:
                        if 'k1sqk2sq' in kk:
                            self.kernel_names.remove(kk)
            self._get_mu_tuples_for_discrete_average()
        else:
            print('Warning. Type not recognised, choose between '
                  '"EggLeeSco" (default), or "IvaPhiNis".')

    def define_units(self, use_Mpc):
        r"""Define units for the bispectrum.

        Sets the internal class attribute **use_Mpc**.

        Parameters
        ----------
        use_Mpc: bool
            Flag that determines if the input and output quantities are
            specified in :math:`\mathrm{Mpc}` (**True**) or
            :math:`h^{-1}\,\mathrm{Mpc}` (**False**) units.
        """
        self.use_Mpc = use_Mpc

    def define_nbar(self, nbar):
        r"""Define the number density of the sample.

        Sets the internal class attribute **nbar** to the value provided as
        input. The latter is intended to be in the set of units currently used
        by the emulator, that can be specified at class instanciation, or using
        the method **define_units**.

        Parameters
        ----------
        nbar: float
            Number density of the sample, in units of
            :math:`\mathrm{Mpc}^{-3}` or :math:`h^3\,\mathrm{Mpc}^{-3}`,
            depending on the value of the class attribute **use_Mpc**.
        """
        self.nbar = np.copy(nbar)

    def set_tri(self, tri, ell, kfun, gl_deg=8, binning=None):
        r"""Define triangular configurations and compute kernels.

        Reads the list of triangular configurations and the request fundamental
        frequency, and sotres them into class attributes. Additionaly computes
        the necessary kernels (and the angular integrals if working in
        redshift-space).

        Parameters
        ----------
        tri:
            List of triangular configurations.
        ell:
            List of multipoles for which the triangular configurations
            correspond to.
        kfun: float
            Fundamental frequency.

        """
        tri_test = max(tri, key=len) if isinstance(tri, list) else tri
        if self.tri is None \
                or any([l not in list(self.ntri_ell.keys()) for l in ell]) \
                or self.kfun != kfun:
            tri_is_subset = False
            tri_has_changed = True
        else:
            tri_mismatch = np.logical_not(np.array_equal(self.tri, tri_test)) \
                or len(tri_test) != max([len(self.tri_id_ell[l])
                                         for l in self.tri_id_ell])
                #or length of tri_test doesn't match length of tri_id_ell
            if tri_mismatch:
                tri_dtype = {'names':['f{}'.format(i) for i in range(3)],
                             'formats':3 * [self.tri.dtype]}
                intersection = np.intersect1d(
                    self.tri.view(tri_dtype),
                    np.ascontiguousarray(tri_test).view(tri_dtype)
                )
                tri_is_subset = len(intersection) == len(tri_test)
                tri_has_changed = np.logical_not(tri_is_subset)
            else:
                if isinstance(tri,list):
                    tri_is_subset = any([self.ntri_ell[l] != len(tri[i])
                                         for i,l in enumerate(ell)])
                else:
                    tri_is_subset = any([self.ntri_ell[l] != tri.shape[0]
                                         for i,l in enumerate(ell)])
                tri_has_changed = False

        self.binning_turned_on = binning is not None \
                                 and not self.last_eval_binned
        self.binning_turned_off = binning is None and self.last_eval_binned

        def change_tri(tri):
            if isinstance(tri, list):
                self.tri = max(tri, key=len)
                self.ntri_ell = {}
                if self.real_space:
                    self.ntri_ell[0] = self.tri.shape[0]
                else:
                    for i,l in enumerate(ell):
                        self.ntri_ell[l] = tri[i].shape[0]
            else:
                self.tri = tri
                self.ntri_ell = {}
                if self.real_space:
                    self.ntri_ell[0] = self.tri.shape[0]
                else:
                    for i,l in enumerate(ell):
                        self.ntri_ell[l] = tri.shape[0]
            if not self.tri.flags['CONTIGUOUS']:
                self.tri = np.ascontiguousarray(self.tri)
            tri_dtype = {'names':['f{}'.format(i) for i in range(3)],
                         'formats':3 * [self.tri.dtype]}
            self.tri_id_ell = {}
            if isinstance(tri, list):
                for i,l in enumerate(ell):
                    self.tri_id_ell[l] = np.sort(np.intersect1d(
                        self.tri.view(tri_dtype),
                        np.ascontiguousarray(tri[i]).view(tri_dtype),
                        return_indices=True)[1])
            else:
                for l in ell:
                    self.tri_id_ell[l] = np.arange(self.tri.shape[0])

        def update_tri_id(tri):
            tri_dtype = {'names':['f{}'.format(i) for i in range(3)],
                         'formats':3 * [self.tri.dtype]}
            if isinstance(tri, list):
                if self.real_space:
                    self.ntri_ell[0] = self.tri.shape[0]
                else:
                    for i,l in enumerate(ell):
                        self.ntri_ell[l] = tri[i].shape[0]
                for i,l in enumerate(ell):
                    self.tri_id_ell[l] = np.sort(np.intersect1d(
                        self.tri.view(tri_dtype),
                        np.ascontiguousarray(tri[i]).view(tri_dtype),
                        return_indices=True)[1])
            else:
                if self.real_space:
                    self.ntri_ell[0] = self.tri.shape[0]
                else:
                    for i,l in enumerate(ell):
                        self.ntri_ell[l] = tri.shape[0]
                self.tri_id_ell[0] = np.sort(np.intersect1d(
                    self.tri.view(tri_dtype),
                    np.ascontiguousarray(tri).view(tri_dtype),
                    return_indices=True)[1])
                for l in ell:
                    self.tri_id_ell[l] = self.tri_id_ell[0]

        if tri_has_changed or self.binning_turned_off:
            change_tri(tri)
            self.kfun = kfun
            self.generate_index_arrays()
            self.cov_mixing_kernel = {}

            if binning is None:
                # print('Recompute (non-binned) kernels!')
                if not self.calibration_mode \
                        and 'VDG_infty_ctr' in self.model:
                    self.pow_ctr = 2.0
                    new_model = self.model.replace('_ctr','')
                    self.change_RSD_model(new_model)
                self.discrete_average = False
                self.use_effective_triangles = False
                self.compute_kernels(self.tri)
                if not self.real_space:
                    if 'VDG_infty' in self.model:
                        self.compute_mu123_integrals(self.tri, max(ell))
                        self.Gauss_Legendre_mu123_integrals(self.tri, gl_deg,
                                                            max(ell))
                    else:
                        self.compute_mu123_integrals(self.tri, max(ell))
        elif tri_is_subset:
            update_tri_id(tri)

        if binning:
            binning_has_changed = self.binning != binning
            if self.model in ['VDG_infty','VDG_infty_nonu'] and \
                    not binning.get('effective',False):
                self.pow_ctr = 1.75
                idx = self.model.index('_nonu') if 'nonu' in self.model \
                      else len(self.model)
                new_model = self.model[:idx] + '_ctr' + self.model[idx:]
                self.change_RSD_model(new_model)
            if tri_has_changed or binning_has_changed or self.binning_turned_on:
                if binning.get('effective',False) and \
                        'VDG_infty_ctr' in self.model:
                    self.pow_ctr = 2.0
                    new_model = self.model.replace('_ctr','')
                    self.change_RSD_model(new_model)
                change_tri(tri)
                self.binning = binning
                if self.grid is None:
                    self.grid = CtypedGrid(**self.binning)
                else:
                    self.grid.update(**self.binning)
                # self.tri_unique = np.arange(
                #     int(np.around(np.amax(self.tri)/self.binning.get('dk')))
                # )
                # self.tri_unique = self.tri_unique * self.binning.get('dk') \
                #                   + self.binning.get('first_bin_centre')
                self.tri_unique = np.unique(self.tri)
                if self.binning.get('effective', False):
                    self.discrete_average = False
                    self.use_effective_triangles = True
                    self.grid.find_discrete_triangles(self.tri_unique)
                    self.grid.compute_effective_triangles(self.tri_unique)
                    self.tri_eff = np.copy(self.grid.k123eff)
                    self.tri_eff = np.flip(np.sort(self.tri_eff, axis=1),
                                           axis=1)
                    self.generate_eff_index_arrays()
                    self.compute_kernels(self.tri_eff)
                    if not self.real_space:
                        if 'VDG_infty' in self.model:
                            self.compute_mu123_integrals(self.tri, max(ell))
                            self.Gauss_Legendre_mu123_integrals(self.tri, gl_deg,
                                                                max(ell))
                        else:
                            self.compute_mu123_integrals(self.tri, max(ell))
                else:
                    self.discrete_average = True
                    self.use_effective_triangles = False

                    if self.binning.get('filename_root_kernels'):
                        try:
                            binning_from_file = np.load(
                                '{}_dict.npy'.format(
                                    self.binning.get('filename_root_kernels')),
                                allow_pickle=True
                            )
                            tri_from_file = np.loadtxt('{}_tri.dat'.format(
                                self.binning.get('filename_root_kernels')))
                            close_tri = [
                                np.isclose(x,tri_from_file).all(axis=1).any() \
                                for x in self.tri
                            ]
                            tri_from_file_is_superset = \
                                len(close_tri) == len(self.tri)
                            if self.binning == binning_from_file.item() \
                                    and tri_from_file_is_superset:
                                self.generate_discrete_kernels = False
                                change_tri(tri_from_file)
                                update_tri_id(tri)
                            else:
                                self.generate_discrete_kernels = True
                        except Exception:
                            self.generate_discrete_kernels = True
                        if self.generate_discrete_kernels:
                            np.save('{}_dict.npy'.format(
                                self.binning.get('filename_root_kernels')),
                                self.binning
                            )
                            np.savetxt('{}_tri.dat'.format(
                                self.binning.get('filename_root_kernels')),
                                self.tri
                            )
                    else:
                        self.generate_discrete_kernels = True

                    self.tri_eff = np.copy(self.tri)
                    self.generate_eff_index_arrays()

                    if self.generate_discrete_kernels:
                        tri_bin_centres = []
                        offset = self.binning.get('first_bin_centre') \
                                 / self.binning.get('dk') * 1.00001
                        for i,k1 in enumerate(self.tri_unique):
                            for j,k2 in enumerate(self.tri_unique[:i+1]):
                                for n,k3 in enumerate(self.tri_unique[:j+1]):
                                    if offset+j+n > i:
                                        tri_bin_centres.append([k1,k2,k3])
                        tri_bin_centres = np.array(tri_bin_centres)
                        if tri_bin_centres.shape[0] > self.tri.shape[0]:
                            tri_bin_centres = \
                                tri_bin_centres[:self.tri.shape[0]]
                        a = np.mean(self.grid.shape_limits)
                        b = 0.5 * (self.grid.shape_limits[1] \
                                   - self.grid.shape_limits[0])
                        check = np.abs(
                            (tri_bin_centres[:,2]+tri_bin_centres[:,1]) \
                            / tri_bin_centres[:,0] - a) < b*(1.0 - 0.00001)
                        self.tri_ids_discrete_binning = np.where(check)[0]
                        self.tri_ids_eff = np.where(
                            np.logical_not(check))[0]
                        self.grid.find_discrete_triangles(self.tri_unique)
                        self.compute_kernels(self.tri[self.tri_ids_eff])
                        self.compute_mu123_integrals(self.tri[self.tri_ids_eff],
                                                     max(ell))
                        # self.compute_kernels_shell_average(max(ell))
            elif tri_is_subset:
                update_tri_id(tri)
            self.last_eval_binned = True
        else:
            binning_has_changed = False
            self.last_eval_binned = False

        return tri_has_changed, binning_has_changed

    def set_fiducial_cosmology(self, params):
        #unique_z, index = np.unique(params['z'], return_index=True)
        #self.num_fiducials = len(unique_z)
        self.num_fiducials = len(np.atleast_1d(params['z']))
        if self.binning.get('fiducial_cosmology') is None:
            self.fiducial_cosmology = {
                'h':np.repeat(0.6736, self.num_fiducials),
                'wc':np.repeat(0.12, self.num_fiducials),
                'wb':np.repeat(0.02237, self.num_fiducials),
                'ns':np.repeat(0.9649, self.num_fiducials),
                'As':np.repeat(2.0989031673, self.num_fiducials),
                'w0':np.repeat(-1.0, self.num_fiducials),
                'wa':np.repeat(0.0, self.num_fiducials),
                'z':np.atleast_1d(params['z']) #unique_z[index.argsort()]
            }
        else:
            self.fiducial_cosmology = self.binning.get('fiducial_cosmology')
            if 'z' not in self.fiducial_cosmology:
                if len(np.atleast_1d(self.fiducial_cosmology['wc'])) != \
                        self.num_fiducials:
                    for p in self.fiducial_cosmology:
                        self.fiducial_cosmology[p] = np.repeat(
                            self.fiducial_cosmology[p], self.num_fiducials)
                self.fiducial_cosmology['z'] = np.atleast_1d(params['z'])

    def init_Pdw(self, Pdw, ell):
        self.fiducial_Pdw = Pdw
        self.fiducial_Pdw_sq = np.zeros_like(Pdw)
        for i in range(3):
            self.fiducial_Pdw_sq[:,i] = Pdw[:,i%3]*Pdw[:,(i+1)%3]
        # self.compute_kernels_shell_average(max(ell))

    def init_Pdw_eff(self, Pdw_eff):
        self.fiducial_Pdw_eff = Pdw_eff

    def F2(self, k1, k2, k3):
        r"""Compute the second-order density kernel.

        Computes the second order density kernel :math:`F_2` on the triangular
        configuration defined by the input wavemodes :math:`(k_1,k_2,k_3)`,
        using the angle between :math:`k_1` and :math:`k_2`.

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        F2: float
            Second-order density kernel :math:`F_2` between the wavemodes
            :math:`k_1` and :math:`k_2`.
        """
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu**2

    def G2(self, k1, k2, k3):
        r"""Compute the second-order velocity divergence kernel.

        Computes the second order velocity divergence kernel :math:`G_2` on
        the triangular configuration defined by the input wavemodes
        :math:`(k_1,k_2,k_3)`, using the angle between :math:`k_1` and
        :math:`k_2`.

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        G2: float
            Second-order velocity divergence kernel :math:`G_2` between the
            wavemodes :math:`k_1` and :math:`k_2`.
        """
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return 3.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 4.0/7.0 * mu**2

    def K(self, k1, k2, k3):
        r"""Compute the Fourier-space kernel of the second-order Galileon.

        Computes the Fourier-space kernel of the second-order Galileon
        :math:`K` on the triangular configuration defined by the input
        wavemodes :math:`(k_1,k_2,k_3)`, using the angle between :math:`k_1`
        and :math:`k_2`.

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        K: float
            Fourier-space kernel of the second-order Galileon :math:`K`
            between the wavemodes :math:`k_1` and :math:`k_2`.
        """
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        return mu**2 - 1.0

    def kernels_real_space(self, k1, k2, k3):
        r"""Compute the kernels for the real-space bispectrum.

        Computes the kernels required for the real-space bispectrum, and
        returns them in a dictionary format. This includes only the
        second-order density kernel :math:`F_2` and the Fourier-space kernel
        of the second-order galileon :math:`K`

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        kernels: dict
            Dictionary containing the kernels required to model the real-space
            bispectrum.
        """
        kernels = {}
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        kernels['F2'] = 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu**2
        kernels['b2'] = 1.0
        kernels['K'] = mu**2 - 1.0
        return kernels

    def kernels_redshift_space(self, k1, k2, k3):
        r"""Compute the kernels for the redshift-space bispectrum.

        Computes the kernels required for the redshift-space bispectrum, and
        returns them in a dictionary format. This includes the second-order
        density and velocity divergence kernels, :math:`F_2` and :math:`G_2`,
        the Fourier-space kernel of the second-order galileon :math:`K`, the
        ratios of :math:`k_3` to the other two wavemodes, and the logarithmic
        derivatives of the previous kernels with respect to :math:`k_1`,
        :math:`k_2` and :math:`k_3`.

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        kernels: dict
            Dictionary containing the kernels required to model the
            redshift-space bispectrum.
        """
        k1sq = k1**2
        k2sq = k2**2
        k3sq = k3**2
        mu = (k3sq - k1sq - k2sq)/(2*k1*k2)
        mu2 = mu**2
        k1muk2 = k1 * mu / k2
        k2muk1 = k2 * mu / k1

        kernels = {}
        kernels['F2'] = 5.0/7.0 + 0.5 * (k1muk2 + k2muk1) + 2.0/7.0 * mu2
        kernels['G2'] = 3.0/7.0 + 0.5 * (k1muk2 + k2muk1) + 4.0/7.0 * mu2
        kernels['b2'] = 1.0
        kernels['K'] = mu2 - 1.0
        kernels['k31'] = k3/k1
        kernels['k32'] = k3/k2

        kernels['dF2_dlnk1'] = -0.5 - 0.5*k1sq/k2sq - 4.0/7.0 * k1muk2 \
                               - k2muk1 - 4.0/7.0*mu2
        kernels['dF2_dlnk2'] = -0.5 - 0.5*k2sq/k1sq - k1muk2 \
                               - 4.0/7.0*k2muk1 - 4.0/7.0*mu2
        kernels['dF2_dlnk3'] = (k3sq*(7.0*(k1sq + k2sq) + 8.0*k1*k2*mu)) \
                               / (14.0*k1sq*k2sq)

        kernels['dG2_dlnk1'] = kernels['dF2_dlnk1'] - 4.0/7.0*(k1muk2 + mu2)
        kernels['dG2_dlnk2'] = kernels['dF2_dlnk2'] - 4.0/7.0*(k2muk1 + mu2)
        kernels['dG2_dlnk3'] = kernels['dF2_dlnk3'] + 4.0*k3sq*mu \
                               / (7.0*k1*k2)

        kernels['db2_dlnk1'] = 0.0
        kernels['db2_dlnk2'] = 0.0
        kernels['db2_dlnk3'] = 0.0

        kernels['dK_dlnk1'] = -2.0*(k1muk2 + mu2)
        kernels['dK_dlnk2'] = -2.0*(k2muk1 + mu2)
        kernels['dK_dlnk3'] = 2.0*(k1muk2 + k2muk1) + 4.0*mu2

        kernels['dk31_dlnk1'] = -kernels['k31']
        kernels['dk31_dlnk2'] = 0.0
        kernels['dk31_dlnk3'] = kernels['k31']

        kernels['dk32_dlnk1'] = 0.0
        kernels['dk32_dlnk2'] = -kernels['k32']
        kernels['dk32_dlnk3'] = kernels['k32']

        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            k123sq = np.vstack((k1sq,k2sq,k3sq))**(self.pow_ctr/2)
            kernel_names = ['F2','G2','b2','K','k31','k32']
            for kk in kernel_names:
                for i in range(3):
                    kernels['k{}sq{}'.format(i+1,kk)] = k123sq[i]*kernels[kk]
                    for j in range(3):
                        kernels['dk{}sq{}_dlnk{}'.format(i+1,kk,j+1)] = \
                            k123sq[i]*kernels['d{}_dlnk{}'.format(kk,j+1)]
                        if i == j:
                            kernels['dk{}sq{}_dlnk{}'.format(i+1,kk,j+1)] += \
                                self.pow_ctr*k123sq[i]*kernels[kk]
            if self.cnlo_type == 'IvaPhiNis':
                k1sqk2sq = k1sq*k2sq
                for kk in kernel_names:
                    kernels['k1sqk2sq{}'.format(kk)] = k1sqk2sq*kernels[kk]
                    for i in range(3):
                        kernels['dk1sqk2sq{}_dlnk{}'.format(kk,i+1)] = \
                            k1sqk2sq*kernels['d{}_dlnk{}'.format(kk,i+1)]
                        if i in [0,1]:
                            kernels['dk1sqk2sq{}_dlnk{}'.format(kk,i+1)] += \
                                self.pow_ctr*k1sqk2sq*kernels[kk]

            # for i in range(3):
            #     kernels['k{}sqb2'.format(i+1)] = k123[i]**2
            #     for j in range(3):
            #         if i == j:
            #             kernels['dk{}sqb2_dlnk{}'.format(i+1,j+1)] = \
            #                 2*k123[i]**2
            #         else:
            #             kernels['dk{}sqb2_dlnk{}'.format(i+1,j+1)] = 0.0

        return kernels

    def _kernels_real_space(self, k1, k2, k3):
        r"""Compute the kernels for the real-space bispectrum.

        Computes the kernels required for the real-space bispectrum, and
        returns them in a dictionary format. This includes only the
        second-order density kernel :math:`F_2` and the Fourier-space kernel
        of the second-order galileon :math:`K`

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        kernels: numpy.array
            Array of size (k1.size, 3) with the first column corresponding to the F2 kernel, the second to the b_2 kernel and the third to the K kernel.
        """
        kernels = np.zeros([k1.size,3])
        mu = (k3**2 - k1**2 - k2**2)/(2*k1*k2)
        mu2 = mu**2
        kernels[:,0] = 5.0/7.0 + mu/2 * (k1/k2 + k2/k1) + 2.0/7.0 * mu2
        kernels[:,1] = 1.0
        kernels[:,2] = mu2 - 1.0
        return kernels

    def _kernels_redshift_space(self, k1, k2, k3):
        r"""Compute the kernels for the redshift-space bispectrum.

        Computes the kernels required for the redshift-space bispectrum, and
        returns them in a dictionary format. This includes the second-order
        density and velocity divergence kernels, :math:`F_2` and :math:`G_2`,
        the Fourier-space kernel of the second-order galileon :math:`K`, the
        ratios of :math:`k_3` to the other two wavemodes, and the logarithmic
        derivatives of the previous kernels with respect to :math:`k_1`,
        :math:`k_2` and :math:`k_3`.

        Parameters
        ----------
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        kernels: numpy.array
            Array of size (k1.size, 20) with the columns corresponding to the following kernels:
            - :math:`F_2`
            - :math:`G_2`
            - :math:`b_2`
            - :math:`K`
            - :math:`k_3/k_1`
            - :math:`k_3/k_2`
            - :math:`d F_2/d \log{k_1}`
            - :math:`d F_2/d \log{k_2}`
            - :math:`d F_2/d \log{k_3}`
            - :math:`d G_2/d \log{k_1}`
            - :math:`d G_2/d \log{k_2}`
            - :math:`d G_2/d \log{k_3}`
            - :math:`d K/d \log{k_1}`
            - :math:`d K/d \log{k_2}`
            - :math:`d K/d \log{k_3}`
        """
        n_kernels = 24 if self.model in ['VDG_infty','VDG_infty_nonu'] else 96
        if self.cnlo_type == 'IvaPhiNis':
            n_kernels += 24
        kernels = np.zeros([k1.size,n_kernels])

        k1sq = k1**2
        k2sq = k2**2
        k3sq = k3**2
        mu = (k3sq - k1sq - k2sq)/(2*k1*k2)
        mu2 = mu**2
        k1muk2 = k1*mu/k2
        k2muk1 = k2*mu/k1

        if self.cnlo_type == 'IvaPhiNis':
            k1sqk2sq = k1sq*k2sq

        # F2
        kernels[:,0] = 5.0/7.0 + 0.5 * (k1muk2 + k2muk1) + 2.0/7.0 * mu2
        kernels[:,1] = -0.5 - 0.5*k1sq/k2sq - 4.0/7.0 * k1muk2 \
                       - k2muk1 - 4.0/7.0*mu2
        kernels[:,2] = -0.5 - 0.5*k2sq/k1sq - k1muk2 \
                       - 4.0/7.0*k2muk1 - 4.0/7.0*mu2
        kernels[:,3] = (k3sq*(7.0*(k1sq + k2sq) + 8.0*k1*k2*mu)) \
                       / (14.0*k1sq*k2sq)

        # G2
        kernels[:,4] = 3.0/7.0 + 0.5 * (k1muk2 + k2muk1) + 4.0/7.0 * mu2
        kernels[:,5] = kernels[:,1] - 4.0/7.0*(k1muk2 + mu2)
        kernels[:,6] = kernels[:,2] - 4.0/7.0*(k2muk1 + mu2)
        kernels[:,7] = kernels[:,3] + 4.0*k3sq*mu/(7.0*k1*k2)

        # b2
        kernels[:,8] = 1.0
        kernels[:,9] = 0.0
        kernels[:,10] = 0.0
        kernels[:,11] = 0.0

        # K
        kernels[:,12] = mu2 - 1.0
        kernels[:,13] = -2.0*(k1muk2 + mu2)
        kernels[:,14] = -2.0*(k2muk1 + mu2)
        kernels[:,15] = 2.0*(k1muk2 + k2muk1) + 4.0*mu2

        # k31
        kernels[:,16] = k3/k1
        kernels[:,17] = -kernels[:,16]
        kernels[:,18] = 0.0
        kernels[:,19] = kernels[:,16]

        # k32
        kernels[:,20] = k3/k2
        kernels[:,21] = 0.0
        kernels[:,22] = -kernels[:,20]
        kernels[:,23] = kernels[:,20]

        k123sq = np.vstack((k1sq,k2sq,k3sq))**(self.pow_ctr/2)
        kernel_names = ['F2','G2','b2','K','k31','k32']
        count = 24
        for n in range(len(kernel_names)):
            kk = kernel_names[n]
            for i in range(3):
                kernels[:,count] = k123sq[i]*kernels[:,n*4]
                count += 1
                for j in range(3):
                    kernels[:,count] = k123sq[i]*kernels[:,n*4+j+1]
                    if i == j:
                        kernels[:,count] += \
                            self.pow_ctr*k123sq[i]*kernels[:,n*4]
                    count += 1
            if self.cnlo_type == 'IvaPhiNis':
                kernels[:,count] = k1sqk2sq*kernels[:,n*4]
                count += 1
                for i in range(3):
                    kernels[:,count] = k1sqk2sq*kernels[:,n*4+i+1]
                    if i in [0,1]:
                        kernels[:,count] += self.pow_ctr \
                                            * kernels[:,count-i-1]
                    count += 1

        return kernels

    def mu123_integrals(self, n1, n2, n3, k1, k2, k3):
        r"""Angular integration.

        Computes the integral

        .. math::
            \frac{1}/{4\pi} \int {\rm{d}}\mu \int {\rm{d}}\phi \
            \mu_1^{n_1}\mu_2^{n_2}\mu_3^{n_3},

        where

        .. math::
            \begin{flalign*}
                & \mu_1 = \mu, \\
                & \mu_2 = \mu\nu - \sqrt(1-\mu^2)\sqrt(1-\nu^2)\cos(\phi), \\
                & \mu_3 = -\frac{k_1}{k_3}\mu_1 - \frac{k_2}{k_3}\mu_2,
            \end{flalign*}

        and :math:`\mu_n` is the cosinus of the angle bewteen the wavemode
        :math:`k_n` and the line of sight, and :math:`\nu` is the cosinus of
        the angle between :math:`k_1` and :math:`k_2`.

        Parameters
        ----------
        n1: int
            Power of wavemode :math:`k_1`.
        n2: int
            Power of wavemode :math:`k_2`.
        n3: int
            Power of wavemode :math:`k_3`.
        k1: float
            Wavemode :math:`k_1`.
        k2: float
            Wavemode :math:`k_2`.
        k3: float
            Wavemode :math:`k_3`.

        Returns
        -------
        I: float
            Angular integration of the different powers of the input angles.
        """
        if n2 == 0 and n3 == 0:
            I = 1.0/(1.0 + n1)
        elif n2 == 1 and n3 == 0:
            I = -0.5/(2.0 + n1) * (k1**2 + k2**2 - k3**2)/(k1*k2)
        elif n2 == 2 and n3 == 0:
            I = (4*k1**2*k2**2 + (k1**2 + k2**2 - k3**2)**2*n1) \
                / (4.*k1**2*k2**2*(1 + n1)*(3 + n1))
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
        elif n2 == 4 and n3 == 0:
            I = (48*k1**4*k2**4 - 2*(k1**2 + k2**2 - k3**2)**2 \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(5*k2**2 + k3**2))*n1 \
                + (k1**2 + k2**2 - k3**2)**4*n1**2) \
                / (16.*k1**4*k2**4*(1 + n1)*(3 + n1)*(5 + n1))
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
        elif n2 == 5 and n3 == 0:
            I = -0.03125*((k1**2 + k2**2 - k3**2)*(k1**8*(-3 + n1)*(-1 + n1) \
                + (k2 - k3)**4*(k2 + k3)**4*(-3 + n1)*(-1 + n1) + 4*k1**6 \
                * (-1 + n1)*(-(k3**2*(-3 + n1)) + k2**2*(7 + n1)) + 4*k1**2 \
                * (k2**2 - k3**2)**2*(-1 + n1)*(-(k3**2*(-3 + n1)) + k2**2*(7 \
                + n1)) + 2*k1**4*(3*k3**4*(-3 + n1)*(-1 + n1) - 2*k2**2*k3**2 \
                * (-1 + n1)*(11 + 3*n1) + k2**4*(89 + n1*(28 + 3*n1))))) \
                / (k1**5*k2**5*(2 + n1)*(4 + n1)*(6 + n1))
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
        elif n2 == 5 and n3 == 1:
            I = (30*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                - 2*(k1**2 + k2**2 - k3**2)*(1 + n1) \
                * (15*(k1**2 + 3*k2**2 - 3*k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**2 + 5*(k1 - k2 - k3) \
                * (k1 + k2 - k3)*(k1 - k2 + k3)*(k1 + k2 + k3) \
                * (k1**2 + k2**2 - k3**2)**2*(k1**2 - 3*k2**2 + 3*k3**2) \
                *(3 + n1) - (k1**2 + k2**2 - k3**2)**4*(k1**2 - k2**2 + k3**2) \
                *(3 + n1)*(5 + n1))) \
                / (128.*k1**6*k2**5*k3*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))
        elif n2 == 6 and n3 == 0:
            I = -0.015625*(15*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**3 - 45*(k1**2 + k2**2 - k3**2)**2*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2*(1 + n1) \
                + 15*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 - k2 + k3)*(k1 + k2 \
                + k3)*(k1**2 + k2**2 - k3**2)**4*(1 + n1)*(3 + n1) - (k1**2 \
                + k2**2 - k3**2)**6*(1 + n1)*(3 + n1)*(5 + n1)) \
                / (k1**6*k2**6*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))
        elif n2 == 4 and n3 == 2:
            I = (-30*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                - 6*(k1**4 - 10*k1**2*(k2**2 - k3**2) - 15*(k2**2 - k3**2)**2) \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 \
                * (1 + n1) + 2*(k1**2 + k2**2 - k3**2)**2*(1 + n1)*(3 + n1) \
                * (4*k1**6*(2*k2**2 - 3*k3**2) + 20*k1**2*(2*k2**6 \
                - 3*k2**4*k3**2 + k3**6) + (k2**2 - k3**2)**4*(-10 + n1) \
                + k1**8*(6 + n1) - 2*k1**4*(k2 - k3)*(k2 + k3) \
                * (-(k3**2*(2 + n1)) + k2**2*(22 + n1)))) \
                / (128.*k1**6*k2**4*k3**2*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))
        elif n2 == 3 and n3 == 3:
            I = (30*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                + 18*(k1**4 - 5*(k2**2 - k3**2)**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 \
                * (1 + n1) + 2*(k1**4 - (k2**2 - k3**2)**2)*(1 + n1)*(3 + n1) \
                * (-6*k1**6*(k2**2 + k3**2) + 30*k1**2*(k2**2 - k3**2)**2 \
                * (k2**2 + k3**2) + (k2**2 - k3**2)**4*(-10 + n1) + k1**8 \
                * (8 + n1) - 2*k1**4*(k2**2 - k3**2)**2*(11 + n1))) \
                / (128.*k1**6*k2**3*k3**3*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))
        elif n2 == 7 and n3 == 0:
            I = ((k1**2 + k2**2 - k3**2)*(105*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3 - 105*(k1**2 + k2**2 \
                - k3**2)**2*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2*(2 + n1) + 21*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 \
                - k2 + k3)*(k1 + k2 + k3)*(k1**2 + k2**2 - k3**2)**4*(2 + n1) \
                * (4 + n1) - (k1**2 + k2**2 - k3**2)**6*(2 + n1)*(4 + n1) \
                * (6 + n1)))/(128.*k1**7*k2**7*(2 + n1)*(4 + n1)*(6 + n1) \
                * (8 + n1))
        elif n2 == 6 and n3 == 1:
            I = (-15*(5*k1**2 + 7*k2**2 - 7*k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3 + 15*(k1**2 + 7*k2**2 - 7*k3**2) \
                * (k1**2 + k2**2 - k3**2)**2*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**2*(2 + n1) + 3*(k1 - k2 - k3) \
                * (k1 + k2 - k3)*(k1 - k2 + k3)*(k1 + k2 + k3)*(k1**2 + k2**2 \
                - k3**2)**4*(3*k1**2 - 7*k2**2 + 7*k3**2)*(2 + n1)*(4 + n1) \
                - (k1**2 + k2**2 - k3**2)**6*(k1**2 - k2**2 + k3**2)*(2 + n1) \
                * (4 + n1)*(6 + n1)) \
                / (128.*k1**7*k2**6*k3*(2 + n1)*(4 + n1)*(6 + n1)*(8 + n1))
        elif n2 == 5 and n3 == 2:
            I = (15*(3*k1**2 + 7*k2**2 - 7*k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3*n1 + (k1**2 + k2**2 - k3**2) \
                * n1*(2 + n1)*(15*(k1**4 - 2*k1**2*(k2**2 - k3**2) - 7*(k2**2 \
                - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2 - (k1**2 + k2**2 - k3**2)**2*(4 + n1)*(4*k1**6 \
                * (5*k2**2 - 4*k3**2) + 12*k1**2*(k2**2 - k3**2)**2*(5*k2**2 \
                + 2*k3**2) + (k2**2 - k3**2)**4*(-15 + n1) + k1**8*(5 + n1) \
                - 2*k1**4*(k2 - k3)*(k2 + k3)*(-(k3**2*(-1 + n1)) + k2**2*(35 \
                + n1)))))/(128.*k1**7*k2**5*k3**2*n1*(2 + n1)*(4 + n1) \
                * (6 + n1)*(8 + n1))
        elif n2 == 4 and n3 == 3:
            I = -0.0078125*(15*(k1**2 + 7*k2**2 - 7*k3**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 + 3*(3*k1**6\
                + 15*k1**4*(k2**2 - k3**2) - 15*k1**2*(k2**2 - k3**2)**2 \
                - 35*(k2**2 - k3**2)**3)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2 \
                * (k2**2 + k3**2))**2*(2 + n1) + (k1**2 + k2**2 - k3**2)**2 \
                * (k1**2 - k2**2 + k3**2)*(2 + n1)*(4 + n1)*(-12*k1**6*k3**2 \
                + 12*k1**2*(k2**2 - k3**2)**2*(4*k2**2 + 3*k3**2) \
                + (k2**2 - k3**2)**4*(-15 + n1) + k1**8*(9 + n1) - 2*k1**4 \
                * (k2 - k3)*(k2 + k3)*(-(k3**2*(9 + n1)) + k2**2*(21 + n1)))) \
                / (k1**7*k2**4*k3**3*(2 + n1)*(4 + n1)*(6 + n1)*(8 + n1))
        elif n2 == 8 and n3 == 0:
            I = (105*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4 \
                - 420*(k1**2 + k2**2 - k3**2)**2*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3*(1 + n1) + 210*(k1**2 + k2**2 \
                - k3**2)**4*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2*(1 + n1)*(3 + n1) - 28*(k1 - k2 - k3)*(k1 + k2 \
                - k3)*(k1 - k2 + k3)*(k1 + k2 + k3)*(k1**2 + k2**2 - k3**2)**6 \
                * (1 + n1)*(3 + n1)*(5 + n1) + (k1**2 + k2**2 - k3**2)**8 \
                * (1 + n1)*(3 + n1)*(5 + n1)*(7 + n1))/(256.*k1**8*k2**8 \
                * (1 + n1)*(3 + n1)*(5 + n1)*(7 + n1)*(9 + n1))
        elif n2 == 7 and n3 == 1:
            I = (-210*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**4 + 4*(k1**2 + k2**2 - k3**2)*(1 + n1)*(105*(k1**2 \
                + 2*k2**2 - 2*k3**2)*(k1**4 + (k2**2 - k3**2)**2 -2*k1**2 \
                * (k2**2 + k3**2))**3 - 105*(k2**2 - k3**2)*(k1**2 + k2**2 \
                - k3**2)**2*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2*(3 + n1) - 7*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 \
                - k2 + k3)*(k1 + k2 + k3)*(k1**2 + k2**2 - k3**2)**4*(k1**2 \
                - 2*k2**2 + 2*k3**2)*(3 + n1)*(5 + n1) + ((k1**2 + k2**2 \
                - k3**2)**6*(k1**2 - k2**2 + k3**2)*(3 + n1)*(5 + n1) \
                * (7 + n1))/2.))/(512.*k1**8*k2**7*k3*(1 + n1)*(3 + n1) \
                * (5 + n1)*(7 + n1)*(9 + n1))
        elif n2 == 6 and n3 == 2:
            I = (210*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**4  - 2*(1 + n1)*(60*(k1**4 + 7*k1**2*(k2**2 \
                - k3**2) + 7*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3 + 30*(k1**2 + k2**2 \
                - k3**2)**2*(k1**4 - 7*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2*(3 + n1) \
                - (k1**2 + k2**2 - k3**2)**4*(3 + n1)*(5 + n1) \
                * (4*k1**6*(9*k2**2 - 5*k3**2) + 28*k1**2*(k2**2 - k3**2)**2 \
                * (3*k2**2 + k3**2) + (k2**2 - k3**2)**4*(-21 + n1) + k1**8 \
                * (3 + n1) - 2*k1**4*(k2 - k3)*(k2 + k3)*(-(k3**2*(-5 + n1)) \
                + k2**2*(51 + n1)))))/(512.*k1**8*k2**6*k3**2*(1 + n1) \
                * (3 + n1)*(5 + n1)*(7 + n1)*(9 + n1))
        elif n2 == 5 and n3 == 3:
            I = -0.00390625*(105*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**4 + 30*(k1**4 - 7*k1**2*(k2**2 - k3**2) - 14 \
                * (k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 -2*k1**2 \
                * (k2**2 + k3**2))**3*(1 + n1) - (k1**2 + k2**2 - k3**2) \
                * (1 + n1)*(3 + n1)*(-30*(k2**2 - k3**2)*(-3*k1**4 + 7*(k2**2 \
                - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2 + (k1**2 + k2**2 - k3**2)**2*(k1**2 - k2**2 \
                + k3**2)*(5 + n1)*(2*k1**6*(5*k2**2 - 9*k3**2) + 14*k1**2 \
                * (k2**2 - k3**2)**2*(5*k2**2 + 3*k3**2) + (k2**2 - k3**2)**4 \
                * (-21 + n1) + k1**8*(9 + n1) - 2*k1**4*(k2 - k3)*(k2 + k3) \
                * (-(k3**2*(6 + n1)) + k2**2*(34 + n1)))))/(k1**8 \
                * k2**5*k3**3*(1 + n1)*(3 + n1)*(5 + n1)*(7 + n1)*(9 + n1))
        elif n2 == 4 and n3 == 4:
            I = (210*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4 \
                + 8*(1 + n1)*(15*(k1**4 - 7*(k2**2 - k3**2)**2) \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                + (3*(3*k1**8 - 30*k1**4*(k2**2 - k3**2)**2 + 35*(k2**2 \
                - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2*(3 + n1))/2. + (k1 - k2 - k3)*(k1 + k2 - k3) \
                * (k1 - k2 + k3)*(k1 + k2 + k3)*(k1**4 - 7*(k2**2 - k3**2)**2) \
                * (k1**4 - (k2**2 - k3**2)**2)**2*(3 + n1)*(5 + n1) \
                + ((k1**4 - (k2**2 - k3**2)**2)**4*(3 + n1)*(5 + n1) \
                * (7 + n1))/4.))/(512.*k1**8*k2**4*k3**4*(1 + n1)*(3 + n1) \
                * (5 + n1)*(7 + n1)*(9 + n1))
        elif n2 == 8 and n3 == 1:
            I = (105*(7*k1**2 + 9*k2**2 - 9*k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**4*n1 - (k1**2 + k2**2 - k3**2)**2 \
                * n1*(2 + n1)*(420*(k1**2 + 3*k2**2 - 3*k3**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 + 42*(k1**2 \
                + k2**2 - k3**2)**2*(k1**2 - 9*k2**2 + 9*k3**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2*(4 + n1) \
                - 4*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 - k2 + k3)*(k1 + k2 \
                + k3)*(k1**2 + k2**2 - k3**2)**4*(5*k1**2 - 9*k2**2 \
                + 9*k3**2)*(4 + n1)*(6 + n1) + (k1**2 + k2**2 - k3**2)**6 \
                * (k1**2 - k2**2 + k3**2)*(4 + n1)*(6 + n1)*(8 + n1))) \
                / (512.*k1**9*k2**8*k3*n1*(2 + n1)*(4 + n1)*(6 + n1)*(8 + n1) \
                * (10 + n1))
        elif n2 == 7 and n3 == 2:
            I = (-105*(5*k1**2 + 9*k2**2 - 9*k3**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4*n1 \
                + (k1**2 + k2**2 - k3**2)*n1*(2 + n1)*(420*(2*k1**2 + 3*k2**2 \
                - 3*k3**2)*(k2**2 - k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3 + 42*(k1**2 + k2**2 - k3**2)**2 \
                * (k1**4 + 2*k1**2*(k2**2 - k3**2) - 9*(k2**2 - k3**2)**2) \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 \
                * (4 + n1) - (k1**2 + k2**2 - k3**2)**4*(4 + n1)*(6 + n1) \
                * (8*k1**6*(7*k2**2 - 3*k3**2) + 16*k1**2*(k2**2 - k3**2)**2 \
                * (7*k2**2 + 2*k3**2) + (k2**2 - k3**2)**4*(-28 + n1) \
                + k1**8*n1 - 2*k1**4*(k2 - k3)*(k2 + k3)*(-(k3**2*(-10 + n1)) \
                + k2**2*(70 + n1)))))/(512.*k1**9*k2**7*k3**2*n1*(2 + n1) \
                * (4 + n1)*(6 + n1)*(8 + n1)*(10 + n1))
        elif n2 == 6 and n3 == 3:
            I = (315*(k1**2 + 3*k2**2 - 3*k3**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**4*n1 + n1*(2 + n1)*(60*(2*k1**6 \
                - 21*k1**2*(k2**2 - k3**2)**2 - 21*(k2**2 - k3**2)**3)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 + 18 \
                * (k1**2 + k2**2 - k3**2)**2*(k1**6 - 7*k1**4*(k2**2 - k3**2) \
                - 7*k1**2*(k2**2 - k3**2)**2 + 21*(k2**2 - k3**2)**3)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2*(4 + n1) \
                - (k1**2 + k2**2 - k3**2)**4*(k1**2 - k2**2 + k3**2)*(4 + n1) \
                * (6 + n1)*(24*k1**6*(k2 - k3)*(k2 + k3) + 48*k1**2*(2*k2**6 \
                - 3*k2**4*k3**2 + k3**6) + (k2**2 - k3**2)**4*(-28 + n1) \
                + k1**8*(8 + n1) - 2*k1**4*(k2 - k3)*(k2 + k3)*(-(k3**2*(2 \
                + n1)) + k2**2*(50 + n1)))))/(512.*k1**9*k2**6*k3**3*n1*(2 \
                + n1)*(4 + n1)*(6 + n1)*(8 + n1)*(10 + n1))
        elif n2 == 5 and n3 == 4:
            I = -0.00390625*((105*(k1**2 + 9*k2**2 - 9*k3**2)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4*n1)/2. + 30*(k1**6 \
                + 7*k1**4*(k2**2 - k3**2) - 7*k1**2*(k2**2 - k3**2)**2 - 21 \
                * (k2**2 - k3**2)**3)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2 \
                * (k2**2 + k3**2))**3*n1*(2 + n1) + (k1**2 + k2**2 - k3**2) \
                * n1*(2 + n1)*(4 + n1)*(3*(3*k1**8 + 12*k1**6*(k2**2 - k3**2) \
                - 42*k1**4*(k2**2 - k3**2)**2 - 28*k1**2*(k2**2 - k3**2)**3 \
                + 63*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2 \
                * (k2**2 + k3**2))**2 + 2*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 \
                - k2 + k3)*(k1 + k2 + k3)*(k1**4 + 2*k1**2*(k2 - k3)*(k2 + k3) \
                - 9*(k2**2 - k3**2)**2)*(k1**4 - (k2**2 - k3**2)**2)**2 \
                * (6 + n1) + ((k1**4 - (k2**2 - k3**2)**2)**4*(6 + n1) \
                * (8 + n1))/2.))/(k1**9*k2**5*k3**4*n1*(2 + n1)*(4 + n1) \
                * (6 + n1)*(8 + n1)*(10 + n1))
        elif n2 == 8 and n3 == 2:
            I = (-1890*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**5 + 210*(k1 + k2 - k3)**4*(k1 - k2 + k3)**4 \
                * (-k1 + k2 + k3)**4*(k1 + k2 + k3)**4*(k1**2 + 3*(k2 - k3) \
                * (k2 + k3))*(13*k1**2 + 15*(k2 - k3)*(k2 + k3))*(1 + n1) \
                + 420*(k1**2 + k2**2 - k3**2)**2*(k1**4 - 6*k1**2*(k2**2 \
                - k3**2) - 15*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3*(1 + n1)*(3 + n1) - 2*(k1**2 \
                + k2**2 - k3**2)**4*(1 + n1)*(3 + n1)*(5 + n1)*(42*(k1**4 \
                + 6*k1**2*(k2**2 - k3**2) - 15*(k2**2 - k3**2)**2)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 - (k1**2 \
                + k2**2 - k3**2)**2*(7 + n1)*(4*k1**6*(20*k2**2 - 7*k3**2) \
                + 36*k1**2*(k2**2 - k3**2)**2*(4*k2**2 + k3**2) + (k2**2 \
                - k3**2)**4*(-36 + n1) + k1**8*(-4 + n1) - 2*k1**4 \
                * (k2 - k3)*(k2 + k3)*(-(k3**2*(-16 + n1)) + k2**2 \
                * (92 + n1)))))/(2048.*k1**10*k2**8*k3**2*(1 + n1)*(3 + n1) \
                * (5 + n1)*(7 + n1)*(9 + n1)*(11 + n1))
        elif n2 == 7 and n3 == 3:
            I = -0.0009765625*(-945*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2 \
                * (k2**2 + k3**2))**5 + 315*(k1**4 + 12*k1**2*(k2**2 - k3**2) \
                + 15*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**4*(1 + n1) + (k1**2 + k2**2 \
                - k3**2)*(1 + n1)*(3 + n1)*(210*(k1**6 + 3*k1**4*(k2**2 \
                - k3**2) - 9*k1**2*(k2**2 - k3**2)**2 - 15*(k2**2 - k3**2)**3) \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 \
                + 42*(k1**2 + k2**2 - k3**2)**2*(k1**6 - 3*k1**4*(k2**2 \
                - k3**2) - 9*k1**2*(k2**2 - k3**2)**2 + 15*(k2**2 - k3**2)**3) \
                * (k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 \
                * (5 + n1) - (k1**2 + k2**2 - k3**2)**4*(k1**2 - k2**2 \
                + k3**2)*(5 + n1)*(7 + n1)*(6*k1**6*(7*k2**2 - 5*k3**2) \
                + 18*k1**2*(k2**2 - k3**2)**2*(7*k2**2 + 3*k3**2) \
                + (k2**2 - k3**2)**4*(-36 + n1) + k1**8*(6 + n1) - 2*k1**4 \
                * (k2 - k3)*(k2 + k3)*(-(k3**2*(-3 + n1)) + k2**2*(69 \
                + n1)))))/(k1**10*k2**7*k3**3*(1 + n1)*(3 + n1)*(5 + n1)*(7 \
                + n1)*(9 + n1)*(11 + n1))
        elif n2 == 6 and n3 == 4:
            I = -0.0009765625*(945*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2 \
                * (k2**2 + k3**2))**5 + 315*(k1**4 - 6*k1**2*(k2**2 - k3**2) \
                - 15*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**4*(1 + n1) + 30*(k1**8 - 28*k1**6 \
                * (k2**2 - k3**2) - 42*k1**4*(k2**2 - k3**2)**2 + 84*k1**2 \
                * (k2**2 - k3**2)**3 + 105*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3*(1 + n1)*(3 + n1) \
                - (k1**2 + k2**2 - k3**2)**2*(1 + n1)*(3 + n1)*(5 + n1) \
                * (6*(k1**8 + 28*k1**6*(k2**2 - k3**2) - 42*k1**4*(k2**2 \
                - k3**2)**2 - 84*k1**2*(k2**2 - k3**2)**3 + 105*(k2**2 \
                - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2 + 3*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 - k2 + k3) \
                * (k1 + k2 + k3)*(k1**4 + 6*k1**2*(k2 - k3)*(k2 + k3) \
                - 15*(k2**2 - k3**2)**2)*(k1**4 - (k2**2 - k3**2)**2)**2 \
                * (7 + n1) + (k1**4 - (k2**2 - k3**2)**2)**4*(7 + n1) \
                * (9 + n1)))/(k1**10*k2**6*k3**4*(1 + n1)*(3 + n1)*(5 + n1) \
                * (7 + n1)*(9 + n1)*(11 + n1))
        elif n2 == 5 and n3 == 5:
            I = (945*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**5 \
                + 525*(k1**4 - 9*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4*(1 + n1) + 150 \
                * (k1**8 - 14*k1**4*(k2**2 - k3**2)**2 + 21*(k2**2 \
                - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**3*(1 + n1)*(3 + n1) + (k1**4 - (k2**2 - k3**2)**2) \
                * (1 + n1)*(3 + n1)*(5 + n1)*(30*(k1**8 - 14*k1**4*(k2**2 \
                - k3**2)**2 + 21*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**2 + 5*(k1 - k2 - k3) \
                * (k1 + k2 - k3)*(k1 - k2 + k3)*(k1 + k2 + k3)*(k1**2 \
                - 3*k2**2 + 3*k3**2)*(k1**2 + 3*(k2 - k3)*(k2 + k3))*(k1**4 \
                - (k2**2 - k3**2)**2)**2*(7 + n1) + (k1**4 - (k2**2 \
                - k3**2)**2)**4*(7 + n1)*(9 + n1)))/(1024.*k1**10*k2**5*k3**5 \
                * (1 + n1)*(3 + n1)*(5 + n1)*(7 + n1)*(9 + n1)*(11 + n1))
        elif n2 == 7 and n3 == 4:
            I = (945*(3*k1**2 + 11*(k2**2 - k3**2))*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**5 + 105*(11*k1**6 \
                + 9*k1**4*(k2**2 - k3**2) - 135*k1**2*(k2**2 - k3**2)**2 \
                - 165*(k2**2 - k3**2)**3)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**4*(2 + n1) + (k1**2 + k2**2 \
                - k3**2)*(2 + n1)*(4 + n1)*(210*(k1**8 - 4*k1**6*(k2**2 \
                - k3**2) - 18*k1**4*(k2**2 - k3**2)**2 + 12*k1**2*(k2**2 \
                - k3**2)**3 + 33*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3 + (k1**2 + k2**2 \
                - k3**2)**2*(6 + n1)*(6*(3*k1**8 - 44*k1**6*(k2**2 - k3**2) \
                + 18*k1**4*(k2**2 - k3**2)**2 + 180*k1**2*(k2**2 - k3**2)**3 \
                - 165*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**2 - (k1**4 - (k2**2 \
                - k3**2)**2)**2*(8 + n1)*(4*k1**6*(7*k2**2 - 8*k3**2) \
                + 20*k1**2*(k2**2 - k3**2)**2*(7*k2**2 + 4*k3**2) \
                + (k2**2 - k3**2)**4*(-45 + n1) + k1**8*(11 + n1) - 2*k1**4 \
                * (k2 - k3)*(k2 + k3)*(-(k3**2*(7 + n1)) + k2**2*(67 \
                + n1))))))/(2048.*k1**11*k2**7*k3**4*(2 + n1)*(4 + n1) \
                * (6 + n1)*(8 + n1)*(10 + n1)*(12 + n1))
        elif n2 == 6 and n3 == 5:
            I = -0.00048828125*(945*(k1**2 + 11*(k2**2 - k3**2))*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**5 \
                + (2 + n1)*(525*(k1**6 + 9*k1**4*(k2**2 - k3**2) - 9*k1**2 \
                * (k2**2 - k3**2)**2 - 33*(k2**2 - k3**2)**3)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4 + 30*(5*k1**10 \
                + 35*k1**8*(k2**2 - k3**2) - 70*k1**6*(k2**2 - k3**2)**2 \
                - 210*k1**4*(k2**2 - k3**2)**3 + 105*k1**2*(k2**2 - k3**2)**4 \
                + 231*(k2**2 - k3**2)**5)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**3*(4 + n1) + (k1**2 + k2**2 \
                - k3**2)**2*(k1**2 - k2**2 + k3**2)*(4 + n1)*(6 + n1) \
                * (30*(k1**8 + 4*k1**6*(k2**2 - k3**2) - 18*k1**4*(k2**2 \
                - k3**2)**2 - 12*k1**2*(k2**2 - k3**2)**3 + 33*(k2**2 \
                - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**2 + 5*(k1 - k2 - k3)*(k1 + k2 - k3)*(k1 - k2 + k3) \
                * (k1 + k2 + k3)*(k1**4 + 2*k1**2*(k2 - k3)*(k2 + k3) \
                - 11*(k2**2 - k3**2)**2)*(k1**4 - (k2**2 - k3**2)**2)**2 \
                * (8 + n1) + (k1**4 - (k2**2 - k3**2)**2)**4*(8 + n1)*(10 \
                + n1))))/(k1**11*k2**6*k3**5*(2 + n1)*(4 + n1)*(6 + n1)*(8 \
                + n1)*(10 + n1)*(12 + n1))
        elif n2 == 8 and n3 == 4:
            I = (20790*(k1**4 + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 \
                + k3**2))**6 - 2*(1 + n1)*(1890*(k1**4 + 22*k1**2*(k2**2 \
                - k3**2) + 33*(k2**2 - k3**2)**2)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**5 + 105*(17*k1**8 + 108*k1**6 \
                * (k2**2 - k3**2) - 90*k1**4*(k2**2 - k3**2)**2 - 660*k1**2 \
                * (k2**2 - k3**2)**3 - 495*(k2**2 - k3**2)**4)*(k1**4 \
                + (k2**2 - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**4*(3 + n1) \
                + 420*(k1**2 + k2**2 - k3**2)**2*(k1**8 - 18*k1**4*(k2**2 \
                - k3**2)**2 + 33*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 \
                - k3**2)**2 - 2*k1**2*(k2**2 + k3**2))**3*(3 + n1)*(5 + n1) \
                + (k1**2 + k2**2 - k3**2)**4*(3 + n1)*(5 + n1)*(7 + n1) \
                * (3*(17*k1**8 - 108*k1**6*(k2**2 - k3**2) - 90*k1**4 \
                * (k2**2 - k3**2)**2 + 660*k1**2*(k2**2 - k3**2)**3 \
                - 495*(k2**2 - k3**2)**4)*(k1**4 + (k2**2 - k3**2)**2 \
                - 2*k1**2*(k2**2 + k3**2))**2 + 2*(k1 - k2 - k3)*(k1 + k2 \
                - k3)*(k1 - k2 + k3)*(k1 + k2 + k3)*(k1**4 - (k2**2 \
                - k3**2)**2)**2*(k1**4 + 33*(k2**2 - k3**2)**2 + 22*k1**2 \
                * (-k2**2 + k3**2))*(9 + n1) - (k1**4 - (k2**2 - k3**2)**2)**4 \
                * (9 + n1)*(11 + n1))))/(8192.*k1**12*k2**8*k3**4*(1 + n1) \
                * (3 + n1)*(5 + n1)*(7 + n1)*(9 + n1)*(11 + n1)*(13 + n1))
        else:
            print(n2,n3)
        return I

    def compute_kernels(self, tri):
        r"""Compute kernels for all triangle configurations.

        Computes the kernels for the various triangle configurations, by
        calling the corresponding class method (depending if the model is
        in real- or redshift-space), and stores them into a class attribute.
        """
        for kk in self.kernel_names:
            self.kernels[kk] = np.zeros([tri.shape[0],3])

        for i in range(3):
            k123_perm = np.roll(tri, -i, axis=1).T
            if self.real_space:
                kernels = self.kernels_real_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels[kk][:,i] = kernels[kk]
            else:
                kernels = self.kernels_redshift_space(*k123_perm)
                for kk in self.kernel_names:
                    self.kernels[kk][:,i] = kernels[kk]

    def Gauss_Legendre_mu123_integrals(self, tri, deg, max_ell):
        def muphi_to_mu123(mu_ij, phi_ij, k1, k2, k3):
            mu12 = (k3**2-k1**2-k2**2)/(2*k1*k2)
            mu1 = mu_ij
            dmu1sq = (1-mu1**2).clip(min=0.0)
            dmu1 = np.sqrt(dmu1sq)
            dmu12sq = (1-mu12**2).clip(min=0.0)
            dmu12 = np.sqrt(dmu12sq)
            mu2 = np.outer(mu1,mu12) - np.outer(dmu1*np.cos(phi_ij),dmu12)
            mu3 = -np.outer(mu1,k1/k3) - k2/k3*mu2
            return mu1.flatten(), mu2.T, mu3.T

        def I(n1, n2, n3, mu1, mu2, mu3):
            return mu1**n1 * mu2**n2 * mu3**n3

        ell_req = np.arange(0, max_ell+1, 2)

        # first, find all n1,n2,n3 tuples
        self.n123_tuples_all = np.array([0,0,0])
        kernel_names = ['F2','G2','k31','k32']
        self.I = {}
        for kk in kernel_names:
            self.I[kk] = {}
            n123_tuples = self.kernel_mu_tuples[kk]
            for n123 in n123_tuples:
                for i in range(3):
                    n123_new = np.copy(n123)
                    n123_new[i] += 2
                    n123_tuples = np.vstack((n123_tuples, n123_new))
            n123_tuples = np.unique(n123_tuples, axis=0)
            for n123 in n123_tuples:
                self.I[kk][tuple(n123)] = {}
                for ell in ell_req:
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        self.n123_tuples_all = np.vstack(
                            (self.n123_tuples_all, n123_perm_even))
        self.n123_tuples_all = np.unique(self.n123_tuples_all, axis=0)

        self.I_tuples_dict = {}
        for kk in self.I:
            self.I_tuples_dict[kk] = {}
            for n123 in self.I[kk]:
                self.I_tuples_dict[kk][n123] = {}
                for ell in ell_req:
                    self.I_tuples_dict[kk][n123][ell] = []
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        id = np.where(
                            (self.n123_tuples_all == n123_perm_even).all(
                                axis=1))[0][0]
                        self.I_tuples_dict[kk][n123][ell].append(id)

        #compute mu1, mu2, mu3 at Gauss-Legendre sampling points
        gl_x, gl_weights = np.polynomial.legendre.leggauss(deg)
        mu = gl_x
        phi = np.pi*gl_x + np.pi
        mu_ij, phi_ij = np.meshgrid(mu, phi)
        # shapes of mu1, mu2, mu3: deg^2, (ntri, deg^2), (ntri, deg^2)
        self.gl_mu1, self.gl_mu2, self.gl_mu3 = muphi_to_mu123(mu_ij, phi_ij,
                                                               *tri.T)
        self.gl_weights_ij = np.outer(gl_weights, gl_weights).reshape(
            [1, deg**2])

        # evaluate mu1^n1 mu2^n2 mu3^n3 for all n1, n2, n3
        self.gl_I_weights = np.zeros([self.n123_tuples_all.shape[0],
                                      tri.shape[0], deg**2])
        for i, n123 in enumerate(self.n123_tuples_all):
            self.gl_I_weights[i] = I(*n123, self.gl_mu1, self.gl_mu2,
                                     self.gl_mu3)
        self.gl_I_weights *= self.gl_weights_ij

        self.gl_I_stoch_weights = \
            np.zeros([3*self.n123_tuples_stoch_all.shape[0], 3,
                      tri.shape[0], deg**2])
        n = 0
        for n123 in self.n123_tuples_stoch_all:
            if np.all(n123 == [0,0,0]):
                for ell in ell_req:
                    n123_temp = np.copy(n123)
                    n123_temp[0] += ell
                    self.gl_I_stoch_weights[n,0] = I(*n123_temp, self.gl_mu1,
                                                     self.gl_mu2, self.gl_mu3)
                    self.gl_I_stoch_weights[n,1] = self.gl_I_stoch_weights[n,0]
                    self.gl_I_stoch_weights[n,2] = self.gl_I_stoch_weights[n,0]
                    n += 1
            else:
                for ell in ell_req:
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        self.gl_I_stoch_weights[n,i] = I(*n123_perm_even,
                            self.gl_mu1, self.gl_mu2, self.gl_mu3)
                    n += 1
        self.gl_I_stoch_weights *= self.gl_weights_ij

    def compute_mu123_integrals(self, tri, max_ell):
        """Compute angular integrals for all triangle configurations.

        Computes the angular integrals for the various triangle configurations,
        by calling the class method **mu123_integrals**, and stores them into
        a class attribute.
        """
        ell_req = np.arange(0, max_ell+1, 2)
        kernel_names = ['F2'] if self.real_space else ['F2','G2','k31','k32']
        self.I = {}
        for kk in kernel_names:
            self.I[kk] = {}
            n123_tuples = self.kernel_mu_tuples[kk]
            if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                # add tuples for bispectrum cnlo counterterm
                if self.cnlo_type == 'EggLeeSco':
                    for n123 in n123_tuples:
                        for i in range(3):
                            n123_new = np.copy(n123)
                            n123_new[i] += 2
                            n123_tuples = np.vstack((n123_tuples, n123_new))
                elif self.cnlo_type == 'IvaPhiNis':
                    max_mu = 2 if kk not in ['k31','k32'] else 3
                    n123_new = []
                    for i in range(2):
                        for n123 in [x for x in n123_tuples if x[i] < max_mu]:
                            n123_i = np.copy(n123)
                            for n in range(2):
                                n123_i[i] += 2
                                n123_new = np.vstack((n123_new, n123_i)) \
                                    if len(n123_new) else n123_i
                    if kk == 'k31':
                        n123_k4ctr = [(1,0,1), (1,2,1)]
                    elif kk == 'k32':
                        n123_k4ctr = [(0,1,1), (2,1,1)]
                    elif kk in ['F2','b2','K']:
                        n123_k4ctr = [(0,0,0)]
                    else:
                        n123_k4ctr = [(0,0,2)]
                    for i in range(2):
                        for j in range(2):
                            for n123 in n123_k4ctr:
                                n123_ij = np.copy(n123)
                                n123_ij[0] += 2*(i+1)
                                n123_ij[1] += 2*(j+1)
                                n123_new = np.vstack((n123_new, n123_ij))
                    n123_tuples = np.vstack((n123_tuples, n123_new))
            else:
                # add kernel needed for stochastic contributions
                if kk == 'F2':
                    n123_tuples = np.vstack((n123_tuples, [4,0,0]))
            n123_tuples = np.unique(n123_tuples, axis=0)
            for n123 in n123_tuples:
                for i in range(3):
                    n123_new = np.copy(n123)
                    n123_new[i] += 2
                    n123_tuples = np.vstack((n123_tuples, n123_new))
            n123_tuples = np.unique(n123_tuples, axis=0)
            for n123 in n123_tuples:
                self.I[kk][tuple(n123)] = {}
                for ell in ell_req:
                    self.I[kk][tuple(n123)][ell] = np.zeros([tri.shape[0],3])
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        ii = np.argsort(n123_perm_even)[::-1]
                        self.I[kk][tuple(n123)][ell][:,i] = \
                            self.mu123_integrals(*n123_perm_even[ii],
                                                 *tri[:,ii].T)

        self.I['b2'] = self.I['F2']
        self.I['K'] = self.I['F2']

        for n123 in self.n123_tuples_stoch_all:
            self.I_stoch[tuple(n123)] = {}
            for ell in ell_req:
                self.I_stoch[tuple(n123)][ell] = self.I['b2'][tuple(n123)][ell]
            if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                n123_ctr = np.copy(n123)
                n123_ctr[0] += 2
                self.I_stoch_ctr[tuple(n123_ctr)] = {}
                for ell in ell_req:
                    if tuple(n123_ctr) in [(6,0,2), (8,0,0)]:
                        self.I_stoch_ctr[tuple(n123_ctr)][ell] = \
                            np.zeros([tri.shape[0],3])
                        for i in range(3):
                            n123_perm_even = np.roll(np.array(n123_ctr), i)
                            n123_perm_even[0] += ell
                            ii = np.argsort(n123_perm_even)[::-1]
                            self.I_stoch_ctr[tuple(n123_ctr)][ell][:,i] = \
                                self.mu123_integrals(*n123_perm_even[ii],
                                                     *tri[:,ii].T)
                    else:
                        self.I_stoch_ctr[tuple(n123_ctr)][ell] = \
                            self.I['b2'][tuple(n123_ctr)][ell]

    def compute_damped_mu123_integrals(self, tri, W_damping, max_ell):
        ell_req = np.arange(0, max_ell+1, 2)
        gl_W3p_damping = W_damping(tri, self.gl_mu1, self.gl_mu2, self.gl_mu3)
        gl_W2p_damping = np.zeros((3, tri.shape[0],
                                   self.gl_mu1.shape[0], self.nparams))
        gl_W2p_damping[0] = W_damping(tri, self.gl_mu1, 0.0, 0.0)
        gl_W2p_damping[1] = W_damping(tri, 0.0, self.gl_mu2, 0.0)
        gl_W2p_damping[2] = W_damping(tri, 0.0, 0.0, self.gl_mu3)

        I_damped = 0.25 * np.einsum("abc,bc...->ab...",
                                    self.gl_I_weights, gl_W3p_damping)
        for kk in self.I_tuples_dict:
            for n123 in self.I_tuples_dict[kk]:
                for ell in ell_req:
                    ids = self.I_tuples_dict[kk][n123][ell]
                    self.I[kk][n123][ell] = np.swapaxes(I_damped[ids],0,1)

        self.I['b2'] = self.I['F2']
        self.I['K'] = self.I['F2']

        n = 0
        I_stoch_damped = 0.25 * np.einsum(
            "abcd,bcd...->abc...", self.gl_I_stoch_weights, gl_W2p_damping)
        for n123 in self.n123_tuples_stoch_all:
            self.I_stoch[tuple(n123)] = {}
            for ell in ell_req:
                self.I_stoch[tuple(n123)][ell] = np.swapaxes(
                    I_stoch_damped[n],0,1)
                n += 1

    def compute_kernels_shell_average(self, max_ell):
        # print('Compute shell averages.')
        ell_req = np.array([x for x in range(0,max_ell+1,2)])

        # first, find all n1,n2,n3 tuples
        self.n123_tuples_all = np.array([0,0,0])
        for kk in self.discrete_kernel_mu_tuples:
            self.kernels_shell_average[kk] = {}
            for n123 in self.discrete_kernel_mu_tuples[kk]:
                self.kernels_shell_average[kk][tuple(n123)] = {}
                for ell in ell_req:
                    self.kernels_shell_average[kk][tuple(n123)][ell] = \
                        np.zeros((self.tri.shape[0],3,self.num_fiducials))
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        self.n123_tuples_all = np.vstack(
                            (self.n123_tuples_all, n123_perm_even))

        for kk in self.discrete_stoch_kernel_mu_tuples:
            self.stoch_kernels_shell_average[kk] = {}
            for n123 in self.discrete_stoch_kernel_mu_tuples[kk]:
                self.stoch_kernels_shell_average[kk][tuple(n123)] = {}
                for ell in ell_req:
                    self.stoch_kernels_shell_average[kk][tuple(n123)][ell] = \
                        np.zeros((self.tri.shape[0],3,self.num_fiducials))
                    for i in range(3):
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        self.n123_tuples_all = np.vstack(
                            (self.n123_tuples_all, n123_perm_even))

        self.n123_tuples_all = np.unique(self.n123_tuples_all, axis=0)

        self.I_tuples_dict = {}
        for kk in self.kernels_shell_average:
            self.I_tuples_dict[kk] = {}
            for i in range(3):
                for n123 in self.kernels_shell_average[kk]:
                    if i == 0: self.I_tuples_dict[kk][n123] = {}
                    for ell in ell_req:
                        if i == 0: self.I_tuples_dict[kk][n123][ell] = []
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        id = np.where(
                            (self.n123_tuples_all == n123_perm_even).all(
                                axis=1))[0][0]
                        self.I_tuples_dict[kk][n123][ell].append(id)

        self.I_tuples_stoch_dict = {}
        for kk in self.stoch_kernels_shell_average:
            self.I_tuples_stoch_dict[kk] = {}
            for i in range(3):
                for n123 in self.stoch_kernels_shell_average[kk]:
                    if i == 0: self.I_tuples_stoch_dict[kk][n123] = {}
                    for ell in ell_req:
                        if i == 0: self.I_tuples_stoch_dict[kk][n123][ell] = []
                        n123_perm_even = np.roll(np.array(n123), i)
                        n123_perm_even[0] += ell
                        id = np.where(
                            (self.n123_tuples_all == n123_perm_even).all(
                                axis=1))[0][0]
                        self.I_tuples_stoch_dict[kk][n123][ell].append(id)

        @nb.njit(parallel=True)
        def _perform_I_computation(n123_tuples_all, mu1, mu2, mu3):
            I_all = np.zeros((len(mu1),len(n123_tuples_all)))
            for i in nb.prange(len(n123_tuples_all)):
                n1, n2, n3 = n123_tuples_all[i]
                I_all[:,i] = mu1**n1 * mu2**n2 * mu3**n3
            return I_all

        @nb.njit(parallel=True)
        def _perform_average(kernels, kernel_ids, I_all, I_ids,
                             weights, weights_sum, i_perm, num_fiducials):
            out = np.zeros((len(kernel_ids), num_fiducials))
            for j in nb.prange(num_fiducials):
                for i in nb.prange(len(kernel_ids)):
                    t = np.sum(kernels[:,kernel_ids[i],j] * I_all[:,I_ids[i]]
                               * weights)/weights_sum
                    out[i,j] = t
            return out

        # loop over triangle configurations
        for n,tri_id in enumerate(self.tri_ids_discrete_binning):
            id1 = self.grid.cum_num_tri_f[n]
            id2 = self.grid.cum_num_tri_f[n+1]
            wsum = np.sum(self.grid.weights[id1:id2])

            # compute I's for all tuples
            I_all = _perform_I_computation(self.n123_tuples_all,
                                           self.grid.kmu123[id1:id2,3],
                                           self.grid.kmu123[id1:id2,4],
                                           self.grid.kmu123[id1:id2,5])

            # loop over permutations
            for i_perm in range(3):
                # compute kernels (!! make sure to check order !!)
                if self.real_space:
                    kernels = self._kernels_real_space(
                        self.grid.kmu123[id1:id2,i_perm%3],
                        self.grid.kmu123[id1:id2,(i_perm+1)%3],
                        self.grid.kmu123[id1:id2,(i_perm+2)%3]
                    )
                else:
                    kernels = self._kernels_redshift_space(
                        self.grid.kmu123[id1:id2,i_perm%3],
                        self.grid.kmu123[id1:id2,(i_perm+1)%3],
                        self.grid.kmu123[id1:id2,(i_perm+2)%3]
                    )
                kernels = np.atleast_3d(
                    np.einsum("ij,i...->ij...", kernels,
                              self.fiducial_Pdw_sq[id1:id2,i_perm])
                )
                if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                    kernels_stoch = np.ones((id2-id1,3))
                    kernels_stoch[:,1] = \
                        self.grid.kmu123[id1:id2,i_perm]**self.pow_ctr
                    kernels_stoch[:,2] = self.pow_ctr * kernels_stoch[:,1]
                    kernels_stoch = np.atleast_3d(
                        np.einsum("ij,i...->ij...", kernels_stoch,
                                  self.fiducial_Pdw[id1:id2,i_perm])
                    )
                else:
                    kernels_stoch = np.atleast_2d(
                        self.fiducial_Pdw[id1:id2,i_perm].T).T[:,None,:]

                I_ids = []
                kernel_ids = []
                for i,kk in enumerate(self.kernels_shell_average):
                    for n123 in self.kernels_shell_average[kk]:
                        for ell in ell_req:
                            I_ids.append(
                                self.I_tuples_dict[kk][n123][ell][i_perm])
                            kernel_ids.append(i)
                I_ids = np.array(I_ids)
                kernel_ids = np.array(kernel_ids)

                I_stoch_ids = []
                kernel_stoch_ids = []
                for i,kk in enumerate(self.stoch_kernels_shell_average):
                    for n123 in self.stoch_kernels_shell_average[kk]:
                        for ell in ell_req:
                            I_stoch_ids.append(
                                self.I_tuples_stoch_dict[kk][n123][ell][i_perm])
                            kernel_stoch_ids.append(i)
                I_stoch_ids = np.array(I_stoch_ids)
                kernel_stoch_ids = np.array(kernel_stoch_ids)

                avg = _perform_average(kernels, kernel_ids, I_all, I_ids,
                                       self.grid.weights[id1:id2], wsum, i_perm,
                                       self.num_fiducials)
                avg_stoch = _perform_average(kernels_stoch, kernel_stoch_ids,
                                             I_all, I_stoch_ids,
                                             self.grid.weights[id1:id2],
                                             wsum, i_perm, self.num_fiducials)

                count = 0
                for kk in self.kernels_shell_average:
                    for n123 in self.kernels_shell_average[kk]:
                        for ell in ell_req:
                            self.kernels_shell_average[kk][n123][ell] \
                                [tri_id,i_perm] = avg[count]
                            count += 1
                count = 0
                for kk in self.stoch_kernels_shell_average:
                    for n123 in self.stoch_kernels_shell_average[kk]:
                        for ell in ell_req:
                            self.stoch_kernels_shell_average[kk][n123][ell] \
                                [tri_id,i_perm] = avg_stoch[count]
                            count += 1

        for n,tri_id in enumerate(self.tri_ids_eff):
            P2 = np.zeros((3,self.num_fiducials))
            for i in range(3):
                i1 = self.tri_eff_to_id[tri_id][i%3]
                i2 = self.tri_eff_to_id[tri_id][(i+1)%3]
                P2[i] = self.fiducial_Pdw_eff[i1]*self.fiducial_Pdw_eff[i2]
            P = np.atleast_2d(
                self.fiducial_Pdw_eff[self.tri_eff_to_id[tri_id]].T).T
            for kk in self.kernels_shell_average:
                kk_bare = [x for x in self.I.keys() if x in kk][0]
                for n123 in self.kernels_shell_average[kk]:
                    for ell in ell_req:
                        self.kernels_shell_average[kk][n123][ell][tri_id] = \
                            np.einsum(
                                "a,ab->ab", self.kernels[kk][n] \
                                * self.I[kk_bare][n123][ell][n], P2)
            for kk in self.stoch_kernels_shell_average:
                for n123 in self.stoch_kernels_shell_average[kk]:
                    for ell in ell_req:
                        if kk == 'id':
                            self.stoch_kernels_shell_average[kk][n123][ell] \
                                                            [tri_id] = \
                                np.einsum("a,a...->a...",
                                          self.I_stoch[n123][ell][n], P)
                        elif kk == 'ksq':
                            self.stoch_kernels_shell_average[kk][n123][ell] \
                                                            [tri_id] = \
                                np.einsum("a,a...->a...",
                                          self.I_stoch_ctr[n123][ell][n] \
                                          * self.kernels['k1sqb2'][n], P)
                        elif kk == 'dksq_dlnk':
                            self.stoch_kernels_shell_average[kk][n123][ell] \
                                                            [tri_id] = \
                                np.einsum("a,a...->a...",
                                          self.I_stoch_ctr[n123][ell][n] \
                                          * self.kernels['dk1sqb2_dlnk1'][n], P)

        # dump kernels
        if self.binning.get('filename_root_kernels'):
            fname = '{}.pickle'.format(
                self.binning.get('filename_root_kernels'))
            fname_stoch = '{}_stoch.pickle'.format(
                self.binning.get('filename_root_kernels'))
            with open(fname, "wb") as f:
                pickle.dump(self.kernels_shell_average, f)
            with open(fname_stoch, "wb") as f:
                pickle.dump(self.stoch_kernels_shell_average, f)

    def load_kernels_shell_average(self):
        # print('Load (binned) kernels!')
        self.kernels_shell_average = pickle.load(
            open('{}.pickle'.format(self.binning.get('filename_root_kernels')),
            "rb")
        )
        self.stoch_kernels_shell_average = pickle.load(
            open('{}_stoch.pickle'.format(
                self.binning.get('filename_root_kernels')),
            "rb")
        )

    def compute_covariance_mixing_kernel(self, l1, l2, l3, l4, l5):
        def legendre_coeff(ell, n):
            ln = np.math.factorial(ell-n)
            ln2 = np.math.factorial(ell-2*n)
            l2n2 = np.math.factorial(2*ell-2*n)
            return (-1)**n*l2n2/(ln * ln2 * np.math.factorial(n))/2**ell

        id_eq_k1k2 = np.where(self.tri[:,0] == self.tri[:,1])
        id_eq_k2k3 = np.where(self.tri[:,1] == self.tri[:,2])
        id_eq_k1k3 = np.where(self.tri[:,0] == self.tri[:,2])
        deltaK_k1k2 = np.zeros(self.tri.shape[0])
        deltaK_k2k3 = np.zeros(self.tri.shape[0])
        deltaK_k1k3 = np.zeros(self.tri.shape[0])
        deltaK_k1k2[id_eq_k1k2] = 1.0
        deltaK_k2k3[id_eq_k2k3] = 1.0
        deltaK_k1k3[id_eq_k1k3] = 1.0

        ell_tuple = (l1,l2,l3,l4,l5)

        self.cov_mixing_kernel[ell_tuple] = np.zeros(self.tri.shape[0])
        for n1 in range(int(l1/2)+1):
            C1 = legendre_coeff(l1, n1)
            for n2 in range(int(l2/2)+1):
                C2 = legendre_coeff(l2, n2)
                for n3 in range(int(l3/2)+1):
                    C3 = legendre_coeff(l3, n3)
                    for n4 in range(int(l4/2)+1):
                        C4 = legendre_coeff(l4, n4)
                        for n5 in range(int(l5/2)+1):
                            C5 = legendre_coeff(l5, n5)
                            m1 = np.array([l1+l2+l3-2*(n1+n2+n3), l4-2*n4,
                                           l5-2*n5])
                            m2 = np.array([l1+l3-2*(n1+n3), l2+l4-2*(n2+n4),
                                           l5-2*n5])
                            m3 = np.array([l1+l3-2*(n1+n3), l4-2*n4,
                                           l2+l5-2*(n2+n5)])
                            if m1[2] <= m1[1]:
                                I1 = self.mu123_integrals(*m1, *self.tri.T)
                            else:
                                I1 = self.mu123_integrals(*m1[[0,2,1]],
                                    *self.tri[:,[0,2,1]].T)
                            if m2[2] <= m2[1]:
                                I2 = self.mu123_integrals(*m2, *self.tri.T)
                            else:
                                I2 = self.mu123_integrals(*m2[[0,2,1]],
                                    *self.tri[:,[0,2,1]].T)
                            if m3[2] <= m3[1]:
                                I3 = self.mu123_integrals(*m3, *self.tri.T)
                            else:
                                I3 = self.mu123_integrals(*m3[[0,2,1]],
                                    *self.tri[:,[0,2,1]].T)
                            self.cov_mixing_kernel[ell_tuple] += \
                                C1 * C2 * C3 * C4 * C5 * ((1. + \
                                deltaK_k2k3)*I1 + (deltaK_k1k2 + \
                                deltaK_k2k3)*I2 + 2*deltaK_k1k3*I3)

    def generate_index_arrays(self, round_decimals=2):
        r"""Generate arrays of indeces of triangular configurations.

        Determines the unique wavemode bins in the triangle configurations,
        approximating them to the ratio with respect to a given fundamental
        frequency.

        Parameters
        ----------
        round_decimals: int, optional
            Number of decimal digits used in the approximation of the ratios
            with respect to the fundamental frequency. Deafults to 2.
        """
        self.tri_rounded = np.around(self.tri/self.kfun,
                                     decimals=round_decimals)
        self.tri_unique = np.unique(self.tri_rounded)
        self.ki, self.kj = np.meshgrid(self.tri_unique,
                                       self.tri_unique)
        ids = np.where(self.ki >= self.kj)
        self.ki = self.ki[ids]
        self.kj = self.kj[ids]

        self.tri_to_id = np.zeros_like(self.tri, dtype=int)
        self.tri_to_id_sq = np.zeros_like(self.tri, dtype=int)

        #define jitted function for better performance
        @nb.njit(parallel=True)
        def get_tri_to_id(tri_to_id, tri_to_id_sq, tri_unique,
                          tri_rounded, ki, kj):
            for n in nb.prange(tri_rounded.shape[0]):
                idi = [0,1,0]
                idj = [1,2,2]
                for d in nb.prange(3):
                    tri_to_id[n,d] = np.where(
                        tri_unique == tri_rounded[n,d])[0][0]
                    tri_to_id_sq[n,d] = np.where(
                        (ki == tri_rounded[n,idi[d]]) & \
                        (kj == tri_rounded[n,idj[d]]))[0][0]

        get_tri_to_id(self.tri_to_id, self.tri_to_id_sq, self.tri_unique,
                      self.tri_rounded, self.ki, self.kj)

        # for n in range(self.tri.shape[0]):
        #     self.tri_to_id[n,0] = np.where(
        #         self.tri_unique == self.tri_rounded[n,0])[0]
        #     self.tri_to_id[n,1] = np.where(
        #         self.tri_unique == self.tri_rounded[n,1])[0]
        #     self.tri_to_id[n,2] = np.where(
        #         self.tri_unique == self.tri_rounded[n,2])[0]
        #     self.tri_to_id_sq[n,0] = np.where(
        #         (self.ki == self.tri_rounded[n,0]) & \
        #         (self.kj == self.tri_rounded[n,1]))[0]
        #     self.tri_to_id_sq[n,1] = np.where(
        #         (self.ki == self.tri_rounded[n,1]) & \
        #         (self.kj == self.tri_rounded[n,2]))[0]
        #     self.tri_to_id_sq[n,2] = np.where(
        #         (self.ki == self.tri_rounded[n,0]) & \
        #         (self.kj == self.tri_rounded[n,2]))[0]

        self.ki = np.searchsorted(self.tri_unique, self.ki)
        self.kj = np.searchsorted(self.tri_unique, self.kj)
        self.tri_unique *= self.kfun

    def generate_eff_index_arrays(self, round_decimals=2):
        r"""Generate arrays of indeces of effective triangular configurations.

        Determines the unique wavemode bins in the effective triangle
        configurations, approximating them to the ratio with respect to a given
        fundamental frequency.

        Parameters
        ----------
        round_decimals: int, optional
            Number of decimal digits used in the approximation of the ratios
            with respect to the fundamental frequency. Deafults to 2.
        """
        self.tri_eff_rounded = np.around(self.tri_eff/self.kfun,
                                         decimals=round_decimals)
        self.tri_eff_unique = np.unique(self.tri_eff_rounded)
        self.ki_eff, self.kj_eff = np.meshgrid(self.tri_eff_unique,
                                               self.tri_eff_unique)
        ids = np.where(self.ki_eff >= self.kj_eff)
        self.ki_eff = self.ki_eff[ids]
        self.kj_eff = self.kj_eff[ids]

        self.tri_eff_to_id = np.zeros_like(self.tri_eff, dtype=int)
        self.tri_eff_to_id_sq = np.zeros_like(self.tri_eff, dtype=int)

        #define jitted function for better performance
        @nb.njit(parallel=True)
        def get_tri_to_id(tri_to_id, tri_to_id_sq, tri_unique,
                          tri_rounded, ki, kj):
            for n in nb.prange(tri_rounded.shape[0]):
                idi = [0,1,0]
                idj = [1,2,2]
                for d in nb.prange(3):
                    tri_to_id[n,d] = np.where(
                        tri_unique == tri_rounded[n,d])[0][0]
                    tri_to_id_sq[n,d] = np.where(
                        (ki == tri_rounded[n,idi[d]]) & \
                        (kj == tri_rounded[n,idj[d]]))[0][0]

        get_tri_to_id(self.tri_eff_to_id, self.tri_eff_to_id_sq,
                      self.tri_eff_unique, self.tri_eff_rounded,
                      self.ki_eff, self.kj_eff)

        self.ki_eff = np.searchsorted(self.tri_eff_unique, self.ki_eff)
        self.kj_eff = np.searchsorted(self.tri_eff_unique, self.kj_eff)
        self.tri_eff_unique *= self.kfun

    def join_kernel_mu123_integral(self, K, n123_tuples, ell, neff, coeff,
                                   q_tr, q_lo, cnloB=None):
        def get_aux_kernels(K, get_K_deriv_sum=True):
            K_neff1 = np.einsum("abc,ab->abc", neff[self.tri_to_id],
                                self.kernels[K])
            K_neff2 = np.einsum("abc,ab->abc",neff[self.tri_to_id[:,[1,2,0]]],
                                self.kernels[K])
            if get_K_deriv_sum:
                K_deriv_sum = 0.0
                for i in range(3):
                    K_deriv_sum += self.kernels['d{}_dlnk{}'.format(K,i+1)]
                return K_neff1, K_neff2, K_deriv_sum
            else:
                return K_neff1, K_neff2

        def add_product(var, IK, n123, KK, kernel_neff1, kernel_neff2,
                        kernel_deriv_sum, coeff):
            for i in range(self.nparams):
                idI = (Ellipsis,i) if 'VDG_infty' in self.model \
                      else Ellipsis
                t1 = self.I[IK][n123][ell][idI] \
                     * ((1.0 + (q_tr[i]-q_lo[i])*sum(n123)) * self.kernels[KK] \
                        + (1.0-q_tr[i]) * kernel_deriv_sum \
                        + (1.0-q_tr[i]) * (kernel_neff1[...,i] \
                                           + kernel_neff2[...,i]))
                t2 = self.I[IK][n123[0]+2,n123[1],n123[2]][ell][idI] \
                     * (q_tr[i] - q_lo[i]) \
                     * (self.kernels['d{}_dlnk1'.format(KK)] \
                        + kernel_neff1[...,i] - n123[0]*self.kernels[KK])
                t3 = self.I[IK][n123[0],n123[1]+2,n123[2]][ell][idI] \
                     * (q_tr[i] - q_lo[i]) \
                     * (self.kernels['d{}_dlnk2'.format(KK)] \
                        + kernel_neff2[...,i] - n123[1]*self.kernels[KK])
                t4 = self.I[IK][n123[0],n123[1],n123[2]+2][ell][idI] \
                     * (q_tr[i] - q_lo[i]) \
                     * (self.kernels['d{}_dlnk3'.format(KK)] \
                        - n123[2]*self.kernels[KK])
                var[...,i] += coeff[i] * (t1+t2+t3+t4)

        K_neff1, K_neff2 = get_aux_kernels(K, False)
        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            Kctr_neff1 = np.zeros((3,self.tri.shape[0],3,self.nparams))
            Kctr_neff2 = np.zeros((3,self.tri.shape[0],3,self.nparams))
            Kctr_deriv_sum = np.zeros((3,self.tri.shape[0],3))
            for i in range(3):
                Kctr_neff1[i], Kctr_neff2[i], Kctr_deriv_sum[i] \
                    = get_aux_kernels('k{}sq{}'.format(i+1,K))
            if self.cnlo_type == 'IvaPhiNis':
                Kctr_k4_neff1, Kctr_k4_neff2, Kctr_k4_deriv_sum \
                    = get_aux_kernels('k1sqk2sq{}'.format(K))

        DeltaB_K = np.zeros(self.kernels['F2'].shape + (self.nparams,))
        for n, n123 in enumerate(n123_tuples):
            add_product(DeltaB_K, K, n123, K, K_neff1, K_neff2, 0.0, coeff[n])

            if ('EFT' in self.model or 'VDG_infty_ctr' in self.model) \
                    and self.cnlo_type == 'EggLeeSco':
                for j in range(3):
                    Kctr = 'k{}sq{}'.format(j+1,K)
                    n123_j = np.copy(n123)
                    n123_j[j] += 2
                    add_product(DeltaB_K, K, tuple(n123_j), Kctr, Kctr_neff1[j],
                                Kctr_neff2[j], Kctr_deriv_sum[j],
                                coeff[n]*cnloB)
            elif ('EFT' in self.model or 'VDG_infty_ctr' in self.model)\
                    and self.cnlo_type == 'IvaPhiNis':
                # k^2 counterterms
                for j in range(2):
                    if (K not in ['k31','k32'] and n123[j] < 2) \
                            or (K in ['k31','k32'] and n123[j] < 3):
                        Kctr = 'k{}sq{}'.format(j+1,K)
                        n123_j = np.copy(n123)
                        for i in range(2):
                            n123_j[j] += 2
                            split_factor = 0.5 if n123[j] >= 2 else 1.0
                            add_product(DeltaB_K, K, tuple(n123_j), Kctr,
                                        Kctr_neff1[j], Kctr_neff2[j],
                                        Kctr_deriv_sum[j],
                                        coeff[n]*cnloB[i]*split_factor)
                # k^4 counterterms
                if (K == 'k31' and n123 in [(1,0,1),(1,2,1)]) \
                        or (K == 'k32' and n123 in [(0,1,1),(2,1,1)]) \
                        or (K in ['F2','b2','K'] and n123 == (0,0,0)) \
                        or (K == 'G2' and n123 == (0,0,2)):
                    split_factor = 0.5 if n123 in [(1,2,1),(2,1,1)] else 1.0
                    for i in range(2):
                        for j in range(2):
                            Kctr_k4 = 'k1sqk2sq{}'.format(K)
                            n123_ij = np.copy(n123)
                            n123_ij[0] += 2*(i+1)
                            n123_ij[1] += 2*(j+1)
                            add_product(DeltaB_K, K, tuple(n123_ij), Kctr_k4,
                                        Kctr_k4_neff1, Kctr_k4_neff2,
                                        Kctr_k4_deriv_sum,
                                        coeff[n]*cnloB[i]*cnloB[j]*split_factor)

        return DeltaB_K

    def join_kernel_mu123_integral_indv(self, K, n123_tuples, ell, neff, coeff,
                                        q_tr, q_lo):
        K_neff1 = neff[self.tri_to_id]*self.kernels[K]
        K_neff2 = neff[self.tri_to_id[:,[1,2,0]]]*self.kernels[K]
        K_deriv_sum = 0.0
        for i in range(3):
            K_deriv_sum += self.kernels['d{}_dlnk{}'.format(K,i+1)]

        if 'EFT' in self.model:
            Kctr_neff1 = np.zeros((3,self.tri.shape[0],3))
            Kctr_neff2 = np.zeros((3,self.tri.shape[0],3))
            Kctr_deriv_sum = np.zeros((3,self.tri.shape[0],3))
            for i in range(3):
                Kctr = 'k{}sq{}'.format(i+1,K)
                Kctr_neff1[i] = neff[self.tri_to_id]*self.kernels[Kctr]
                Kctr_neff2[i] = neff[self.tri_to_id[:,[1,2,0]]] \
                                * self.kernels[Kctr]
                for j in range(3):
                    Kctr_deriv_sum[i] += \
                        self.kernels['d{}_dlnk{}'.format(Kctr,j+1)]

        if 'EFT' in self.model:
            DeltaB_K = np.zeros([2*len(n123_tuples),
                                 self.kernels['F2'].shape[0],
                                 self.kernels['F2'].shape[1]])
        else:
            DeltaB_K = np.zeros([len(n123_tuples),
                                 self.kernels['F2'].shape[0],
                                 self.kernels['F2'].shape[1]])
        for i, n123 in enumerate(n123_tuples):
            t1 = self.I[K][n123][ell] * ((1.0 + (q_tr-q_lo)*sum(n123)) * \
                                         self.kernels[K] \
                                         + (1.0-q_tr) * K_deriv_sum \
                                         + (1.0-q_tr) * (K_neff1 + K_neff2))
            t2 = self.I[K][n123[0]+2,n123[1],n123[2]][ell] * (q_tr - q_lo) \
                 * (self.kernels['d{}_dlnk1'.format(K)] + K_neff1 \
                    - n123[0]*self.kernels[K])
            t3 = self.I[K][n123[0],n123[1]+2,n123[2]][ell] * (q_tr - q_lo) \
                 * (self.kernels['d{}_dlnk2'.format(K)] + K_neff2 \
                    - n123[1]*self.kernels[K])
            t4 = self.I[K][n123[0],n123[1],n123[2]+2][ell] * (q_tr - q_lo) \
                 * (self.kernels['d{}_dlnk3'.format(K)] \
                    - n123[2]*self.kernels[K])

            DeltaB_K[i] = coeff[i] * (t1 + t2 + t3 + t4)

            if 'EFT' in self.model:
                for j in range(3):
                    Kctr = 'k{}sq{}'.format(j+1,K)
                    n123_j = np.copy(n123)
                    n123_j[j] += 2
                    tctr1 = self.I[K][tuple(n123_j)][ell] \
                        * ((1.0 + (q_tr-q_lo)*sum(n123_j))*self.kernels[Kctr] \
                           + (1.0-q_tr) * Kctr_deriv_sum[j] \
                           + (1.0-q_tr) * (Kctr_neff1[j] + Kctr_neff2[j]))
                    tctr2 = self.I[K][n123_j[0]+2,n123_j[1],n123_j[2]][ell] \
                        * (q_tr - q_lo) \
                        * (self.kernels['d{}_dlnk1'.format(Kctr)] \
                           + Kctr_neff1[j] - n123_j[0]*self.kernels[Kctr])
                    tctr3 = self.I[K][n123_j[0],n123_j[1]+2,n123_j[2]][ell] \
                        * (q_tr - q_lo) \
                        * (self.kernels['d{}_dlnk2'.format(Kctr)] \
                           + Kctr_neff2[j] - n123_j[1]*self.kernels[Kctr])
                    tctr4 = self.I[K][n123_j[0],n123_j[1],n123_j[2]+2][ell] \
                        * (q_tr - q_lo) \
                        * (self.kernels['d{}_dlnk3'.format(Kctr)] \
                           - n123_j[2]*self.kernels[Kctr])

                    DeltaB_K[i+len(n123_tuples)] += coeff[i] \
                        * (tctr1 + tctr2 + tctr3 + tctr4)

        return DeltaB_K

    def join_stoch_kernel_mu123_integral(self, n123_tuples, ell, neff, coeff,
                                         q_tr, q_lo, cnloB_stoch=0):
        def add_product(var, I, n123, kernel, kernel_neff, kernel_deriv, coeff):
            for i in range(self.nparams):
                idI = (Ellipsis,i) if 'VDG_infty' in self.model \
                      else Ellipsis
                t1 = I[n123][ell][idI] \
                     * ((1.0 + (q_tr[i]-q_lo[i])*sum(n123))*kernel \
                        + (1.0-q_tr[i])*kernel_neff[...,i] \
                        + (1.0-q_tr[i])*kernel_deriv)
                t2 = I[n123[0]+2,n123[1],n123[2]][ell][idI] \
                     * (q_tr[i] - q_lo[i]) \
                     * (kernel_deriv + kernel_neff[...,i] - n123[0]*kernel)
                t3 = - I[n123[0],n123[1]+2,n123[2]][ell][idI] \
                     * (q_tr[i] - q_lo[i]) * n123[1] * kernel
                t4 = - I[n123[0],n123[1],n123[2]+2][ell][idI] \
                     * (q_tr[i] - q_lo[i]) * n123[2] * kernel
                var[...,i] += coeff[i] * (t1+t2+t3+t4)

        if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
            Kctr = 'k1sqb2'
            Kctr_neff = np.einsum("abc,ab->abc", neff[self.tri_to_id],
                                  self.kernels[Kctr])

        DeltaB_stoch = np.zeros(self.kernels['F2'].shape + (self.nparams,))
        for i, n123 in enumerate(n123_tuples):
            add_product(DeltaB_stoch, self.I_stoch, n123,
                        1.0, neff[self.tri_to_id], 0.0, coeff[i])
            if 'VDG_infty_ctr' in self.model \
                    or ('EFT' in self.model and self.cnlo_type == 'EggLeeSco'):
                n123_ctr = np.copy(n123)
                n123_ctr[0] += 2
                add_product(DeltaB_stoch, self.I_stoch_ctr, tuple(n123_ctr),
                            self.kernels[Kctr], Kctr_neff,
                            self.kernels['d{}_dlnk1'.format(Kctr)],
                            coeff[i] * cnloB_stoch)
            elif 'EFT' in self.model and self.cnlo_type == 'IvaPhiNis':
                if n123 == (0,0,0):
                    n123_ctr = np.copy(n123)
                    for n in range(2):
                        n123_ctr[0] += 2
                        add_product(DeltaB_stoch, self.I_stoch_ctr,
                                    tuple(n123_ctr), self.kernels[Kctr],
                                    Kctr_neff,
                                    self.kernels['d{}_dlnk1'.format(Kctr)],
                                    coeff[i] * cnloB_stoch[n])

        return DeltaB_stoch

    def join_stoch_kernel_mu123_integral_indv(self, n123_tuples, ell, neff,
                                              coeff, q_tr, q_lo):
        DeltaB_stoch = np.zeros([len(n123_tuples), self.kernels['F2'].shape[0],
                                 self.kernels['F2'].shape[1]])
        for i, n123 in enumerate(n123_tuples):
            t1 = self.I_stoch[n123][ell] * (1.0 + (q_tr-q_lo)*sum(n123) \
                                            + (1.0-q_tr)*neff[self.tri_to_id])
            t2 = self.I_stoch[n123[0]+2,n123[1],n123[2]][ell] * (q_tr - q_lo) \
                 * (neff[self.tri_to_id] - n123[0])
            t3 = - self.I_stoch[n123[0],n123[1]+2,n123[2]][ell] \
                 * (q_tr - q_lo) * n123[1]
            t4 = - self.I_stoch[n123[0],n123[1],n123[2]+2][ell] \
                 * (q_tr - q_lo) * n123[2]
            DeltaB_stoch[i] = coeff[i] * (t1 + t2 + t3 + t4)

        return DeltaB_stoch

    def join_kernel_mu123_shell_average(self, K, n123_tuples, ell, neff, coeff,
                                        q_tr, q_lo, cnloB=None):
        def add_product(var, n123, KK, neff1, neff2, kernel_deriv_sum, coeff):
            n123p200 = tuple(np.array(n123)+np.array((2,0,0)))
            n123p020 = tuple(np.array(n123)+np.array((0,2,0)))
            n123p002 = tuple(np.array(n123)+np.array((0,0,2)))
            KK1 = 'd{}_dlnk1'.format(KK)
            KK2 = 'd{}_dlnk2'.format(KK)
            KK3 = 'd{}_dlnk3'.format(KK)
            for i in range(self.nparams):
                t1 = (1.0 + (q_tr[i] - q_lo[i])*sum(n123)) \
                     * self.kernels_shell_average[KK][n123][ell][...,i] \
                     + (1.0 - q_tr[i]) * kernel_deriv_sum[...,i] \
                     + (1.0 - q_tr[i]) * (neff1[...,i] + neff2[...,i]) \
                     * self.kernels_shell_average[KK][n123][ell][...,i]
                t2 = (q_tr[i] - q_lo[i]) \
                     * (self.kernels_shell_average[KK1][n123p200][ell][...,i] \
                        + (neff1[...,i]-n123[0]) \
                        * self.kernels_shell_average[KK][n123p200][ell][...,i])
                t3 = (q_tr[i] - q_lo[i]) \
                     * (self.kernels_shell_average[KK2][n123p020][ell][...,i] \
                        + (neff2[...,i]-n123[1]) \
                        * self.kernels_shell_average[KK][n123p020][ell][...,i])
                t4 = (q_tr[i] - q_lo[i]) \
                     * (self.kernels_shell_average[KK3][n123p002][ell][...,i] \
                        - self.kernels_shell_average[KK][n123p002][ell][...,i] \
                        * n123[2])
                var[...,i] += coeff[i] * (t1+t2+t3+t4)

        DeltaB_K = np.zeros_like(self.kernels_shell_average['F2'][0,0,0][0])
        for i, n123 in enumerate(n123_tuples):
            neff1 = neff[self.tri_eff_to_id]
            neff2 = neff[self.tri_eff_to_id[:,[1,2,0]]]
            if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                if self.cnlo_type == 'EggLeeSco':
                    Kctr_deriv_sum = np.zeros(
                        (3,self.tri.shape[0],3,self.nparams))
                    for j in range(3):
                        Kctr = 'k{}sq{}'.format(j+1,K)
                        n123_j = np.copy(n123)
                        n123_j[j] += 2
                        for k in range(3):
                            Kctr_deriv_sum[j] += \
                                self.kernels_shell_average[
                                    'd{}_dlnk{}'.format(
                                        Kctr,k+1)][tuple(n123_j)][ell]
                elif self.cnlo_type == 'IvaPhiNis':
                    Kctr_deriv_sum = np.zeros(
                        (4,self.tri.shape[0],3,self.nparams))
                    for j in range(2):
                        if (K not in ['k31','k32'] and n123[j] < 2) \
                                or (K in ['k31','k32'] and n123[j] < 3):
                            Kctr = 'k{}sq{}'.format(j+1,K)
                            n123_j = np.copy(n123)
                            for n in range(2):
                                n123_j[j] += 2
                                for k in range(3):
                                    Kctr_deriv_sum[2*j+n] += \
                                        self.kernels_shell_average[
                                            'd{}_dlnk{}'.format(
                                                Kctr,k+1)][tuple(n123_j)][ell]
                    if (K == 'k31' and n123 in [(1,0,1),(1,2,1)]) \
                            or (K == 'k32' and n123 in [(0,1,1),(2,1,1)]) \
                            or (K in ['F2','b2','K'] and n123 == (0,0,0)) \
                            or (K == 'G2' and n123 == (0,0,2)):
                        Kctr_k4_deriv_sum = np.zeros(
                            (4,self.tri.shape[0],3,self.nparams))
                        Kctr = 'k1sqk2sq{}'.format(K)
                        for j in range(2):
                            for n in range(2):
                                n123_jn = np.copy(n123)
                                n123_jn[0] += 2*(j+1)
                                n123_jn[1] += 2*(n+1)
                                for k in range(3):
                                    Kctr_k4_deriv_sum[2*j+n] += \
                                        self.kernels_shell_average[
                                            'd{}_dlnk{}'.format(
                                                Kctr,k+1)][tuple(n123_jn)][ell]

            add_product(DeltaB_K, n123, K, neff1, neff2,
                        np.zeros(self.nparams), coeff[i])

            # add bispectrum counterterms
            if ('EFT' in self.model or 'VDG_infty_ctr' in self.model) \
                    and self.cnlo_type == 'EggLeeSco':
                for j in range(3):
                    Kctr = 'k{}sq{}'.format(j+1,K)
                    n123_j = np.copy(n123)
                    n123_j[j] += 2
                    add_product(DeltaB_K, tuple(n123_j), Kctr, neff1, neff2,
                                Kctr_deriv_sum[j], coeff[i]*cnloB)
            elif ('EFT' in self.model or 'VDG_infty_ctr' in self.model)\
                    and self.cnlo_type == 'IvaPhiNis':
                # k^2 counterterms
                for j in range(2):
                    if (K not in ['k31','k32'] and n123[j] < 2) \
                            or (K in ['k31','k32'] and n123[j] < 3):
                        Kctr = 'k{}sq{}'.format(j+1,K)
                        n123_j = np.copy(n123)
                        for n in range(2):
                            n123_j[j] += 2
                            split_factor = 0.5 if n123[j] >= 2 else 1.0
                            add_product(DeltaB_K, tuple(n123_j), Kctr,
                                        neff1, neff2, Kctr_deriv_sum[2*j+n],
                                        coeff[i]*cnloB[n]*split_factor)
                # k^4 counterterms
                if (K == 'k31' and n123 in [(1,0,1),(1,2,1)]) \
                        or (K == 'k32' and n123 in [(0,1,1),(2,1,1)]) \
                        or (K in ['F2','b2','K'] and n123 == (0,0,0)) \
                        or (K == 'G2' and n123 == (0,0,2)):
                    split_factor = 0.5 if n123 in [(1,2,1),(2,1,1)] else 1.0
                    for j in range(2):
                        for n in range(2):
                            Kctr = 'k1sqk2sq{}'.format(K)
                            n123_jn = np.copy(n123)
                            n123_jn[0] += 2*(j+1)
                            n123_jn[1] += 2*(n+1)
                            add_product(DeltaB_K, tuple(n123_jn), Kctr,
                                        neff1, neff2, Kctr_k4_deriv_sum[2*j+n],
                                        coeff[i]*cnloB[j]*cnloB[n]*split_factor)

        return DeltaB_K

    def join_stoch_kernel_mu123_shell_average(self, n123_tuples, ell, neff,
                                              coeff, q_tr, q_lo, cnloB_stoch=0):
        def add_product(var, n123, IK, coeff, IK_deriv=None):
            n123p200 = tuple(np.array(n123)+np.array((2,0,0)))
            n123p020 = tuple(np.array(n123)+np.array((0,2,0)))
            n123p002 = tuple(np.array(n123)+np.array((0,0,2)))
            for i in range(self.nparams):
                t1 = self.stoch_kernels_shell_average[IK][n123][ell][...,i] \
                     * (1.0 + (q_tr[i]-q_lo[i])*sum(n123) + (1.0-q_tr[i]) \
                        * neff[self.tri_eff_to_id][...,i])
                t2 = self.stoch_kernels_shell_average[IK][n123p200][ell][...,i]\
                     * (q_tr[i] - q_lo[i]) \
                     * (neff[self.tri_eff_to_id][...,i] - n123[0])
                t3 = - self.stoch_kernels_shell_average[IK][n123p020] \
                       [ell][...,i] * (q_tr[i] - q_lo[i]) * n123[1]
                t4 = - self.stoch_kernels_shell_average[IK][n123p002] \
                       [ell][...,i] * (q_tr[i] - q_lo[i]) * n123[2]
                if IK_deriv is not None:
                    t1 += self.stoch_kernels_shell_average[IK_deriv][n123] \
                          [ell][...,i] * (1.0-q_tr[i])
                    t2 += self.stoch_kernels_shell_average[IK_deriv][n123p200] \
                          [ell][...,i] * (q_tr[i]-q_lo[i])
                var[...,i] += coeff[i] * (t1+t2+t3+t4)

        DeltaB_stoch = np.zeros_like(
            self.stoch_kernels_shell_average['id'][0,0,0][0])
        for i, n123 in enumerate(n123_tuples):
            add_product(DeltaB_stoch, n123, 'id', coeff[i])
            if 'VDG_infty_ctr' in self.model \
                    or ('EFT' in self.model and self.cnlo_type == 'EggLeeSco'):
                n123_ctr = np.copy(n123)
                n123_ctr[0] += 2
                add_product(DeltaB_stoch, tuple(n123_ctr), 'ksq',
                            coeff[i]*cnloB_stoch, 'dksq_dlnk')
            elif 'EFT' in self.model and self.cnlo_type == 'IvaPhiNis':
                if n123 == (0,0,0):
                    n123_ctr = np.copy(n123)
                    for n in range(2):
                        n123_ctr[0] += 2
                        add_product(DeltaB_stoch, tuple(n123_ctr), 'ksq',
                                    coeff[i]*cnloB_stoch[n], 'dksq_dlnk')

        return DeltaB_stoch

    def Bell(self, PL_dw, neff, params, ell=[0], W_damping=None):
        kernel = {}
        kernel_stoch = {}
        self.nparams = len(np.atleast_1d(params['wc']))
        if self.real_space:
            b1sq = params['b1']**2
            params_kernels = {}
            params_kernels['F2'] = 2*params['b1']*b1sq
            params_kernels['b2'] = params['b2']*b1sq
            params_kernels['K'] = 2*params['g2']*b1sq
            params_stoch = params['MB0']*b1sq/self.nbar
            if self.discrete_average:
                kernel[0] = np.zeros_like(
                    self.kernels_shell_average['F2'][0,0,0][0])
                for KK in self.kernel_names:
                    kernel[0] += np.einsum(
                        "abc,c->abc",
                        self.kernels_shell_average[KK][0,0,0][0],
                        params_kernels[KK]
                    )
                kernel_stoch[0] = np.einsum(
                    "abc,c->abc",
                    self.stoch_kernels_shell_average['id'][0,0,0][0],
                    params_stoch
                )
            else:
                kernel[0] = np.zeros(self.kernels['F2'].shape + (self.nparams,))
                for KK in self.kernel_names:
                    kernel[0] += np.einsum("ab,c->abc", self.kernels[KK],
                                           params_kernels[KK])
                kernel_stoch[0] = np.full(
                    self.kernels['F2'].shape + (self.nparams,),
                    params_stoch)
        else:
            b1sq = params['b1']**2
            b1f = params['b1']*params['f']
            f2 = params['f']**2
            f4 = params['f']**4

            params_kernels = {}
            params_kernels['F2'] = 2*params['b1'] \
                                   * np.array([b1sq, b1f, b1f, f2])
            params_kernels['b2'] = params['b2'] * np.array([b1sq, b1f, b1f, f2])
            params_kernels['K'] = 2*params['g2'] \
                                  * np.array([b1sq, b1f, b1f, f2])
            params_kernels['G2'] = 2*params['f'] \
                                   * np.array([b1sq, b1f, b1f, f2])
            params_kernels['k31'] = -b1f * np.array([b1sq, b1f, 2*b1f, f2,
                                                     2*f2, f4/b1f])
            params_kernels['k32'] = params_kernels['k31']
            params_stoch = np.array([
                b1sq * params['MB0'],
                b1f * (params['MB0'] + params['NP0']),
                f2 * (params['NP0'])
            ]) / self.nbar
            if ('EFT' in self.model or 'VDG_infty_ctr' in self.model) \
                    and self.cnlo_type == 'EggLeeSco':
                cnloB = params['cnloB']*f2
            elif ('EFT' in self.model or 'VDG_infty_ctr' in self.model)\
                    and self.cnlo_type == 'IvaPhiNis':
                cnloB = -np.array([params['cB1'], params['cB2']])/params['b1']
            else:
                cnloB = None
            if 'EFT' in self.model or 'VDG_infty_ctr' in self.model:
                cnloB_stoch = cnloB
            else:
                cnloB_stoch = 0.0

            if 'VDG_infty' in self.model and not self.discrete_average:
                self.compute_damped_mu123_integrals(self.tri, W_damping,
                                                    max(ell))

            for l in np.arange(0, max(ell)+1, 2):
                if self.discrete_average:
                    kernel[l] = np.zeros(
                        self.kernels_shell_average['F2'][0,0,0][l].shape)
                    for KK in ['F2','b2','K','G2','k31','k32']:
                        kernel[l] += self.join_kernel_mu123_shell_average(
                            KK, self.kernel_mu_tuples[KK], l, neff,
                            params_kernels[KK], params['q_tr'], params['q_lo'],
                            cnloB
                        )
                    kernel_stoch[l] = \
                        self.join_stoch_kernel_mu123_shell_average(
                            [(0,0,0),(2,0,0),(4,0,0)], l, neff, params_stoch,
                            params['q_tr'], params['q_lo'], cnloB_stoch
                        )
                else:
                    kernel[l] = np.zeros(
                        self.kernels['F2'].shape + (self.nparams,))
                    for KK in ['F2','b2','K','G2','k31','k32']:
                        kernel[l] += self.join_kernel_mu123_integral(
                            KK, self.kernel_mu_tuples[KK], l, neff,
                            params_kernels[KK], params['q_tr'], params['q_lo'],
                            cnloB
                        )
                    kernel_stoch[l] = self.join_stoch_kernel_mu123_integral(
                        [(0,0,0),(2,0,0),(4,0,0)], l, neff, params_stoch,
                        params['q_tr'], params['q_lo'], cnloB_stoch
                    )

            if 4 in ell:
                kernel[4] = 1.125 * (35*kernel[4] - 30*kernel[2] + 3*kernel[0])
                kernel_stoch[4] = 1.125 * (35*kernel_stoch[4] \
                                  - 30*kernel_stoch[2] + 3*kernel_stoch[0])
            if 2 in ell:
                kernel[2] = 2.5 * (3*kernel[2] - kernel[0])
                kernel_stoch[2] = 2.5 * (3*kernel_stoch[2] - kernel_stoch[0])

        if self.discrete_average:
            P = PL_dw/self.fiducial_Pdw_eff
            P2 = P[self.ki_eff] * P[self.kj_eff]
            tri_to_id = self.tri_eff_to_id
            tri_to_id_sq = self.tri_eff_to_id_sq
        elif self.use_effective_triangles:
            P2 = PL_dw[self.ki_eff]*PL_dw[self.kj_eff]
            P = PL_dw
            tri_to_id = self.tri_eff_to_id
            tri_to_id_sq = self.tri_eff_to_id_sq
        else:
            P2 = PL_dw[self.ki]*PL_dw[self.kj]
            P = PL_dw
            tri_to_id = self.tri_to_id
            tri_to_id_sq = self.tri_to_id_sq
        q6 = params['q_tr']**4 * params['q_lo']**2

        Bell_dict = {}
        for l in ell:
            ids = self.tri_id_ell[l]
            B_SPT = np.einsum("ij...,ij...->i...", kernel[l][ids],
                              P2[tri_to_id_sq][ids])
            B_stoch = np.einsum("ij...,ij...->i...", kernel_stoch[l][ids],
                                P[tri_to_id][ids])
            if l == 0:
                B_stoch += params['NB0']/self.nbar**2

            Bell_dict['ell{}'.format(l)] = np.squeeze((B_SPT + B_stoch) / q6)

        return Bell_dict

    def BX_ell(self, PL_dw, neff, params, ell=[0], W_damping=None):
        kernel = {}
        kernel_stoch = {}
        if self.real_space:
            kernel[0] = {}
            kernel_stoch[0] = {}

            for kk in ['F2','b2','K','G2','k31','k32']:
                for diagram in self.kernel_diagrams[kk]:
                    if diagram == 'B0L_b1b1b1':
                        kernel[0][diagram] = 2*self.kernels['F2']
                    elif diagram == 'B0L_b1b1b2':
                        kernel[0][diagram] = np.ones(self.kernels['F2'].shape)
                    elif diagram == 'B0L_b1b1g2':
                        kernel[0][diagram] = 2*self.kernels['K']
                    else:
                        kernel[0][diagram] = np.zeros(self.kernels['F2'].shape)

            kernel_stoch[0]['Bnoise_MB0b1b1'] = 1.0
            kernel_stoch[0]['Bnoise_MB0b1'] = 0.0
            kernel_stoch[0]['Bnoise_NP0'] = 0.0
        else:
            f2 = params['f']**2
            f3 = params['f']**3
            f4 = f2**2
            params_kernels = {}
            params_kernels['F2'] = 2 * np.array([1.0, params['f'], params['f'],
                                                 f2])
            params_kernels['b2'] = np.array([1.0, params['f'], params['f'], f2])
            params_kernels['K'] = 2 * np.array([1.0, params['f'], params['f'],
                                                f2])
            params_kernels['G2'] = 2 * np.array([params['f'], f2, f2, f3])
            params_kernels['k31'] = -np.array([params['f'], f2, 2*f2, f3,
                                                     2*f3, f4])
            params_kernels['k32'] = params_kernels['k31']
            params_stoch = np.array([1.0, params['f'], f2])

            if 'VDG_infty' in self.model and not self.discrete_average:
                    self.compute_damped_mu123_integrals(self.tri, W_damping,
                                                        max(ell))

            for l in np.arange(0, max(ell)+1, 2):
                kernel[l] = {}
                kernel_stoch[l] = {}
                if not self.discrete_average:
                    for kk in ['F2','b2','K','G2','k31','k32']:
                        kernel_temp = self.join_kernel_mu123_integral_indv(
                            kk, self.kernel_mu_tuples[kk], l, neff,
                            params_kernels[kk], params['q_tr'], params['q_lo']
                        )
                        for i,diagram in enumerate(self.kernel_diagrams[kk]):
                            if diagram not in kernel[l]:
                                kernel[l][diagram] = kernel_temp[i]
                            else:
                                kernel[l][diagram] += kernel_temp[i]
                    kernel_stoch_temp = \
                        self.join_stoch_kernel_mu123_integral_indv(
                            [(0,0,0),(2,0,0),(4,0,0)], l, neff, params_stoch,
                            params['q_tr'], params['q_lo']
                        )
                    kernel_stoch[l]['Bnoise_MB0b1b1'] = kernel_stoch_temp[0]
                    kernel_stoch[l]['Bnoise_MB0b1'] = kernel_stoch_temp[1]
                    kernel_stoch[l]['Bnoise_NP0'] = kernel_stoch_temp[2]

            if 4 in ell:
                for diagram in kernel[4]:
                    kernel[4][diagram] = 1.125 * (35*kernel[4][diagram] \
                                                  - 30*kernel[2][diagram] \
                                                  + 3*kernel[0][diagram])
                for diagram in kernel_stoch[4]:
                    kernel_stoch[4][diagram] = \
                        1.125 * (35*kernel_stoch[4][diagram] \
                                 - 30*kernel_stoch[2][diagram] \
                                 + 3*kernel_stoch[0][diagram])
            if 2 in ell:
                for diagram in kernel[2]:
                    kernel[2][diagram] = 2.5 * (3*kernel[2][diagram] \
                                                - kernel[0][diagram])
                for diagram in kernel_stoch[2]:
                    kernel_stoch[2][diagram] = \
                        2.5 * (3*kernel_stoch[2][diagram] \
                               - kernel_stoch[0][diagram])

        if not self.discrete_average:
            P2 = PL_dw[self.ki]*PL_dw[self.kj]
            tri_to_id = self.tri_to_id
            tri_to_id_sq = self.tri_to_id_sq
        q6 = params['q_tr']**4 * params['q_lo']**2

        BX_ell_dict = {}
        for l in ell:
            ids = self.tri_id_ell[l]
            BX_ell_dict['ell{}'.format(l)] = {}
            for diagram in kernel[l]:
                BX_ell_dict['ell{}'.format(l)][diagram] = np.einsum(
                    "ij,ij->i", kernel[l][diagram][ids], P2[tri_to_id_sq][ids]
                ) / q6
            if self.real_space:
                for diagram in kernel_stoch[l]:
                    BX_ell_dict['ell{}'.format(l)][diagram] = np.sum(
                        PL_dw[self.tri_to_id][ids], axis=1
                    ) * kernel_stoch[l][diagram] / q6
            else:
                for diagram in kernel_stoch[l]:
                    BX_ell_dict['ell{}'.format(l)][diagram] = np.einsum(
                        "ij,ij->i", kernel_stoch[l][diagram][ids],
                        PL_dw[tri_to_id][ids]
                    ) / q6
            if l == 0:
                BX_ell_dict['ell{}'.format(l)]['Bnoise_NB0'] = \
                    np.ones(len(self.tri_id_ell[0])) / q6
            else:
                BX_ell_dict['ell{}'.format(l)]['Bnoise_NB0'] = \
                    np.zeros(len(self.tri_id_ell[l]))

        return BX_ell_dict

    def Gaussian_covariance(self, l1, l2, dk, Pell, volume, Ntri=None):
        if Ntri is None:
            Ntri = volume**2 * 8*np.pi**2*np.prod(self.tri, axis=1) \
                   * dk**3/(2*np.pi)**6

        ell_for_cov = [0,2,4] if not self.real_space else [0]

        for l3 in ell_for_cov:
            for l4 in ell_for_cov:
                for l5 in ell_for_cov:
                    try:
                        self.cov_mixing_kernel[(l1,l2,l3,l4,l5)]
                    except KeyError:
                        self.compute_covariance_mixing_kernel(l1,l2,l3,l4,l5)

        Pell_array = np.zeros((self.tri_unique.shape[0],len(Pell.keys())))
        for i,ell in enumerate(Pell.keys()):
            Pell_array[:,i] = Pell[ell]

        cov = np.zeros(self.tri.shape[0])
        for i3,l3 in enumerate(ell_for_cov):
            for i4,l4 in enumerate(ell_for_cov):
                for i5,l5 in enumerate(ell_for_cov):
                    mask = np.array([[False]*len(ell_for_cov)]*3)
                    mask[0,i3] = True
                    mask[1,i4] = True
                    mask[2,i5] = True
                    cov += self.cov_mixing_kernel[(l1,l2,l3,l4,l5)] * \
                        np.prod(Pell_array[self.tri_to_id], axis=(1,2),
                                where=mask)

        cov *= (2*l1+1) * (2*l2+1) * volume / Ntri
        return cov
