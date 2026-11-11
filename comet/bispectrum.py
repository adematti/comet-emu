"""Bispectrum module."""

import math
import numpy as np
import numba as nb
import pickle
from scipy.special import factorial, lpmv
from comet.grid import Grid, CtypedGrid
from comet.splines import _linear_bucket_table

nb.config.THREADING_LAYER = 'workqueue'

try:
    import jax
    import jax.numpy as jnp
    from functools import partial as _partial
    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

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



@nb.njit(cache=True, fastmath=True)
def _bisp_node_total(k1, k2, k3, mu1, mu2, mu3, pd1, pd2, pd3,
                     bb1, bb2, gg2, ff, av, sv_p, mb0, np0, cnl, ccB1,
                     ccB2, inb, iqs):
    """Bias-weighted VDG bispectrum at one quadrature node (AP-corrected
    wavemodes and cosines, ``pd*`` the Pdw of each leg), used by
    `BispectrumNum._fused_moments_njit`. Its JAX counterpart is the node
    computation of `BispectrumNum._fused_moments_jax`."""
    mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu3_sq = mu3*mu3
    # Z1 includes the IvaPhiNis counterterms. Z2 is written out
    # explicitly so its velocity term uses the bare Z1 = b1 + f mu^2.
    Z1_1 = bb1 + ff*mu1_sq - ccB1*(k1*k1)*mu1_sq - ccB2*(k1*k1)*mu1_sq*mu1_sq
    Z1_2 = bb1 + ff*mu2_sq - ccB1*(k2*k2)*mu2_sq - ccB2*(k2*k2)*mu2_sq*mu2_sq
    Z1_3 = bb1 + ff*mu3_sq - ccB1*(k3*k3)*mu3_sq - ccB2*(k3*k3)*mu3_sq*mu3_sq

    # the divisions by k_i are done with three reciprocals
    i1 = 1.0/k1; i2 = 1.0/k2; i3 = 1.0/k3
    k1_sq = k1*k1; k2_sq = k2*k2; k3_sq = k3*k3

    # tree leg 12 (i=1, j=2, k=3, muk = -mu3)
    muij = 0.5*(k3_sq - k1_sq - k2_sq)*i1*i2
    muij2 = muij*muij
    ratio = 0.5*(k1*i2 + k2*i1)*muij
    F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
    G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
    K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
    muk = -mu3
    Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k3*muk*(
        bb1*(mu1*i1 + mu2*i2) + ff*mu1*mu2*(mu2*i1 + mu1*i2))
    tree = 2.0*Z1_1*Z1_2*Z2 * pd1*pd2

    # tree leg 23
    muij = 0.5*(k1_sq - k2_sq - k3_sq)*i2*i3
    muij2 = muij*muij
    ratio = 0.5*(k2*i3 + k3*i2)*muij
    F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
    G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
    K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
    muk = -mu1
    Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k1*muk*(
        bb1*(mu2*i2 + mu3*i3) + ff*mu2*mu3*(mu3*i2 + mu2*i3))
    tree += 2.0*Z1_2*Z1_3*Z2 * pd2*pd3

    # tree leg 31
    muij = 0.5*(k2_sq - k3_sq - k1_sq)*i3*i1
    muij2 = muij*muij
    ratio = 0.5*(k3*i1 + k1*i3)*muij
    F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
    G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
    K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
    muk = -mu2
    Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k2*muk*(
        bb1*(mu3*i3 + mu1*i1) + ff*mu3*mu1*(mu1*i3 + mu3*i1))
    tree += 2.0*Z1_3*Z1_1*Z2 * pd3*pd1

    kmu1_sq = k1*k1*mu1_sq
    kmu2_sq = k2*k2*mu2_sq
    kmu3_sq = k3*k3*mu3_sq
    eft = 1.0 + cnl*ff*ff*(kmu1_sq + kmu2_sq + kmu3_sq)
    tree = tree * eft
    st1 = (bb1*mb0 + ff*np0*mu1_sq) * Z1_1 * (1.0 + cnl*ff*ff*kmu1_sq)
    st2 = (bb1*mb0 + ff*np0*mu2_sq) * Z1_2 * (1.0 + cnl*ff*ff*kmu2_sq)
    st3 = (bb1*mb0 + ff*np0*mu3_sq) * Z1_3 * (1.0 + cnl*ff*ff*kmu3_sq)

    # W_infty damping (identically 1 when sv = 0, e.g. for EFT): the powers
    # are written with sqrt, much cheaper than a generic pow
    winfty = 1.0
    if sv_p != 0.0:
        sv2 = sv_p*sv_p
        av2 = av*av
        lamb2 = -0.5*ff*ff*(kmu1_sq + kmu2_sq + kmu3_sq)
        denom = 1.0 - lamb2*av2
        winfty = np.exp(lamb2*sv2/denom) / (denom*np.sqrt(denom))
        lam1 = -ff*ff*kmu1_sq
        den1 = 1.0 - lam1*av2
        st1 *= np.exp(lam1*sv2/den1) / np.sqrt(den1)
        lam2 = -ff*ff*kmu2_sq
        den2 = 1.0 - lam2*av2
        st2 *= np.exp(lam2*sv2/den2) / np.sqrt(den2)
        lam3 = -ff*ff*kmu3_sq
        den3 = 1.0 - lam3*av2
        st3 *= np.exp(lam3*sv2/den3) / np.sqrt(den3)
    stoch = (st1*pd1 + st2*pd2 + st3*pd3) * inb

    return (tree*winfty + stoch) * iqs


@nb.njit(cache=True, fastmath=True)
def _bisp_node_diagrams(k1, k2, k3, mu1, mu2, mu3, pd1, pd2, pd3, ff, av,
                        sv_p, iqs, tree_col, tree_a, tree_b, tree_ctr,
                        stoch_part, stoch_ctr, need_ctr, A, Z, S, spt,
                        stoch):
    """Per-diagram, bias-stripped bispectrum at one quadrature node,
    written into ``spt`` (tree diagrams) and ``stoch`` (stochastic
    ones); ``A``, ``Z`` and ``S`` are scratch buffers. Tree diagram ``c``
    is ``sum_legs sum_t 2 A[a_t] Z[b_t] P_i P_j`` over the terms ``t`` with
    ``tree_col[t] == c``, times ``W_infty / qiso6`` and, if ``tree_ctr[c]``,
    ``f^2 sum_j k_j^2 mu_j^2``. Stochastic diagram ``c`` is
    ``sum_i S[stoch_part[c]] W_i P_i / qiso6`` (times ``f^2 k_i^2 mu_i^2``
    if ``stoch_ctr[c]``). See `BispectrumNum._build_diagram_tables` for the
    A, Z and S pieces. Used by `BispectrumNum._fused_moments_diagrams_njit`;
    its JAX counterpart is `BispectrumNum._fused_moments_diagrams_jax`."""
    n_terms = tree_col.size
    n_tree = tree_ctr.size
    n_stoch = stoch_part.size
    f2 = ff*ff
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu3_sq = mu3*mu3
    kmu1_sq = k1*k1*mu1_sq
    kmu2_sq = k2*k2*mu2_sq
    kmu3_sq = k3*k3*mu3_sq
    kmu_sum = kmu1_sq + kmu2_sq + kmu3_sq

    if n_tree > 0:
        for c in range(n_tree):
            spt[c] = 0.0
        for leg in range(3):
            if leg == 0:
                ki = k1; kj = k2; kk = k3
                mui = mu1; muj = mu2; muk = mu3
                mui_sq = mu1_sq; muj_sq = mu2_sq; muk_sq = mu3_sq
                ui = kmu1_sq; uj = kmu2_sq
                pp = pd1*pd2
            elif leg == 1:
                ki = k2; kj = k3; kk = k1
                mui = mu2; muj = mu3; muk = mu1
                mui_sq = mu2_sq; muj_sq = mu3_sq; muk_sq = mu1_sq
                ui = kmu2_sq; uj = kmu3_sq
                pp = pd2*pd3
            else:
                ki = k3; kj = k1; kk = k2
                mui = mu3; muj = mu1; muk = mu2
                mui_sq = mu3_sq; muj_sq = mu1_sq; muk_sq = mu2_sq
                ui = kmu3_sq; uj = kmu1_sq
                pp = pd3*pd1

            inv_i = 1.0/ki
            inv_j = 1.0/kj
            muij = 0.5*(kk*kk - ki*ki - kj*kj)*inv_i*inv_j
            muij2 = muij*muij
            ratio = 0.5*(ki*inv_j + kj*inv_i)*muij
            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
            fkm = ff*kk*muk
            a_i = fkm * mui*inv_i
            a_j = fkm * muj*inv_j

            Z[0] = F2 - 0.5*(a_i + a_j)
            Z[1] = 0.5
            Z[2] = muij2 - 1.0
            Z[3] = ff*muk_sq*G2 - 0.5*ff*(a_i*muj_sq + a_j*mui_sq)

            A[0] = 1.0
            A[1] = ff*(mui_sq + muj_sq)
            A[2] = f2*mui_sq*muj_sq
            if need_ctr:
                vi = ui*mui_sq
                vj = uj*muj_sq
                A[3] = -(ui + uj)
                A[4] = -ff*(mui_sq*uj + muj_sq*ui)
                A[5] = -(vi + vj)
                A[6] = -ff*(mui_sq*vj + muj_sq*vi)
                A[7] = ui*uj
                A[8] = ui*vj + vi*uj
                A[9] = vi*vj

            for t in range(n_terms):
                spt[tree_col[t]] += \
                    2.0*A[tree_a[t]]*Z[tree_b[t]]*pp

        winfac = iqs
        if sv_p != 0.0:
            lamb2 = -0.5*f2*kmu_sum
            denom = 1.0 - lamb2*av*av
            winfac *= np.exp(lamb2*sv_p*sv_p/denom) / (denom*np.sqrt(denom))
        for c in range(n_tree):
            if tree_ctr[c]:
                spt[c] *= winfac*f2*kmu_sum
            else:
                spt[c] *= winfac

    if n_stoch > 0:
        for c in range(n_stoch):
            stoch[c] = 0.0
        for leg in range(3):
            if leg == 0:
                mui_sq = mu1_sq; ui = kmu1_sq; pdi = pd1
            elif leg == 1:
                mui_sq = mu2_sq; ui = kmu2_sq; pdi = pd2
            else:
                mui_sq = mu3_sq; ui = kmu3_sq; pdi = pd3
            wp = pdi * iqs
            if sv_p != 0.0:
                lam = -f2*ui
                den = 1.0 - lam*av*av
                wp *= np.exp(lam*sv_p*sv_p/den) / np.sqrt(den)
            vi = ui*mui_sq
            S[0] = 1.0
            S[1] = ff*mui_sq
            S[2] = f2*mui_sq*mui_sq
            S[3] = -ui
            S[4] = -ff*mui_sq*ui
            S[5] = -vi
            S[6] = -ff*mui_sq*vi
            for c in range(n_stoch):
                if stoch_ctr[c]:
                    stoch[c] += \
                        S[stoch_part[c]]*wp*f2*ui
                else:
                    stoch[c] += S[stoch_part[c]]*wp


@nb.njit(cache=True, fastmath=True, inline="always")
def _ppoly_segment(q, j, V, seg, log_lo, inv_dlog):
    """Index ``j`` of the largest distinct knot ``V[j] <= q``, for
    ``V[0] <= q <= V[-1]``. With ``j < 0`` it is found with the O(1) bucket
    table of `Splines.ppoly_table`; otherwise by stepping from the guess
    ``j``, which is cheaper (no log) when ``q`` is in the guessed segment or
    a neighbouring one, as between consecutive quadrature nodes. Both give
    the same index. The JAX path (`BispectrumNum._ppoly_eval_jax`) uses
    linear buckets instead (`comet.splines._linear_bucket_table`)."""
    nv = V.size
    if j < 0:
        b = min(max(int((np.log(q) - log_lo)*inv_dlog), 0), seg.size - 1)
        j = seg[b]
        if j + 1 < nv and q >= V[j + 1]:
            j += 1
        return j
    while j + 1 < nv and q >= V[j + 1]:
        j += 1
    while j > 0 and q < V[j]:
        j -= 1
    return j


@nb.njit(cache=True, fastmath=True, inline="always")
def _ppoly_node(k, mu2, m, j, G, scale, seg, last_index, V, log_lo, inv_dlog,
                coeffs, tails):
    """Value at ``(k, mu^2)`` of the spline exported by
    `_PdwEvaluator.ppoly_table`, for the parameter set ``m``, including its
    power-law tails (exactly as `Splines.eval`). ``coeffs`` (4, nseg, nmu2,
    nparams) holds the coefficients of the polynomial in ``mu^2``, which is
    of degree 0 for a fixed-mu Pdw. The segment is found once (see
    `_ppoly_segment`, ``j`` being the guess, or -1 for none) and returned
    with the value, as the guess for the next call. Its JAX counterpart is
    `BispectrumNum._ppoly_eval_jax`."""
    sc = scale[m]
    lo = G[0]*sc
    hi = G[G.size - 1]*sc
    if k < lo:
        return tails[0, m]*(k/lo)**tails[1, m], j
    if k > hi:
        return tails[2, m]*(k/hi)**tails[3, m], j
    q = min(max(k/sc, V[0]), V[V.size - 1])
    j = _ppoly_segment(q, j, V, seg, log_lo, inv_dlog)
    i = min(max(last_index[j], 0), G.size - 2)
    dx = (q - G[i])*sc
    n = coeffs.shape[2] - 1
    val = ((coeffs[0, i, n, m]*dx + coeffs[1, i, n, m])*dx
           + coeffs[2, i, n, m])*dx + coeffs[3, i, n, m]
    for n in range(coeffs.shape[2] - 2, -1, -1):
        val = val*mu2 + ((coeffs[0, i, n, m]*dx + coeffs[1, i, n, m])*dx
                         + coeffs[2, i, n, m])*dx + coeffs[3, i, n, m]
    return val, j


class _PdwEvaluator:
    r"""Callable returning the IR-resummed :math:`P_\mathrm{dw}(k)`, passed to
    the numerical-bispectrum routines.

    Besides ``__call__`` (which delegates to :meth:`PTEmu.Pdw`), it exposes
    :meth:`ppoly_table`, handing out the underlying ``Pdw_spline`` in
    power-basis form so the fused numba/jax kernels can evaluate it inline
    without resampling and re-fitting a second spline. Both share the single
    emulator evaluation that :meth:`PTEmu.Pdw` performs (cached via
    ``dw_spline_up_to_date``).
    """

    def __init__(self, emu, params, de_model, ell_for_recon):
        self._emu = emu
        self._params = params
        self._de_model = de_model
        self._ell_for_recon = ell_for_recon

    def __call__(self, k, mu=None):
        mu = self._emu.Pdw_mu if mu is None else mu
        if mu == 'leg':
            raise ValueError("Pdw_mu is 'leg': pass an explicit mu.")
        return self._emu.Pdw(k, self._params, de_model=self._de_model,
                             mu=mu, ell_for_recon=self._ell_for_recon)

    def ppoly_table(self):
        """Power-basis export of ``Pdw_spline`` with its segment-lookup table
        (see :meth:`comet.splines.Splines.ppoly_table`), with ``coeffs`` of
        shape (4, nseg, nmu2, nparams) holding a polynomial in ``mu^2`` (see
        `_ppoly_node`). For a fixed ``PTEmu.Pdw_mu``, ``nmu2 = 1``; for
        ``'leg'`` it is the table of :meth:`PTEmu._Pdw_mu_ppoly_table`."""
        if self._emu.Pdw_mu == 'leg':
            return self._emu._Pdw_mu_ppoly_table(self._ell_for_recon)
        # Ensure the fixed-mu IR-resummed spline is built (cheap single-point
        # trigger; the emulator eval + spline build are cached inside Pdw)
        self(self._emu.k_table[:1])
        t = dict(self._emu.Pdw_spline.ppoly_table())
        t['coeffs'] = t['coeffs'][:, :, None, :]
        return t


class BispectrumNum:
    """Numerical-integration bispectrum multipoles.

    Supports two projection bases, both evaluated by the same fused kernels
    (see `_fused_moments`), which project the bispectrum node by node
    without building the full quadrature grid:

    - **Sugiyama**: Projects onto ``(l1, l2, L)`` multipoles via a
      ``(mu1, mu12, phi)`` quadrature per ``(k1, k2)`` pair.
    - **Scoccimarro**: Projects onto ``(l, m)`` multipoles via a
      ``(mu1, phi)`` quadrature per ``(k1, k2, k3)`` triangle.

    Supports real-space and redshift-space (VDG_infty / EFT / VDG_infty_ctr).
    EFT counterterms are enabled by including ``'EFT'`` or
    ``'VDG_infty_ctr'`` in ``model``; `change_cnlo_type` selects the flavour:

    - ``'EggLeeSco'`` (default): the tree integrand is multiplied by
      ``1 + cnloB f^2 sum_j(k_j^2 mu_j^2)`` and each stochastic leg by
      ``1 + cnloB f^2 k_i^2 mu_i^2``.
    - ``'IvaPhiNis'``: ``Z1(k_i) -> Z1(k_i) - cB1 k_i^2 mu_i^2
      - cB2 k_i^2 mu_i^4``.

    The diagram-resolved outputs (`BX_ell_Sugi`, `BX_ell_Scocc`) add the
    corresponding counterterm diagrams; see `_build_diagram_tables`.

    Two inner-loop backends are available:

    - ``backend='numba'`` (default): fused njit kernels.
    - ``backend='jax'``: JAX-traced kernels (requires `jax`).

    """

    def __init__(self, real_space, model, use_Mpc, backend='numba',
                 reparametrisation=None):
        self.real_space = real_space
        self.model = model
        self.use_Mpc = use_Mpc
        self.nbar = 1.0
        self.cnlo_type = 'EggLeeSco'
        self.change_reparametrisation(reparametrisation)
        if backend not in ('numba', 'jax'):
            raise ValueError(
                f"backend must be 'numba' or 'jax'; got {backend!r}.")
        if backend == 'jax' and not HAS_JAX:
            raise ImportError(
                "backend='jax' requested but JAX is not installed.")
        self.backend = backend
        self._sugi_quad_cache = {}
        self._sugi_weights_cache = {}
        self._scocc_quad_cache = {}
        self._scocc_proj_cache = {}

        self._build_diagram_tables()

    # Configuration

    def define_nbar(self, nbar):
        self.nbar = np.copy(nbar)

    def define_units(self, use_Mpc):
        self.use_Mpc = use_Mpc

    def change_reparametrisation(self, reparametrisation):
        """Set the nuisance-parameter reparametrisation (None, 'TCM' or
        'TCMlinear'), as in `PTEmu`. It only affects
        `_get_bias_coeff_for_AM`: all other methods take physical
        parameters."""
        if reparametrisation not in (None, 'TCM', 'TCMlinear'):
            raise ValueError(
                "Unknown reparametrisation {!r}; use None, 'TCM' or "
                "'TCMlinear'.".format(reparametrisation))
        self.reparametrisation = reparametrisation

    def change_RSD_model(self, model):
        """Switch the RSD model and rebuild the diagram tables. Mirrors
        `Bispectrum.change_RSD_model`."""
        self.model = model
        self._build_diagram_tables()

    def change_cnlo_type(self, type):
        """Switch the EFT counterterm flavour. Mirrors `Bispectrum.change_cnlo_type`."""
        if type in ('EggLeeSco', 'IvaPhiNis'):
            self.cnlo_type = type
            self._build_diagram_tables()
        else:
            raise ValueError(
                f"Unknown cnlo_type {type!r}; use 'EggLeeSco' or 'IvaPhiNis'.")

    def clear_caches(self):
        """Clear all internal caches."""
        self._sugi_quad_cache.clear()
        self._sugi_weights_cache.clear()
        self._scocc_quad_cache.clear()
        self._scocc_proj_cache.clear()

    def _has_ctr(self):
        return 'EFT' in self.model or 'VDG_infty_ctr' in self.model

    def _check_model(self, fn):
        if 'VDG_infty' not in self.model and 'EFT' not in self.model:
            raise NotImplementedError(
                "{} is currently implemented only for VDG_infty and EFT "
                "models; got model={!r}.".format(fn, self.model))

    # Diagrams and their coefficients

    # Diagram decomposition used by the diagram kernels. Per leg (i, j; k),
    # the tree integrand is 2 Z1_i Z1_j Z2 P_i P_j, with
    #   Z1_i = b1 + f mu_i^2 - cB1 u_i - cB2 v_i,
    #   u_i = k_i^2 mu_i^2,  v_i = k_i^2 mu_i^4.
    # Z1_i Z1_j is split into the pieces A[a] and Z2 into the pieces Z[b];
    # each tree diagram is a sum of 2 A[a] Z[b] products.
    #   a: 0 -> 1 [b1^2]     1 -> f(mu_i^2+mu_j^2) [b1]   2 -> f^2 mu_i^2 mu_j^2
    #      3 -> -(u_i+u_j) [b1 cB1]        4 -> -f(mu_i^2 u_j + mu_j^2 u_i) [cB1]
    #      5 -> -(v_i+v_j) [b1 cB2]        6 -> -f(mu_i^2 v_j + mu_j^2 v_i) [cB2]
    #      7 -> u_i u_j [cB1^2]   8 -> u_i v_j + v_i u_j [cB1 cB2]
    #      9 -> v_i v_j [cB2^2]
    #   b: 0 -> F2 + velocity [b1]   1 -> 1/2 [b2]   2 -> mu_ij^2 - 1 [g2]
    #      3 -> f mu_k^2 G2 + f velocity [1]
    # The stochastic integrand per leg is W_i P_i (b1 MB0 + f NP0 mu_i^2) Z1_i,
    # split into the pieces ``s``:
    #   s: 0 -> 1   1 -> f mu^2   2 -> f^2 mu^4   3 -> -u   4 -> -f mu^2 u
    #      5 -> -v   6 -> -f mu^2 v
    # EggLeeSco counterterm diagrams (``ctr=1``) carry an extra f^2 sum_j u_j
    # (tree) or f^2 u_i (stoch); their coefficient is ``cnloB`` times that
    # of the parent diagram.
    # Each entry: (name, kernel terms or stoch part, coefficient factors).
    # The coefficient is the product of the factors, see `_COEFF_FACTORS`.
    _BASE_TREE_TERMS = (
        ('B0L_b1b1b1', ((0, 0),), ('b1', 'b1', 'b1')),
        ('B0L_b1b1', ((1, 0), (0, 3)), ('b1', 'b1')),
        ('B0L_b1', ((2, 0), (1, 3)), ('b1',)),
        ('B0L_b1b1b2', ((0, 1),), ('b1', 'b1', 'b2')),
        ('B0L_b1b2', ((1, 1),), ('b1', 'b2')),
        ('B0L_b2', ((2, 1),), ('b2',)),
        ('B0L_b1b1g2', ((0, 2),), ('b1', 'b1', 'g2')),
        ('B0L_b1g2', ((1, 2),), ('b1', 'g2')),
        ('B0L_g2', ((2, 2),), ('g2',)),
        ('B0L_id', ((2, 3),), ()),
    )

    _BASE_STOCH_PARTS = (
        ('Bnoise_MB0b1b1', 0, ('MB0', 'b1', 'b1', '1/nbar')),
        ('Bnoise_MB0b1', 1, ('MB0', 'b1', '1/nbar')),
        ('Bnoise_NP0b1', 1, ('NP0', 'b1', '1/nbar')),
        ('Bnoise_NP0', 2, ('NP0', '1/nbar')),
    )

    _NB0_FACTORS = ('NB0', '1/nbar', '1/nbar')

    # Analytically marginalisable parameters and the power of
    # Aap = 1/(q_tr^2 q_lo) relating them to their TCM counterparts,
    # theta_TCM = Aap^n theta (see `PTEmu._rescale_params`).
    _TCM_AAP_POWER = {'MB0': 1, 'NP0': 1, 'NB0': 2}

    _COEFF_FACTORS = {
        'b1': lambda p, inb: p['b1'],
        'b2': lambda p, inb: p['b2'],
        'g2': lambda p, inb: p['g2'],
        'cnloB': lambda p, inb: p['cnloB'],
        'cB1': lambda p, inb: p['cB1'],
        'cB2': lambda p, inb: p['cB2'],
        'MB0': lambda p, inb: p['MB0'],
        'NP0': lambda p, inb: p['NP0'],
        'NB0': lambda p, inb: p['NB0'],
        '1/nbar': lambda p, inb: inb,
    }

    def _build_diagram_tables(self):
        """Set the diagram names, kernel term tables and coefficient factors
        for the current ``model`` and ``cnlo_type``.

        Each diagram is bias-stripped: the bispectrum is the sum of the
        diagrams weighted by their coefficient (see `diagram_coeffs`). Base
        diagrams use the bias monomial in their name (``B0L_b1b1b1`` -> b1^3,
        ``B0L_id`` -> 1); stochastic diagrams also carry the ``1/nbar``
        factors (``Bnoise_MB0b1b1`` -> MB0 b1^2/nbar, ``Bnoise_MB0b1`` ->
        MB0 b1/nbar, ``Bnoise_NP0b1`` -> NP0 b1/nbar, ``Bnoise_NP0`` ->
        NP0/nbar, ``Bnoise_NB0`` -> NB0/nbar^2). ``Bnoise_MB0b1`` and
        ``Bnoise_NP0b1`` share the same kernel.

        - EggLeeSco: each base diagram ``X`` gets a counterterm partner
          (``B0L_b1b1b1cnloB``, ..., ``B0L_cnloB``, ``Bnoise_MB0b1b1cnloB``,
          ...) with coefficient ``cnloB`` times that of ``X``.
        - IvaPhiNis: tree diagrams linear (``B0L_b1b1cB1``, ``B0L_b2cB2``,
          ...) and quadratic (``B0L_b1cB1cB1``, ``B0L_cB1cB2``, ...) in the
          counterterms, named by their monomial, and stochastic diagrams
          ``Bnoise_MB0b1cB1`` -> MB0 b1 cB1/nbar,
          ``Bnoise_NP0cB1`` -> NP0 cB1/nbar (same for cB2).
        """
        tree = [(name, terms, 0, fac)
                for name, terms, fac in self._BASE_TREE_TERMS]
        stoch = [(name, s, 0, fac) for name, s, fac in self._BASE_STOCH_PARTS]
        if self._has_ctr() and self.cnlo_type == 'EggLeeSco':
            tree += [('B0L_cnloB' if name == 'B0L_id' else name + 'cnloB',
                      terms, 1, fac + ('cnloB',))
                     for name, terms, fac in self._BASE_TREE_TERMS]
            stoch += [(name + 'cnloB', s, 1, fac + ('cnloB',))
                      for name, s, fac in self._BASE_STOCH_PARTS]
        elif self._has_ctr() and self.cnlo_type == 'IvaPhiNis':
            # Linear in cB: (a with a b1, a without b1) for each counterterm.
            for c, a1, a0 in (('cB1', 3, 4), ('cB2', 5, 6)):
                cf = (c,)
                tree += [
                    ('B0L_b1b1' + c, ((a1, 0),), 0, ('b1', 'b1') + cf),
                    ('B0L_b1' + c, ((a0, 0), (a1, 3)), 0, ('b1',) + cf),
                    ('B0L_' + c, ((a0, 3),), 0, cf),
                    ('B0L_b1b2' + c, ((a1, 1),), 0, ('b1', 'b2') + cf),
                    ('B0L_b2' + c, ((a0, 1),), 0, ('b2',) + cf),
                    ('B0L_b1g2' + c, ((a1, 2),), 0, ('b1', 'g2') + cf),
                    ('B0L_g2' + c, ((a0, 2),), 0, ('g2',) + cf),
                ]
            for c, a, cf in (('cB1cB1', 7, ('cB1', 'cB1')),
                             ('cB1cB2', 8, ('cB1', 'cB2')),
                             ('cB2cB2', 9, ('cB2', 'cB2'))):
                tree += [
                    ('B0L_b1' + c, ((a, 0),), 0, ('b1',) + cf),
                    ('B0L_b2' + c, ((a, 1),), 0, ('b2',) + cf),
                    ('B0L_g2' + c, ((a, 2),), 0, ('g2',) + cf),
                    ('B0L_' + c, ((a, 3),), 0, cf),
                ]
            for c, s_mb, s_np in (('cB1', 3, 4), ('cB2', 5, 6)):
                stoch += [
                    ('Bnoise_MB0b1' + c, s_mb, 0, ('MB0', 'b1', c, '1/nbar')),
                    ('Bnoise_NP0' + c, s_np, 0, ('NP0', c, '1/nbar')),
                ]

        self._tree_specs = tuple((terms, ctr) for _, terms, ctr, _ in tree)
        self._stoch_specs = tuple((s, ctr) for _, s, ctr, _ in stoch)
        self.tree_diagrams = tuple(name for name, _, _, _ in tree)
        self.tree_index = {d: i for i, d in enumerate(self.tree_diagrams)}
        # `Bnoise_NB0` is analytic and must stay last.
        self.stoch_diagrams = tuple(name for name, _, _, _ in stoch) \
            + ('Bnoise_NB0',)
        self.stoch_index = {d: i for i, d in enumerate(self.stoch_diagrams)}
        self._diagram_factors = {name: fac for name, _, _, fac in tree + stoch}
        self._diagram_factors['Bnoise_NB0'] = self._NB0_FACTORS
        # Diagrams whose coefficient is linear in each analytically
        # marginalisable parameter, as in `PTEmu.diagrams_to_marg`. The
        # counterterms (cnloB, cB1, cB2) and b2, g2 are sampled instead.
        self.diagrams_to_marg = {
            p: [d for d in self.tree_diagrams + self.stoch_diagrams
                if p in self._diagram_factors[d]]
            for p in self._TCM_AAP_POWER}

    def diagram_coeffs(self, params, X_list=None):
        """Coefficients multiplying the diagrams of `BX_ell_Sugi` and
        `BX_ell_Scocc`, such that ``Bell = sum_X coeff[X] * BX_ell[X]``.

        Parameters
        ----------
        params : dict
            COMET parameter dictionary (single-z or batched-z).
        X_list : str or iterable of str, optional
            Diagram names. Defaults to all diagrams of the current model.

        Returns
        -------
        If ``X_list`` is **None**, ``{diagram_name: ndarray}`` with values
        squeezed to scalars for single-z and shape ``(nparams,)`` for
        batched-z. Otherwise an array of shape ``(nx, nparams)`` for
        batched-z and ``(nx,)`` for single-z, ordered as ``X_list``.
        """
        names = (self.tree_diagrams + self.stoch_diagrams if X_list is None
                 else [X_list] if isinstance(X_list, str) else list(X_list))
        coeffs = [self._coeff_product(self._diagram_factors_of(name), params)
                  for name in names]
        if X_list is None:
            return {name: np.squeeze(c) for name, c in zip(names, coeffs)}
        coeffs = np.array(coeffs)
        return coeffs[:, 0] if coeffs.shape[1] == 1 else coeffs

    def _get_bias_coeff_for_AM(self, params, diagrams_to_marg):
        """Coefficients of the diagrams to marginalise, with the
        marginalised parameter stripped. Mirrors
        `PTEmu._get_bias_coeff_for_AM`.

        Each diagram in ``diagrams_to_marg`` must belong to one entry of
        `diagrams_to_marg`; its returned coefficient is the derivative of
        its `diagram_coeffs` entry with respect to that parameter (e.g.
        ``Bnoise_MB0b1cnloB`` -> ``b1 cnloB / nbar``). With the 'TCM' or
        'TCMlinear' reparametrisation the derivative is taken with respect
        to the TCM parameter, i.e. divided by ``Aap^n``.

        Returns
        -------
        ndarray of shape ``(n_diag, nparams)``, squeezed.
        """
        coeffs = []
        for name in diagrams_to_marg:
            factors = list(self._diagram_factors_of(name))
            marg = [p for p in self.diagrams_to_marg
                    if name in self.diagrams_to_marg[p]]
            if not marg:
                raise ValueError(
                    "Diagram '{}' is not linear in any of the analytically "
                    "marginalisable parameters {}.".format(
                        name, list(self.diagrams_to_marg)))
            factors.remove(marg[0])
            c = self._coeff_product(factors, params)
            if self.reparametrisation in ('TCM', 'TCMlinear'):
                Aap = 1.0 / (np.atleast_1d(params['q_tr'])**2
                             * np.atleast_1d(params['q_lo']))
                c = c / Aap**self._TCM_AAP_POWER[marg[0]]
            coeffs.append(c)
        return np.array(coeffs).squeeze()

    def _diagram_factors_of(self, name):
        if name not in self._diagram_factors:
            raise ValueError(
                "Unknown diagram '{}'. Valid names: {}.".format(
                    name, sorted(self._diagram_factors)))
        return self._diagram_factors[name]

    def _coeff_product(self, factors, params):
        """Product of the coefficient ``factors`` as a ``(nparams,)`` array."""
        nparams = np.atleast_1d(params['q_lo']).size
        inv_nbar = np.broadcast_to(
            1.0 / np.atleast_1d(np.asarray(self.nbar, dtype=float)),
            (nparams,))
        c = np.ones(nparams)
        for fac in factors:
            c = c * np.atleast_1d(self._COEFF_FACTORS[fac](params, inv_nbar))
        return c

    def _diagram_names(self, X_list):
        """**X_list** as a list of diagram names (all diagrams if **None**),
        checking that they are valid."""
        valid = self.tree_diagrams + self.stoch_diagrams
        if X_list is None:
            return list(valid)
        X_list = [X_list] if isinstance(X_list, str) else list(X_list)
        for x in X_list:
            if x not in valid:
                raise ValueError(
                    "Unknown diagram '{}' in X_list. Valid names: {}."
                    .format(x, sorted(valid)))
        return X_list

    def _diagram_kernel_tables(self, tree_keep, stoch_keep):
        """Static specs (JAX) and flat term tables (numba) for the requested
        diagram indices. Output columns follow the order of ``*_keep``."""
        tree_spec = tuple(self._tree_specs[d] for d in tree_keep)
        stoch_spec = tuple(self._stoch_specs[d] for d in stoch_keep)
        cols, a_idx, b_idx = [], [], []
        for c, (terms, _) in enumerate(tree_spec):
            for a, b in terms:
                cols.append(c); a_idx.append(a); b_idx.append(b)
        tables = (np.array(cols, dtype=np.int64),
                  np.array(a_idx, dtype=np.int64),
                  np.array(b_idx, dtype=np.int64),
                  np.array([ctr for _, ctr in tree_spec], dtype=np.int64),
                  np.array([s for s, _ in stoch_spec], dtype=np.int64),
                  np.array([ctr for _, ctr in stoch_spec], dtype=np.int64))
        return tree_spec, stoch_spec, tables

    @staticmethod
    def _format_BX(BX, names, as_dict):
        """Format the diagram projections ``{ell: (n, nx, nparams)}``, as
        ``{ell: {name: ndarray}}`` squeezed if **as_dict**, and otherwise as
        ``{ell: ndarray}`` without the parameter axis for single-z."""
        if as_dict:
            return {ll: {name: np.squeeze(B[:, ix])
                         for ix, name in enumerate(names)}
                    for ll, B in BX.items()}
        return {ll: B[..., 0] if B.shape[-1] == 1 else B
                for ll, B in BX.items()}

    # Sugiyama multipoles

    def Bell_Sugi(self, pair, Pdw_eval, params,
                  ell=((0, 0, 0), (2, 0, 2)),
                  nmu1=5, nmu12=12, nphi=5,
                  mu12_transform='quadratic'):
        """Bispectrum Sugiyama multipoles via numerical 3D quadrature, with
        the fused kernels of `Bell_Sugi_factorized`.

        Returns
        -------
        dict
            ``{(l1, l2, L): ndarray}`` with shape ``(n_pair,)`` for
            single-z and ``(n_pair, nparams)`` for batched-z.
        """
        self._check_model('Bell_Sugi')
        out = self.Bell_Sugi_factorized(
            pair, Pdw_eval, params, ell, nmu1=nmu1, nmu12=nmu12, nphi=nphi,
            mu12_transform=mu12_transform)
        return {ll: np.squeeze(B) for ll, B in out.items()}

    def BX_ell_Sugi(self, pair, Pdw_eval, params,
                    ell=((0, 0, 0), (2, 0, 2)),
                    nmu1=5, nmu12=12, nphi=5,
                    mu12_transform='quadratic',
                    X_list=None):
        """Diagram-resolved Sugiyama bispectrum multipoles, with the fused
        kernels of `Bell_Sugi_factorized`.

        If ``X_list`` is **None** (default), returns
        ``{(l1, l2, L): {diagram_name: ndarray}}`` where each diagram value
        is the contribution stripped of its bias coefficient (same
        convention as `BX_ell`). Single-z output is squeezed to
        ``(n_pair,)``; batched-z output is ``(n_pair, nparams)``.

        If ``X_list`` is provided (a string or iterable of diagram names),
        returns ``{(l1, l2, L): ndarray}`` where the array has shape
        ``(n_pair, nx, nparams)`` for batched-z and ``(n_pair, nx)`` for
        single-z, with the diagram axis ordered to match ``X_list``. Only
        the requested diagrams are computed.

        The `Bnoise_NB0` slice is `1/qiso6` for `(0, 0, 0)` and zero
        otherwise, matching the analytical path.
        """
        self._check_model('BX_ell_Sugi')
        names = self._diagram_names(X_list)
        BX = self.Bell_Sugi_factorized(
            pair, Pdw_eval, params, ell, nmu1=nmu1, nmu12=nmu12, nphi=nphi,
            mu12_transform=mu12_transform, X_list=names)
        return self._format_BX(BX, names, X_list is None)

    def Bell_Sugi_factorized(self, pair, Pdw_eval, params, ell,
                             nmu1=5, nmu12=12, nphi=5,
                             mu12_transform='quadratic', X_list=None,
                             dedupe=True):
        r"""Sugiyama multipoles for many ``(l1, l2, L)`` at once.

        Same quadrature and conventions as `Bell_Sugi` (``X_list=None``) and
        `BX_ell_Sugi` (``X_list`` given), but with the factorised projection
        of `_sugi_factorized_plan`, evaluated by fused kernels that never
        build the 5D quadrature grid (memory only grows with the number of
        pairs and moments). The cost barely grows with the number of
        multipoles. ``Pdw_eval`` must provide ``ppoly_table()`` (see
        `PTEmu._get_pdw_fn`). Used by `Bell_Sugi` and `BX_ell_Sugi`.

        Each pair is only evaluated once up to the order of ``(k1, k2)``
        (see `_sugi_unique_pairs`): the pairs with ``k1 > k2``, or repeated,
        are taken from the multipoles at ``k1 <= k2``, using
        :math:`B_{\ell_1\ell_2L}(k_2,k_1)=B_{\ell_2\ell_1L}(k_1,k_2)`. On
        the full ``(k1, k2)`` grid of a window convolution, this halves the
        cost. ``dedupe=False`` skips this and its check, for pairs with
        ``k1 <= k2`` and (almost) no repeats. The multipoles with odd
        :math:`\ell_1+\ell_2` vanish by parity and are returned as zeros.

        Returns
        -------
        dict ``{(l1, l2, L): ndarray}`` with shape ``(n_pair, nparams)`` if
        ``X_list`` is **None**, and ``(n_pair, nx, nparams)`` otherwise, with
        the diagram axis ordered as ``X_list``. Never squeezed.
        """
        self._check_model('Bell_Sugi_factorized')

        ell = tuple(tuple(int(x) for x in ll) for ll in ell)
        pair = np.atleast_2d(pair)
        uniq = self._sugi_unique_pairs(pair) if dedupe else None
        if uniq is None:
            return self._bell_sugi_pairs(pair, Pdw_eval, params, ell, nmu1,
                                         nmu12, nphi, mu12_transform, X_list)
        upair, src = uniq
        B = self._bell_sugi_pairs(upair, Pdw_eval, params,
                                  self._sugi_swapped_ell(ell), nmu1, nmu12,
                                  nphi, mu12_transform, X_list)
        return {ll: np.concatenate((B[ll], B[(ll[1], ll[0], ll[2])]))[src]
                for ll in ell}

    # Deduplicate only if at most this fraction of the pairs is left
    _SUGI_DEDUPE_MAX_FRAC = 0.9

    @staticmethod
    def _sugi_swapped_ell(ell):
        """``ell`` together with its ``(l2, l1, L)`` swaps, needed to fill the
        pairs with ``k1 > k2`` (see `_sugi_unique_pairs`)."""
        return tuple(sorted(set(ell) | {(l2, l1, L) for (l1, l2, L) in ell}))

    @staticmethod
    def _sugi_unique_pairs(pair):
        """Pairs to evaluate, up to the order of ``(k1, k2)`` and exact
        repeats.

        Returns **None** if fewer than 10% of the pairs would be saved
        (`_SUGI_DEDUPE_MAX_FRAC`): the swapped multipoles can need extra
        moments, which cost more than such a small saving. Otherwise returns
        ``(upair, src)``: the ``(nu, 2)`` distinct pairs with ``k1 <= k2``,
        and for every input pair the row of ``concatenate((B_l1l2L,
        B_l2l1L))`` (both evaluated at ``upair``) that gives its
        ``B_l1l2L``, i.e. ``i`` if it equals ``upair[i]`` and ``nu + i`` if
        it equals ``upair[i]`` swapped.

        Pure numpy on the (static) pairs, so the JAX path (`PTEmu._jax_sugi`)
        uses it at trace time and applies ``src`` as a constant gather.

        A square grid closed under the swap (``pair[i, j] == pair[j, i]``
        swapped once reshaped to ``(m, m, 2)``, e.g. a meshgrid of the same
        k for k1 and k2) is recognised in O(n), a few microseconds. Other
        pairs are sorted (``lexsort``), about 50 us for 1500 pairs and 1 ms
        for 1e4, i.e. 1-3% of their bispectrum.
        """
        pair = np.asarray(pair)
        n = pair.shape[0]
        if n < 2 or pair.ndim != 2 or pair.shape[1] != 2:
            return None
        m = math.isqrt(n)
        if m*m == n:
            sq = pair.reshape(m, m, 2)
            if np.array_equal(sq[:, :, 0], sq[:, :, 1].T):
                # (i, j) from the upper triangle, (j, i) from its swap; the
                # diagonal is assigned last, so it is not swapped
                iu, ju = np.triu_indices(m)
                ntri = iu.size
                src = np.empty((m, m), dtype=np.intp)
                src[ju, iu] = np.arange(ntri, 2*ntri)
                src[iu, ju] = np.arange(ntri)
                return sq[iu, ju], src.ravel()
        k1, k2 = pair[:, 0], pair[:, 1]
        swapped = k1 > k2
        lo = np.where(swapped, k2, k1)
        hi = np.where(swapped, k1, k2)
        order = np.lexsort((hi, lo))
        lo_s, hi_s = lo[order], hi[order]
        new = np.empty(n, dtype=bool)
        new[0] = True
        np.not_equal(lo_s[1:], lo_s[:-1], out=new[1:])
        new[1:] |= hi_s[1:] != hi_s[:-1]
        nu = int(np.count_nonzero(new))
        if nu > BispectrumNum._SUGI_DEDUPE_MAX_FRAC * n:
            return None
        group = np.cumsum(new) - 1
        src = np.empty(n, dtype=np.intp)
        src[order] = group
        src += nu * swapped
        upair = np.column_stack((lo_s[new], hi_s[new]))
        return upair, src

    def _bell_sugi_pairs(self, pair, Pdw_eval, params, ell, nmu1, nmu12,
                         nphi, mu12_transform, X_list):
        """`Bell_Sugi_factorized` evaluated at every pair, as given."""
        n_pair = pair.shape[0]
        _, LM, combos, terms = self._sugi_factorized_plan(ell)
        qlo = np.atleast_1d(params['q_lo'])
        qtr = np.atleast_1d(params['q_tr'])
        nparams = qlo.size
        qiso6 = qlo**2 * qtr**4

        if X_list is None:
            if combos:
                moments, qiso6 = self._sugi_moments(
                    pair, Pdw_eval, params, LM, combos, nmu1, nmu12, nphi,
                    mu12_transform)
            else:
                moments = np.zeros((n_pair, 0, nparams))
            proj = self._sugi_factorized_combine(terms, moments)
            NB0 = np.atleast_1d(params['NB0'])
            out = {}
            for ll in ell:
                out[ll] = proj[ll]
                if ll == (0, 0, 0):
                    out[ll] = out[ll] \
                        + (NB0 / self.nbar**2)[None, :] / qiso6[None, :]
            return out

        X_list = self._diagram_names(X_list)
        nb0_name = self.stoch_diagrams[-1]
        tree_set = set(self.tree_diagrams)
        # Moments are only needed for the diagrams other than NB0 (constant)
        integrate = bool(combos) and any(x != nb0_name for x in X_list)
        if integrate:
            (mt, ms, tree_keep, stoch_keep), qiso6 = self._sugi_moments(
                pair, Pdw_eval, params, LM, combos, nmu1, nmu12, nphi,
                mu12_transform, X_list=X_list)
            spt_proj = self._sugi_factorized_combine(terms, mt)
            stoch_proj = self._sugi_factorized_combine(terms, ms)
            tree_pos = {orig: k for k, orig in enumerate(tree_keep)}
            stoch_pos = {orig: k for k, orig in enumerate(stoch_keep)}

        BX_dict = {}
        for ll in ell:
            res = np.zeros((n_pair, len(X_list), nparams))
            for ix, name in enumerate(X_list):
                if name == nb0_name:
                    res[:, ix, :] = 1.0 / qiso6[None, :] \
                        if ll == (0, 0, 0) else 0.0
                elif not integrate:
                    continue
                elif name in tree_set:
                    res[:, ix, :] = \
                        spt_proj[ll][..., tree_pos[self.tree_index[name]]]
                else:
                    res[:, ix, :] = \
                        stoch_proj[ll][..., stoch_pos[self.stoch_index[name]]]
            BX_dict[ll] = res
        return BX_dict

    # Sugiyama quadrature and factorised projection

    # Transforms sampling k3 per (k1, k2) pair instead of mu12
    _K3_TRANSFORMS = ('k3', 'logk3')

    # Linear core of the 'logk3' map, in h/Mpc, close to the peak of P(k)
    _LOGK3_SCALE = 0.05

    def _sugi_get_quadrature(self, nmu1, nmu12, nphi, mu12_transform):
        """Build (and cache) the (mu1, mu12, phi) quadrature grids.

        Supported ``mu12_transform`` modes:
        - ``'linear'``: Gauss-Legendre nodes in mu12 directly.
        - ``'quadratic'``: clusters nodes near mu12 = -1 to capture the
          colinear (k3 -> 0) limit.
        - ``'quartic'``: stronger clustering near mu12 = -1.
        - ``'k3'``: sample k3 uniformly per (k1, k2) pair instead of mu12.
          Returns Gauss-Legendre t-nodes in [-1, 1]; the caller maps
          ``k3(t; k1, k2) = max(k1,k2) + min(k1,k2)*t`` per pair, derives
          ``mu12 = (k3^2 - k1^2 - k2^2)/(2 k1 k2)``, and folds the
          Jacobian ``dmu12/dt = min(k1,k2) * k3/(k1*k2)`` into the
          projection kernel. This regularises the colinear region by
          construction.
        - ``'logk3'``: as ``'k3'``, but with the k3 nodes log-spaced above
          ``|k1 - k2|`` with a linear core of size ``_LOGK3_SCALE`` (see
          `_k3_map`). This resolves the peak of P(k3) at small k3 for large
          k1 ~ k2, and needs about half the mu12 nodes of ``'k3'`` for the
          same accuracy.
        """
        key = (nmu1, nmu12, nphi, mu12_transform)
        cached = self._sugi_quad_cache.get(key)
        if cached is not None:
            return cached

        mu1, w_mu1 = np.polynomial.legendre.leggauss(nmu1)
        x_mu12, w_x_mu12 = np.polynomial.legendre.leggauss(nmu12)
        if mu12_transform == 'linear':
            mu12 = x_mu12
            w_mu12 = w_x_mu12
        elif mu12_transform == 'quadratic':
            # Resolves the k3 ~ 0 singularity when k1 ~ k2 by clustering nodes
            # near mu12 = -1, where (k1+k2*mu12) collapses.
            mu12 = 0.5 * (x_mu12 + 1.0)**2 - 1.0
            w_mu12 = w_x_mu12 * (x_mu12 + 1.0)
        elif mu12_transform == 'quartic':
            mu12 = 0.125 * (x_mu12 + 1.0)**4 - 1.0
            w_mu12 = w_x_mu12 * 0.5 * (x_mu12 + 1.0)**3
        elif mu12_transform in self._K3_TRANSFORMS:
            # mu12 is per-pair; the values stashed here are the underlying
            # t-nodes in [-1, 1] (the caller maps them to per-pair mu12).
            mu12 = x_mu12
            w_mu12 = w_x_mu12
        else:
            raise ValueError(
                f"Unsupported mu12_transform: {mu12_transform!r}")

        phi = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)
        w_phi = 2.0 * np.pi / nphi

        # Shape: (1, nmu1, 1, 1), (1, 1, nmu12, 1), (1, 1, 1, nphi)
        mu1_g = mu1[None, :, None, None]
        w_mu1_g = w_mu1[None, :, None, None]
        mu12_g = mu12[None, None, :, None]
        w_mu12_g = w_mu12[None, None, :, None]
        cphi_g = np.cos(phi)[None, None, None, :]
        phi_g = phi[None, None, None, :]
        w_phi_g = w_phi * np.ones_like(cphi_g)

        out = (mu1_g, w_mu1_g, mu12_g, w_mu12_g, cphi_g, phi_g, w_phi_g)
        self._sugi_quad_cache[key] = out
        return out

    def _k3_map(self, t, k_lo, k_hi, mu12_transform):
        """Map t in [-1, 1] -> (k3, dk3/dt) for the per-pair-k3 transforms.

        ``'k3'`` uses the linear map of `_k3_from_t`. ``'logk3'`` uses
        ``k3 = k_lo + s (exp(x ln(1 + (k_hi - k_lo)/s)) - 1)``, with
        ``x = (1 + t)/2`` and ``s`` equal to `_LOGK3_SCALE` (converted to
        1/Mpc with h = 0.7 if ``use_Mpc``; its precise value is not
        important).
        """
        if mu12_transform == 'k3':
            return self._k3_from_t(t, k_lo, k_hi)
        scale = self._LOGK3_SCALE * (0.7 if self.use_Mpc else 1.0)
        x = 0.5 * (1.0 + t)
        lnr = np.log1p((k_hi - k_lo) / scale)
        u = scale * np.expm1(x * lnr)
        return k_lo + u, 0.5 * lnr * (u + scale)

    @staticmethod
    def _k3_from_t(t, k_lo, k_hi):
        """Map t in [-1, 1] -> (k3, dk3/dt).
        All arrays broadcast against `t * (k_lo, k_hi)`. The Jacobian
        ``dk3/dt`` is needed by the per-pair projection weights (folded via
        ``dmu12/dt = (dk3/dt) * k3 / (k1 k2)``).
        """
        mid = 0.5 * (k_hi + k_lo)
        half = 0.5 * (k_hi - k_lo)
        k3 = mid + half * t
        dk3_dt = half * np.ones_like(t)
        return k3, dk3_dt

    @staticmethod
    def _wigner_3j_000M(l1, l2, L, M, _cache={}):
        """Cached ``(wigner_3j(l1, l2, L, 0, 0, 0), wigner_3j(l1, l2, L, 0, -M, M))``."""
        key = (l1, l2, L, M)
        if key not in _cache:
            from sympy.physics.wigner import wigner_3j
            _cache[key] = (float(wigner_3j(l1, l2, L, 0, 0, 0).evalf()),
                           float(wigner_3j(l1, l2, L, 0, -M, M).evalf()))
        return _cache[key]

    def _sugi_factorized_plan(self, ell):
        """Factorised Sugiyama projection, as moments of the bispectrum.

        Pairing the ``+-M`` terms of the Sugiyama kernel
        (``l1 + l2 + L`` even) gives

        ``B_l1l2L = h (2l1+1)(2l2+1)(2L+1)/(8pi) sum_{M>=0} (2-delta_M0)
        (-1)^M w3j(l1,l2,L;0,-M,M) int n_l2M P_l2^M(mu12) n_LM P_L^M(mu1)
        cos(M phi) B``,

        with ``n_lm = sqrt((l-m)!/(l+m)!)``. The integral only depends on
        ``(l2, L, M)``: these moments are computed once (see
        `_fused_moments_njit`) and ``l1`` only enters through the Wigner-3j
        coefficients, so many multipoles cost about as much as one.

        Returns
        -------
        ell : tuple of (l1, l2, L)
        LM : list of the (L, M) angular modes
        combos : list of the (l2, L, M) moments
        terms : dict ``{(l1, l2, L): [(index in combos, coefficient), ...]}``
        """
        ell = tuple(sorted(set(tuple(int(x) for x in ll) for ll in ell)))
        LM, combos, terms = [], [], {}
        for ll in ell:
            l1, l2, L = ll
            terms[ll] = []
            # odd l1 + l2 vanish by parity (the projection below assumes it)
            if (l1 + l2) % 2:
                continue
            h = self._wigner_3j_000M(l1, l2, L, 0)[0]
            if h == 0.0:
                continue
            prefactor = h * (2*l1 + 1) * (2*l2 + 1) * (2*L + 1) / (8.0*np.pi)
            for M in range(min(l2, L) + 1):
                w3j = self._wigner_3j_000M(l1, l2, L, M)[1]
                if w3j == 0.0:
                    continue
                if (L, M) not in LM:
                    LM.append((L, M))
                if (l2, L, M) not in combos:
                    combos.append((l2, L, M))
                terms[ll].append((combos.index((l2, L, M)),
                                  prefactor * (2.0 - (M == 0)) * (-1.0)**M
                                  * w3j))
        return ell, LM, combos, terms

    def _sugi_factorized_weights(self, LM, combos, pair, nmu1, nmu12, nphi,
                                 mu12_transform):
        """Per-pair geometry and projection weights of the moments.

        Returns
        -------
        k3, mu12 : ndarray, shape (n_pair, nmu12)
            Third side and cosine of each ``mu12`` node.
        ang : ndarray, shape (nLM, nmu1, nphi)
            ``n_LM P_L^M(mu1) cos(M phi)`` times the ``(mu1, phi)`` weights.
        ang_id : ndarray, shape (nq,)
            Row of ``ang`` used by each moment in ``combos``.
        rad : ndarray, shape (n_pair, nq, nmu12)
            ``n_l2M P_l2^M(mu12)`` times the ``mu12`` weights (and, for the
            per-pair-k3 transforms, the ``dmu12/dt`` Jacobian).
        """
        pair = np.atleast_2d(pair)
        n_pair = pair.shape[0]
        mu1_g, w_mu1_g, mu12_g, w_mu12_g, _, phi_g, w_phi_g = \
            self._sugi_get_quadrature(nmu1, nmu12, nphi, mu12_transform)
        mu1, w_mu1 = mu1_g.ravel(), w_mu1_g.ravel()
        phi, w_phi = phi_g.ravel(), w_phi_g.ravel()
        k1 = pair[:, 0][:, None]
        k2 = pair[:, 1][:, None]
        if mu12_transform in self._K3_TRANSFORMS:
            k3, dk3_dt = self._k3_map(mu12_g.ravel()[None, :],
                                      np.abs(k1 - k2), k1 + k2,
                                      mu12_transform)
            k3 = np.maximum(k3, 1e-30)
            mu12 = np.clip((k3*k3 - k1*k1 - k2*k2) / (2.0 * k1 * k2),
                           -1.0, 1.0)
            w_mu12 = w_mu12_g.ravel()[None, :] * dk3_dt * k3 / (k1 * k2)
        else:
            mu12 = np.broadcast_to(mu12_g.ravel(), (n_pair, nmu12))
            k3 = np.maximum(np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12), 1e-30)
            w_mu12 = np.broadcast_to(w_mu12_g.ravel(), (n_pair, nmu12))

        def _n(l, m):
            return np.sqrt(factorial(l - m) / factorial(l + m))

        ang = np.stack([np.outer(_n(L, M) * lpmv(M, L, mu1) * w_mu1,
                                 np.cos(M*phi) * w_phi) for L, M in LM])
        ang_id = np.array([LM.index((L, M)) for _, L, M in combos],
                          dtype=np.int64)
        rad = np.stack([_n(l2, M) * lpmv(M, l2, mu12) * w_mu12
                        for l2, _, M in combos], axis=1)
        return (np.ascontiguousarray(k3), np.ascontiguousarray(mu12),
                np.ascontiguousarray(ang), ang_id, np.ascontiguousarray(rad))

    # Number of pair sets whose weights are kept (the 3PCF k-grid weights
    # take ~70 MB for the default settings)
    _SUGI_WEIGHTS_CACHE_SIZE = 2

    def _sugi_cached_weights(self, LM, combos, pair, nmu1, nmu12, nphi,
                             mu12_transform):
        """`_sugi_factorized_weights`, cached: they only depend on the pairs,
        the multipoles and the quadrature, not on the parameters, so repeated
        calls on the same pairs (e.g. the FFTLog grid of the 3PCF) reuse
        them."""
        key = (tuple(LM), tuple(combos), nmu1, nmu12, nphi, mu12_transform,
               self.use_Mpc, pair.shape)
        hit = self._sugi_weights_cache.get(key)
        if hit is not None and np.array_equal(hit[0], pair):
            return hit[1]
        out = self._sugi_factorized_weights(LM, combos, pair, nmu1, nmu12,
                                            nphi, mu12_transform)
        if len(self._sugi_weights_cache) >= self._SUGI_WEIGHTS_CACHE_SIZE:
            self._sugi_weights_cache.pop(next(iter(self._sugi_weights_cache)))
        self._sugi_weights_cache[key] = (pair.copy(), out)
        return out

    @staticmethod
    def _sugi_factorized_combine(terms, moments):
        """Sugiyama multipoles from the moments (n_pair, nq, ...) of
        `_sugi_factorized_weights`, see `_sugi_factorized_plan`."""
        out = {}
        for ll, t in terms.items():
            B = np.zeros(moments.shape[:1] + moments.shape[2:])
            for iq, coeff in t:
                B += coeff * moments[:, iq]
            out[ll] = B
        return out

    def _sugi_moments(self, pair, Pdw_eval, params, LM, combos, nmu1, nmu12,
                      nphi, mu12_transform, X_list=None):
        """Moments of the bispectrum (or of its diagrams) for all pairs, see
        `_fused_moments`, with the weights of `_sugi_factorized_weights`."""
        pair = np.atleast_2d(pair)
        k3, mu12, ang, ang_id, rad = self._sugi_cached_weights(
            LM, combos, pair, nmu1, nmu12, nphi, mu12_transform)
        mu1_g, _, _, _, cphi_g, _, _ = self._sugi_get_quadrature(
            nmu1, nmu12, nphi, mu12_transform)
        return self._fused_moments(pair[:, 0], pair[:, 1], k3, mu12,
                                   mu1_g.ravel(), cphi_g.ravel(), ang, ang_id,
                                   rad, Pdw_eval, params, X_list=X_list)

    # Scoccimarro multipoles

    def Bell_Scocc(self, tri, Pdw_eval, params, ell=((0, 0), (2, 0)),
                   norm='sphharm', nmu=5, nphi=5):
        """Scoccimarro bispectrum multipoles via numerical 2D quadrature,
        with the fused kernels (see `_scocc_moments`).

        Parameters
        ----------
        tri : ndarray, shape (n_tri, 3)
            Triangle wavemodes ``(k1, k2, k3)``.
        Pdw_eval : callable
            Pdw evaluator providing ``ppoly_table()`` (see
            `PTEmu._get_pdw_fn`).
        params : dict
            COMET parameter dictionary (single-z or batched-z).
        ell : iterable of (l, m) tuples
        norm: str
            Normalisation convention for the multipoles. Options are
            'sphharm' (default), which gives the standard spherical harmonic
            normalisation, and 'legendre', which gives the normalisation used
            in Legendre multipoles.
        nmu, nphi : int
        Returns
        -------
        dict ``{(l, m): ndarray}`` with shape ``(n_tri,)`` for single-z and
        ``(n_tri, nparams)`` for batched-z. NB0 contributes only to ``(0, 0)``.
        """
        if self.real_space:
            raise NotImplementedError('Numerical quadrature is not implemented in real space yet.')
        self._check_model('Bell_Scocc')

        ell = tuple(tuple(ll) for ll in ell)
        moments, qiso6 = self._scocc_moments(
            tri, Pdw_eval, params, ell, nmu, nphi)
        NB0 = np.atleast_1d(params['NB0'])
        inv_nbar = np.broadcast_to(1.0 / np.atleast_1d(self.nbar),
                                   qiso6.shape)

        Bell_dict = {}
        for i, ll in enumerate(ell):
            B = moments[:, i]
            if ll == (0, 0):
                # Y_00 = 1/sqrt(4pi): a constant c projects to sqrt(4pi)*c.
                B = B + np.sqrt(4.0*np.pi) \
                    * (NB0 * inv_nbar**2)[None, :] / qiso6[None, :]
            Bell_dict[ll] = self._scocc_normfact(ll, norm) * np.squeeze(B)
        return Bell_dict

    def BX_ell_Scocc(self, tri, Pdw_eval, params,
                     ell=((0, 0), (2, 0)), norm='sphharm', nmu=5, nphi=5,
                     X_list=None):
        """Diagram-resolved Scoccimarro bispectrum multipoles.

        If ``X_list`` is **None** (default), returns
        ``{(l, m): {diagram_name: ndarray}}`` with values squeezed to
        ``(n_tri,)`` for single-z and ``(n_tri, nparams)`` for batched-z.

        If ``X_list`` is provided, returns ``{(l, m): ndarray}`` with shape
        ``(n_tri, nx, nparams)`` for batched-z and ``(n_tri, nx)`` for
        single-z. The diagram axis is ordered to match ``X_list`` and only
        the requested diagrams are computed.
        """
        if self.real_space:
            raise NotImplementedError('Numerical quadrature is not implemented in real space yet.')
        self._check_model('BX_ell_Scocc')

        ell = tuple(tuple(ll) for ll in ell)
        names = self._diagram_names(X_list)
        nb0_name = self.stoch_diagrams[-1]
        tri = np.atleast_2d(tri)
        qiso6 = np.atleast_1d(params['q_lo'])**2 \
            * np.atleast_1d(params['q_tr'])**4

        res = np.zeros((tri.shape[0], len(ell), len(names), qiso6.size))
        if any(x != nb0_name for x in names):
            (mt, ms, tree_keep, stoch_keep), qiso6 = self._scocc_moments(
                tri, Pdw_eval, params, ell, nmu, nphi, X_list=names)
            for ix, name in enumerate(names):
                if name in self.tree_index:
                    res[:, :, ix] = mt[..., tree_keep.index(
                        self.tree_index[name])]
                elif name != nb0_name:
                    res[:, :, ix] = ms[..., stoch_keep.index(
                        self.stoch_index[name])]

        BX = {}
        for i, ll in enumerate(ell):
            B = res[:, i]
            if ll == (0, 0):
                # Y_00 = 1/sqrt(4pi): a constant c projects to sqrt(4pi)*c.
                for ix, name in enumerate(names):
                    if name == nb0_name:
                        B[:, ix] = np.sqrt(4.0*np.pi) / qiso6[None, :]
            BX[ll] = self._scocc_normfact(ll, norm) * B
        return self._format_BX(BX, names, X_list is None)

    @staticmethod
    def _scocc_normfact(ll, norm):
        """Factor converting the ``'sphharm'`` normalisation to **norm**."""
        if norm == 'legendre':
            return np.sqrt((2*ll[0] + 1)/(4*np.pi))
        return 1.0

    # Scoccimarro quadrature and projection

    def _scocc_get_quadrature(self, nmu, nphi):
        """Build (and cache) the (mu, phi) quadrature grids for Scoccimarro multipoles."""
        key = (nmu, nphi)
        cached = self._scocc_quad_cache.get(key)
        if cached is not None:
            return cached

        mu, w_mu = np.polynomial.legendre.leggauss(nmu)
        phi = np.linspace(0.0, 2.0*np.pi, nphi, endpoint=False)
        w_phi = 2.0 * np.pi / nphi

        # Shape: (1, nmu, 1), (1, 1, nphi)
        mu_g = mu[None, :, None]
        w_mu_g = w_mu[None, :, None]
        cphi_g = np.cos(phi)[None, None, :]
        sphi_g = np.sin(phi)[None, None, :]
        phi_g = phi[None, None, :]
        w_phi_g = w_phi * np.ones_like(cphi_g)

        out = (mu_g, w_mu_g, cphi_g, sphi_g, phi_g, w_phi_g)
        self._scocc_quad_cache[key] = out
        return out

    @staticmethod
    def _sph_harm_real(l, m, costheta, phi, normalized=True):
        norm = np.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))
        norm = norm * (-1)**(0.5 * (m - abs(m)))
        if normalized:
            norm = norm * np.sqrt((2*l + 1)/(4*np.pi))
        if m > 0:
            return np.sqrt(2) * norm * lpmv(m, l, costheta) * np.cos(m * phi)
        elif m < 0:
            return np.sqrt(2) * norm * lpmv(-m, l, costheta) * np.sin(-m * phi)
        else:
            return norm * lpmv(0, l, costheta)

    def _scocc_get_proj_ops(self, ell, nmu, nphi):
        """Compute and cache the ``(l, m)`` Scoccimarro projection operators.

        For each ``(l, m)`` the kernel is
        ``sign(m) * fact(m) * (2l+1)/(4pi) * Re[Y_l^m(mu1, phi)] * weights``,
        flattened to 1D so the multipole becomes ``proj_op . B_5d_flat``.
        """
        ell_key = tuple(sorted(set(tuple(int(x) for x in ll) for ll in ell)))
        key = (nmu, nphi, ell_key)
        cached = self._scocc_proj_cache.get(key)
        if cached is not None:
            return cached

        mu_g, w_mu_g, _, _, phi_g, w_phi_g = \
            self._scocc_get_quadrature(nmu, nphi)
        weights = np.broadcast_to(w_mu_g * w_phi_g,
                                  (1, nmu, nphi)).reshape(nmu, nphi)

        proj = {}
        for ll in ell_key:
            l, m = ll
            m_abs = abs(m)
            sign = (-1.0)**m if m < 0 else 1.0
            fact = 1.0/np.sqrt(2.0) if m != 0 else 1.0
            ylm = self._sph_harm_real(l, m_abs, mu_g, phi_g)
            ylm_b = np.broadcast_to(ylm, (1, nmu, nphi)).reshape(nmu, nphi)
            op = ylm_b * weights
            prefactor = sign * fact 
            proj[ll] = (op * prefactor).ravel()
        self._scocc_proj_cache[key] = proj
        return proj

    def _scocc_moments(self, tri, Pdw_eval, params, ell, nmu, nphi,
                       X_list=None):
        """Scoccimarro ``(l, m)`` projections of the bispectrum (or of its
        diagrams) for all triangles, before the normalisation and NB0, see
        `_fused_moments`. Each triangle is a pair with a single ``mu12``
        node, and the operators of `_scocc_get_proj_ops` are the angular
        weights. The moment axis follows **ell**."""
        tri = np.atleast_2d(tri)
        mu_g, _, cphi_g, _, _, _ = self._scocc_get_quadrature(nmu, nphi)
        proj = self._scocc_get_proj_ops(ell, nmu, nphi)
        ang = np.stack([proj[ll].reshape(nmu, nphi) for ll in ell])
        ang_id = np.arange(len(ell), dtype=np.int64)
        rad = np.ones((tri.shape[0], len(ell), 1))
        k1 = tri[:, 0:1]
        k2 = tri[:, 1:2]
        k3 = tri[:, 2:3]
        mu12 = np.clip((k3*k3 - k1*k1 - k2*k2) / (2.0*k1*k2), -1.0, 1.0)
        return self._fused_moments(tri[:, 0], tri[:, 1], k3, mu12,
                                   mu_g.ravel(), cphi_g.ravel(), ang, ang_id,
                                   rad, Pdw_eval, params, X_list=X_list)

    # Fused moments, shared by both bases

    def _get_damping_arrays(self, params):
        """Return ``(avir, sv)`` as contiguous per-z arrays.
        Set to zero for 'EFT', which makes ``W_infty`` = 1.
        """
        nparams = np.atleast_1d(params['q_lo']).size
        if 'VDG_infty' not in self.model:
            zeros = np.zeros(nparams, dtype=float)
            return zeros, zeros
        def _get(k):
            if k in params:
                return np.ascontiguousarray(np.atleast_1d(params[k]))
            return np.zeros(nparams, dtype=float)
        return _get('avirB'), _get('sv')

    def _get_ctr_arrays(self, params):
        """Return the per-z cnloB, cB1, cB2 arrays.
        Set to zero for pure VDG.
        """
        nparams = np.atleast_1d(params['q_lo']).size
        if not ('EFT' in self.model or 'VDG_infty_ctr' in self.model):
            return np.zeros(nparams, dtype=float), np.zeros(nparams, dtype=float), np.zeros(nparams, dtype=float)
        elif self.cnlo_type == 'EggLeeSco':
            return np.ascontiguousarray(
                np.atleast_1d(params['cnloB'])), np.zeros(nparams, dtype=float), np.zeros(nparams, dtype=float)
        elif self.cnlo_type == 'IvaPhiNis':
            return np.zeros(nparams, dtype=float), np.ascontiguousarray(
                np.atleast_1d(params['cB1'])), np.ascontiguousarray(
                np.atleast_1d(params['cB2']))

    def _jax_kernel_flags(self):
        """Static model switches of the JAX kernels, ``(vdg, ctr)``: whether
        the VDG_infty damping is on, and the counterterm type (None,
        'EggLeeSco' or 'IvaPhiNis'), as `_get_damping_arrays` and
        `_get_ctr_arrays` select them."""
        return ('VDG_infty' in self.model,
                self.cnlo_type if self._has_ctr() else None)

    @staticmethod
    def _jax_spline_table(tab):
        """Spline table of the JAX kernels (see `_ppoly_eval_jax`) from the
        tuple ``(G, scale, seg, last_index, V, log_lo, inv_dlog, coeffs,
        tails)`` of `Splines.ppoly_table` (or `PTEmu._jax_Pdw_ppoly_table`),
        whose breakpoints ``G`` are static: ``(scale, coeffs, tails, seg,
        iseg, gseg, V, lo, inv_dk, log_g0, log_g1)``, with the linear
        buckets of `comet.splines._linear_bucket_table`."""
        G = np.asarray(tab[0], dtype=float)
        seg, iseg, gseg, V, lo, inv_dk = _linear_bucket_table(G)
        return (tab[1], tab[7], tab[8], seg, iseg, gseg, V, lo, inv_dk,
                float(np.log(G[0])), float(np.log(G[-1])))

    @staticmethod
    def _merge_phi_nodes(cphi, ang):
        """Merge the ``phi`` nodes with the same ``cos(phi)``.

        The bispectrum only depends on ``phi`` through ``cos(phi)``, so the
        nodes ``phi`` and ``2 pi - phi`` of the uniform ``phi`` grids give the
        same value: they are evaluated once, with their angular weights
        ``ang`` (nang, nmu1, nphi) summed. This removes a third of the nodes
        for ``nphi = 3`` and two fifths for ``nphi = 5``.
        """
        cphi = np.asarray(cphi, dtype=float)
        _, first, inv = np.unique(np.round(cphi, 12), return_index=True,
                                  return_inverse=True)
        ang = np.asarray(ang, dtype=float)
        if first.size < cphi.size:
            merge = np.zeros((cphi.size, first.size))
            merge[np.arange(cphi.size), inv.ravel()] = 1.0
            ang = ang @ merge
            cphi = cphi[first]
        return (np.ascontiguousarray(cphi), np.ascontiguousarray(ang))

    def _fused_moments(self, k1, k2, k3, mu12, mu1, cphi, ang, ang_id, rad,
                       Pdw_eval, params, X_list=None):
        """Moments ``sum_{j,a,b} B ang[ang_id[q], a, b] rad[i, q, j]`` of the
        bispectrum (or of its diagrams) over the ``(mu12, mu1, phi)`` nodes
        ``(j, a, b)`` of each pair ``i``, with the fused kernels
        (`_fused_moments_njit` or its JAX counterpart), which evaluate the
        exact Pdw spline of ``Pdw_eval.ppoly_table()`` inline.

        Returns ``(moments, qiso6)``, with ``moments`` of shape
        (n_pair, nq, nparams) if **X_list** is **None**, and otherwise a tuple
        ``(tree, stoch, tree_keep, stoch_keep)`` with the moments of the
        requested tree and stochastic diagrams, (n_pair, nq, nparams, n).
        """
        n_pair = k1.size
        k1 = np.ascontiguousarray(k1, dtype=float)
        k2 = np.ascontiguousarray(k2, dtype=float)
        k3 = np.ascontiguousarray(k3, dtype=float)
        mu12 = np.ascontiguousarray(mu12, dtype=float)
        mu1 = np.ascontiguousarray(mu1, dtype=float)
        cphi, ang = self._merge_phi_nodes(cphi, ang)
        rad = np.ascontiguousarray(rad, dtype=float)

        qlo = np.atleast_1d(params['q_lo'])
        qtr = np.atleast_1d(params['q_tr'])
        qiso6 = qlo**2 * qtr**4
        nparams = qiso6.size
        inv_qtr = np.ascontiguousarray(1.0 / qtr, dtype=float)
        inv_F = np.ascontiguousarray(qtr / qlo, dtype=float)
        c = np.ascontiguousarray(inv_F*inv_F - 1.0)
        t = Pdw_eval.ppoly_table()
        tab = (t['G'], t['scale'], t['seg'], t['last_index'], t['V'],
               t['log_lo'], t['inv_dlog'], t['coeffs'], t['tails'])
        f = np.ascontiguousarray(np.atleast_1d(params['f']))
        avir, sv = self._get_damping_arrays(params)
        inv_qiso6 = np.ascontiguousarray(1.0 / qiso6)
        geom = (k1, k2, k3, mu12, mu1, cphi, inv_qtr, inv_F, c)
        nq = ang_id.size

        if X_list is None:
            b1 = np.ascontiguousarray(np.atleast_1d(params['b1']))
            b2 = np.ascontiguousarray(np.atleast_1d(params['b2']))
            g2 = np.ascontiguousarray(np.atleast_1d(params['g2']))
            MB0 = np.ascontiguousarray(np.atleast_1d(params['MB0']))
            NP0 = np.ascontiguousarray(np.atleast_1d(params['NP0']))
            cnloB, cB1, cB2 = self._get_ctr_arrays(params)
            nb_ = np.atleast_1d(np.asarray(self.nbar, dtype=float)).reshape(-1)
            inv_nbar = np.ascontiguousarray(
                1.0 / np.broadcast_to(nb_, inv_qiso6.shape))
            bias = (b1, b2, g2, f, avir, sv, MB0, NP0, cnloB, cB1, cB2,
                    inv_nbar, inv_qiso6)
            if self.backend == 'jax':
                stab = self._jax_spline_table(tab)
                vdg, ctr = self._jax_kernel_flags()
                out = self._jax_blocks(
                    lambda g, r: self._fused_moments_jax(
                        *g, stab, *bias, jnp.asarray(ang[ang_id]), r,
                        vdg=vdg, ctr=ctr),
                    geom, rad)
            else:
                out = np.empty((n_pair, nq, nparams))
                self._fused_moments_njit(*geom, *tab, *bias, ang, ang_id, rad,
                                        out)
            return out, qiso6

        tree_set = set(self.tree_diagrams)
        stoch_set = set(self.stoch_diagrams[:-1])
        tree_keep = tuple(sorted({self.tree_index[x] for x in X_list
                                  if x in tree_set}))
        stoch_keep = tuple(sorted({self.stoch_index[x] for x in X_list
                                   if x in stoch_set}))
        tree_spec, stoch_spec, tables = self._diagram_kernel_tables(
            tree_keep, stoch_keep)
        if self.backend == 'jax':
            ang_q = jnp.asarray(ang[ang_id])
            stab = self._jax_spline_table(tab)
            vdg = self._jax_kernel_flags()[0]
            out_t, out_s = self._jax_blocks(
                lambda g, r: self._fused_moments_diagrams_jax(
                    *g, stab, f, avir, sv, inv_qiso6, ang_q, r, tree_spec,
                    stoch_spec, vdg=vdg), geom, rad, n_out=2)
        else:
            out_t = np.empty((n_pair, nq, nparams, len(tree_keep)))
            out_s = np.empty((n_pair, nq, nparams, len(stoch_keep)))
            self._fused_moments_diagrams_njit(
                *geom, *tab, f, avir, sv, inv_qiso6, *tables, ang, ang_id, rad,
                out_t, out_s)
        return (out_t, out_s, tree_keep, stoch_keep), qiso6

    # Pairs per jitted block in the JAX path (blocks are padded to this size,
    # so the kernels are compiled once per shape)
    _JAX_PAIR_BLOCK = 1024

    def _jax_blocks(self, fn, geom, rad, n_out=1):
        """Apply the jitted block kernel ``fn(geom_block, rad_block)`` over
        blocks of `_JAX_PAIR_BLOCK` pairs or triangles (the last one padded
        by repeating its last pair, so the kernel is compiled once) and
        concatenate."""
        k1, k2, k3, mu12 = geom[:4]
        n_pair = k1.size
        blk = min(self._JAX_PAIR_BLOCK, n_pair)
        outs = [[] for _ in range(n_out)]
        for i0 in range(0, n_pair, blk):
            idx = np.minimum(np.arange(i0, i0 + blk), n_pair - 1)
            g = tuple(jnp.asarray(a[idx]) for a in (k1, k2, k3, mu12)) \
                + tuple(jnp.asarray(a) for a in geom[4:])
            res = fn(g, jnp.asarray(rad[idx]))
            res = (res,) if n_out == 1 else res
            nvalid = min(blk, n_pair - i0)
            for o, r in zip(outs, res):
                o.append(np.asarray(r)[:nvalid])
        outs = [np.concatenate(o, axis=0) for o in outs]
        return outs[0] if n_out == 1 else tuple(outs)

    # Numba kernels (JAX counterparts below)

    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _fused_moments_njit(k1, k2, k3, mu12, mu1, cphi, inv_qtr, inv_F, c,
                            G, scale, seg, last_index, V, log_lo, inv_dlog,
                            coeffs, tails, b1, b2, g2, f, avir, sv, MB0, NP0,
                            cnloB, cB1, cB2, inv_nbar, inv_qiso6,
                            ang, ang_id, rad, out):
        """Projected moments of the bispectrum, fusing geometry, AP, Pdw and
        projection per (k1, k2) pair, without building the 5D grid. Used by
        both bases (see `_fused_moments`): Sugiyama integrates over the
        ``mu12`` nodes, while Scoccimarro has a single node per triangle.

        For each pair ``i``, moment ``q`` and parameter set ``m``,
        ``out[i, q, m] = sum_{j,a,b} B(i, a, j, b, m) ang[ang_id[q], a, b]
        rad[i, q, j]``, with ``j``, ``a``, ``b`` the ``mu12``, ``mu1`` and
        ``phi`` nodes, ``B`` from `_bisp_node_total` and Pdw from
        `_ppoly_node`. See `_sugi_factorized_weights` and `_scocc_moments`
        for ``ang``/``rad``. Its JAX counterpart is `_fused_moments_jax`.

        Inputs: ``k1``, ``k2`` (n_pair,); ``k3``, ``mu12`` (n_pair, nmu12);
        ``mu1`` (nmu1,); ``cphi`` (nphi,); the AP scalars and the bias
        parameters (nparams,); the spline table of `Splines.ppoly_table`.
        Output: ``out`` (n_pair, nq, nparams).
        """
        n_pair = k1.size
        nmu12 = k3.shape[1]
        nmu1 = mu1.size
        nphi = cphi.size
        nq = ang_id.size
        nparams = b1.size
        for i in nb.prange(n_pair):
            k1_i = k1[i]
            k2_i = k2[i]
            k1p = np.empty(nmu1)
            mu1p = np.empty(nmu1)
            pd1 = np.empty(nmu1)
            acc = np.empty(nq)
            # the parameter sets are the outer loop, so that the spline
            # segments of legs 2 and 3 at the previous node (the guesses for
            # the next one, see _ppoly_segment) are scalars
            for m in range(nparams):
                # leg 1 only depends on mu1
                for a in range(nmu1):
                    fac1 = np.sqrt(1.0 + mu1[a]*mu1[a]*c[m])
                    k1p[a] = k1_i*inv_qtr[m]*fac1
                    mu1p[a] = mu1[a]*inv_F[m]/fac1
                    pd1[a] = _ppoly_node(k1p[a], mu1p[a]**2, m, -1, G, scale,
                                         seg, last_index, V, log_lo,
                                         inv_dlog, coeffs, tails)[0]
                j2 = -1
                j3 = -1
                for q in range(nq):
                    acc[q] = 0.0
                for j in range(nmu12):
                    k3_ij = k3[i, j]
                    mu12_ij = mu12[i, j]
                    s12 = 1.0 - mu12_ij*mu12_ij
                    sin12 = np.sqrt(s12) if s12 > 0.0 else 0.0
                    inv_k3 = 1.0/k3_ij
                    for a in range(nmu1):
                        mu1_a = mu1[a]
                        s1 = 1.0 - mu1_a*mu1_a
                        sin1 = np.sqrt(s1) if s1 > 0.0 else 0.0
                        for b in range(nphi):
                            mu2 = mu12_ij*mu1_a + sin12*sin1*cphi[b]
                            mu3 = -(k1_i*mu1_a + k2_i*mu2)*inv_k3
                            fac2 = np.sqrt(1.0 + mu2*mu2*c[m])
                            fac3 = np.sqrt(1.0 + mu3*mu3*c[m])
                            k2p = k2_i*inv_qtr[m]*fac2
                            k3p = k3_ij*inv_qtr[m]*fac3
                            mu2p = mu2*inv_F[m]*(1.0/fac2)
                            mu3p = mu3*inv_F[m]*(1.0/fac3)
                            pd2, j2 = _ppoly_node(
                                k2p, mu2p*mu2p, m, j2, G, scale, seg,
                                last_index, V, log_lo, inv_dlog, coeffs,
                                tails)
                            pd3, j3 = _ppoly_node(
                                k3p, mu3p*mu3p, m, j3, G, scale, seg,
                                last_index, V, log_lo, inv_dlog, coeffs,
                                tails)
                            Bv = _bisp_node_total(
                                k1p[a], k2p, k3p, mu1p[a], mu2p, mu3p,
                                pd1[a], pd2, pd3, b1[m], b2[m], g2[m],
                                f[m], avir[m], sv[m], MB0[m], NP0[m],
                                cnloB[m], cB1[m], cB2[m], inv_nbar[m],
                                inv_qiso6[m])
                            for q in range(nq):
                                acc[q] += Bv*ang[ang_id[q], a, b] \
                                    * rad[i, q, j]
                for q in range(nq):
                    out[i, q, m] = acc[q]

    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _fused_moments_diagrams_njit(k1, k2, k3, mu12, mu1, cphi, inv_qtr,
                                     inv_F, c, G, scale, seg, last_index, V,
                                     log_lo, inv_dlog, coeffs, tails, f, avir,
                                     sv, inv_qiso6, tree_col, tree_a, tree_b,
                                     tree_ctr, stoch_part, stoch_ctr,
                                     ang, ang_id, rad, out_spt, out_stoch):
        """Diagram-resolved companion to `_fused_moments_njit`, with the
        per-node diagrams of `_bisp_node_diagrams`. Outputs ``out_spt``
        (n_pair, nq, nparams, n_tree) and ``out_stoch`` (n_pair, nq, nparams,
        n_stoch). Its JAX counterpart is `_fused_moments_diagrams_jax`."""
        n_pair = k1.size
        nmu12 = k3.shape[1]
        nmu1 = mu1.size
        nphi = cphi.size
        nq = ang_id.size
        nparams = f.size
        n_tree = tree_ctr.size
        n_stoch = stoch_part.size
        need_ctr = False
        for t in range(tree_col.size):
            if tree_a[t] >= 3:
                need_ctr = True
        for i in nb.prange(n_pair):
            k1_i = k1[i]
            k2_i = k2[i]
            A = np.zeros(10)
            Z = np.empty(4)
            S = np.empty(7)
            spt = np.zeros(max(n_tree, 1))
            st = np.zeros(max(n_stoch, 1))
            acc_t = np.empty((nq, max(n_tree, 1)))
            acc_s = np.empty((nq, max(n_stoch, 1)))
            k1p = np.empty(nmu1)
            mu1p = np.empty(nmu1)
            pd1 = np.empty(nmu1)
            # parameter sets as the outer loop, see _fused_moments_njit
            for m in range(nparams):
                for a in range(nmu1):
                    fac1 = np.sqrt(1.0 + mu1[a]*mu1[a]*c[m])
                    k1p[a] = k1_i*inv_qtr[m]*fac1
                    mu1p[a] = mu1[a]*inv_F[m]/fac1
                    pd1[a] = _ppoly_node(k1p[a], mu1p[a]**2, m, -1, G, scale,
                                         seg, last_index, V, log_lo,
                                         inv_dlog, coeffs, tails)[0]
                j2 = -1
                j3 = -1
                acc_t[:] = 0.0
                acc_s[:] = 0.0
                for j in range(nmu12):
                    k3_ij = k3[i, j]
                    mu12_ij = mu12[i, j]
                    s12 = 1.0 - mu12_ij*mu12_ij
                    sin12 = np.sqrt(s12) if s12 > 0.0 else 0.0
                    inv_k3 = 1.0/k3_ij
                    for a in range(nmu1):
                        mu1_a = mu1[a]
                        s1 = 1.0 - mu1_a*mu1_a
                        sin1 = np.sqrt(s1) if s1 > 0.0 else 0.0
                        for b in range(nphi):
                            mu2 = mu12_ij*mu1_a + sin12*sin1*cphi[b]
                            mu3 = -(k1_i*mu1_a + k2_i*mu2)*inv_k3
                            fac2 = np.sqrt(1.0 + mu2*mu2*c[m])
                            fac3 = np.sqrt(1.0 + mu3*mu3*c[m])
                            k2p = k2_i*inv_qtr[m]*fac2
                            k3p = k3_ij*inv_qtr[m]*fac3
                            mu2p = mu2*inv_F[m]*(1.0/fac2)
                            mu3p = mu3*inv_F[m]*(1.0/fac3)
                            pd2, j2 = _ppoly_node(
                                k2p, mu2p*mu2p, m, j2, G, scale, seg,
                                last_index, V, log_lo, inv_dlog, coeffs,
                                tails)
                            pd3, j3 = _ppoly_node(
                                k3p, mu3p*mu3p, m, j3, G, scale, seg,
                                last_index, V, log_lo, inv_dlog, coeffs,
                                tails)
                            _bisp_node_diagrams(
                                k1p[a], k2p, k3p, mu1p[a], mu2p, mu3p,
                                pd1[a], pd2, pd3, f[m], avir[m], sv[m],
                                inv_qiso6[m], tree_col, tree_a, tree_b,
                                tree_ctr, stoch_part, stoch_ctr, need_ctr,
                                A, Z, S, spt, st)
                            for q in range(nq):
                                w = ang[ang_id[q], a, b]*rad[i, q, j]
                                for d in range(n_tree):
                                    acc_t[q, d] += w*spt[d]
                                for d in range(n_stoch):
                                    acc_s[q, d] += w*st[d]
                for q in range(nq):
                    for d in range(n_tree):
                        out_spt[i, q, m, d] = acc_t[q, d]
                    for d in range(n_stoch):
                        out_stoch[i, q, m, d] = acc_s[q, d]

    # JAX kernels

    if HAS_JAX:
        # Per-node pieces of `_fused_moments_jax`. Each leg is a dict of the
        # quantities shared by the tree and stochastic terms (`_leg_jax`),
        # so that leg 1, which only depends on (pair, mu1), is computed once
        # on that grid. ``vdg`` and ``ctr`` select the model terms
        # statically (see `_jax_kernel_flags`): the terms of absent
        # parameters are not evaluated, instead of being multiplied by zero.

        @staticmethod
        def _leg_jax(kp, mup, pd, inv_kp, b1, f, cB1, cB2, ctr):
            """AP-corrected wavemode, cosine, Pdw and 1/k of one leg, and the
            products used by `_tree_leg_jax` and `_stoch_leg_jax`. Z1
            includes the IvaPhiNis counterterms."""
            mu_sq = mup*mup
            ksq = kp*kp
            u = ksq*mu_sq
            Z1 = b1 + f*mu_sq
            if ctr == 'IvaPhiNis':
                Z1 = Z1 - cB1*u - cB2*u*mu_sq
            return dict(mu_sq=mu_sq, ksq=ksq, u=u, Z1=Z1, pd=pd,
                        a=mup*inv_kp, km=kp*mup, ik=inv_kp)

        @staticmethod
        def _tree_leg_jax(Li, Lj, Lk, b1, b2, g2, f):
            """``2 Z1_i Z1_j Z2(k_i, k_j) P_i P_j``, with the angle between
            ``k_i`` and ``k_j`` from the three (AP-corrected) sides. Z2 is
            written with the bare ``Z1 = b1 + f mu^2`` in its velocity term,
            and ``a = mu/k``."""
            w = Li['ik']*Lj['ik']
            muij = 0.5*(Lk['ksq'] - Li['ksq'] - Lj['ksq'])*w
            muij2 = muij*muij
            ratio = 0.5*(Li['ksq'] + Lj['ksq'])*w*muij
            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
            K2 = b1*F2 + 0.5*b2 + g2*(muij2 - 1.0)
            Z2 = (K2 + f*Lk['mu_sq']*G2
                  - 0.5*f*Lk['km']*(b1*(Li['a'] + Lj['a'])
                                    + f*(Lj['mu_sq']*Li['a']
                                         + Li['mu_sq']*Lj['a'])))
            return 2.0*Li['Z1']*Lj['Z1']*Z2*(Li['pd']*Lj['pd'])

        @staticmethod
        def _stoch_leg_jax(L, b1, f, avir, sv, MB0, NP0, cnloB, vdg, ctr):
            """Stochastic term of one leg, without 1/nbar:
            ``W_i (b1 MB0 + f NP0 mu_i^2) Z1_i P_i``, times the EggLeeSco
            counterterm."""
            t = (b1*MB0 + f*NP0*L['mu_sq'])*L['Z1']*L['pd']
            if ctr == 'EggLeeSco':
                t = t*(1.0 + cnloB*f*f*L['u'])
            if vdg:
                lam = -f*f*L['u']
                rs = jax.lax.rsqrt(1.0 - lam*avir*avir)
                t = t*jnp.exp(lam*sv*sv*rs*rs)*rs
            return t

        @staticmethod
        def _pair_pieces_jax(Li, Lj, Lk, f, a_need, b_need):
            """Pieces ``A[a]`` and ``Z[b] P_i P_j`` of the tree diagrams for
            the leg pair ``(i, j; k)`` (see `_BASE_TREE_TERMS`), for the
            ``a`` and ``b`` in **a_need** and **b_need**, as in
            `_bisp_node_diagrams`."""
            w = Li['ik']*Lj['ik']
            muij = 0.5*(Lk['ksq'] - Li['ksq'] - Lj['ksq'])*w
            muij2 = muij*muij
            ratio = 0.5*(Li['ksq'] + Lj['ksq'])*w*muij
            pp = Li['pd']*Lj['pd']
            fkm = f*Lk['km']
            ai = fkm*Li['a']
            aj = fkm*Lj['a']
            Z = {}
            if 0 in b_need:
                Z[0] = (5.0/7.0 + 2.0/7.0*muij2 + ratio - 0.5*(ai + aj))*pp
            if 1 in b_need:
                Z[1] = 0.5*pp
            if 2 in b_need:
                Z[2] = (muij2 - 1.0)*pp
            if 3 in b_need:
                G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
                Z[3] = (f*Lk['mu_sq']*G2
                        - 0.5*f*(ai*Lj['mu_sq'] + aj*Li['mu_sq']))*pp
            mi, mj = Li['mu_sq'], Lj['mu_sq']
            A = {0: 1.0, 1: f*(mi + mj), 2: f*f*mi*mj}
            if max(a_need) >= 3:
                ui, uj = Li['u'], Lj['u']
                vi, vj = ui*mi, uj*mj
                A.update({3: -(ui + uj), 4: -f*(mi*uj + mj*ui),
                          5: -(vi + vj), 6: -f*(mi*vj + mj*vi), 7: ui*uj,
                          8: ui*vj + vi*uj, 9: vi*vj})
            return A, Z

        @staticmethod
        def _stoch_part_jax(L, f, s):
            """Piece ``s`` of the stochastic diagrams of one leg (see
            `_BASE_STOCH_PARTS`)."""
            m, u = L['mu_sq'], L['u']
            return (1.0, f*m, f*f*m*m, -u, -f*m*u, -u*m, -f*m*u*m)[s]

        @staticmethod
        def _keep_jax(x, unit):
            """``x``, stored once by XLA instead of being recomputed by each of
            its consumers. On CPU, `jax.lax.optimization_barrier` does not
            stop XLA from fusing cheap producers into every consumer: a
            value used by many diagram columns is then recomputed in each.
            XLA does not duplicate a division, so dividing by ``unit``, a
            traced 1.0 (one XLA cannot constant-fold), keeps ``x`` (exactly)
            in memory."""
            return x/unit

        @staticmethod
        def _ppoly_eval_jax(k, mu2, stab):
            """JAX counterpart of `_ppoly_node`, for ``k`` and ``mu2`` with
            the parameter batch along their last axis and the spline table
            ``stab`` of `_jax_spline_table`.

            Arranged for XLA rather than as a copy of the numba code (same
            values, to rounding): the segment comes from the linear buckets
            of `comet.splines._linear_bucket_table`, so the lookup (which
            XLA evaluates in a pass of its own) needs no log, and the
            coefficients are gathered from the flattened table at a computed
            index (XLA handles these 1-D gathers much better than indexing
            the 3-D table). The log of ``k`` is only taken for the power-law
            tails, which share one exp."""
            scale, coeffs, tails, seg, iseg, gseg, V, lo, inv_dk, log_g0, \
                log_g1 = stab
            nv, nbucket = V.shape[0], seg.shape[0]
            nmu2, npar = coeffs.shape[2], coeffs.shape[3]
            x = k/scale                              # k in units of G
            q = jnp.clip(x, V[0], V[-1])
            b = jnp.clip(((q - lo)*inv_dk).astype(jnp.int32), 0, nbucket - 1)
            j = seg[b]
            j = jnp.minimum(j + (q >= V[jnp.minimum(j + 1, nv - 1)]), nv - 1)
            dx = (q - gseg[j])*scale
            cf = coeffs.reshape(4, -1)
            base = iseg[j]*(nmu2*npar) + jnp.arange(npar, dtype=jnp.int32)
            val = 0.0
            for n in range(nmu2 - 1, -1, -1):
                ix = base + n*npar
                val = val*mu2 + ((cf[0][ix]*dx + cf[1][ix])*dx
                                 + cf[2][ix])*dx + cf[3][ix]
            # power-law tails, y_edge (k/k_edge)^neff
            lraw = jnp.log(x)
            below = lraw < log_g0
            outside = below | (lraw > log_g1)
            amp = jnp.where(below, tails[0], tails[2])
            neff = jnp.where(below, tails[1], tails[3])
            edge = jnp.where(below, log_g0, log_g1)
            return jnp.where(outside, amp*jnp.exp(neff*(lraw - edge)), val)

        @staticmethod
        def _legs_jax(k1, k2, k3, mu12, mu1, cphi, inv_qtr, inv_F, c, stab,
                      b1, f, cB1, cB2, ctr, keep=(0, 1, 2), mu12_major=False):
            """The three legs (see `_leg_jax`) of the AP-corrected nodes:
            leg 1 on the (pair, mu1) grid, legs 2 and 3 on the (pair, mu1,
            mu12, phi, params) grid, or (pair, mu12, mu1, phi, params) if
            **mu12_major**. The Pdw of the legs in **keep** (0 for leg 1) is
            stored once instead of being recomputed by each of its consumers
            (see `_keep_jax`)."""
            BN = BispectrumNum
            qtr = 1.0/inv_qtr
            k1_ = k1[:, None, None, None, None]
            k2_ = k2[:, None, None, None, None]
            cphi_ = cphi[None, None, None, :, None]
            if mu12_major:
                k3_ = k3[:, :, None, None, None]
                mu1_ = mu1[None, None, :, None, None]
                mu12_ = mu12[:, :, None, None, None]
            else:
                k3_ = k3[:, None, :, None, None]
                mu1_ = mu1[None, :, None, None, None]
                mu12_ = mu12[:, None, :, None, None]
            sin1 = jnp.sqrt(jnp.clip(1.0 - mu1_*mu1_, 0.0))
            sin12 = jnp.sqrt(jnp.clip(1.0 - mu12_*mu12_, 0.0))
            mu2 = mu12_*mu1_ + sin12*sin1*cphi_
            mu3 = -(k1_*mu1_ + k2_*mu2)/k3_

            # traced, so that XLA cannot fold the division of _keep_jax away
            # (0*f is not simplified, because of nan and inf)
            unit = 1.0 + 0.0*f

            def leg(i, k_, mu_):
                r = jax.lax.rsqrt(1.0 + mu_*mu_*c)
                kp, mup = k_*inv_qtr/r, mu_*inv_F*r
                pd = BN._ppoly_eval_jax(kp, mup*mup, stab)
                if i in keep:
                    pd = BN._keep_jax(pd, unit)
                return BN._leg_jax(kp, mup, pd, r*qtr/k_, b1, f, cB1, cB2,
                                   ctr)
            return leg(0, k1_, mu1_), leg(1, k2_, mu2), leg(2, k3_, mu3)

        @staticmethod
        def _winfty_jax(U, f, avir, sv):
            """VDG_infty damping ``exp(lamb2 sv^2/den)/den^1.5``, with
            ``lamb2 = -f^2 U/2`` and ``den = 1 - lamb2 avir^2``."""
            lamb2 = -0.5*f*f*U
            rs = jax.lax.rsqrt(1.0 - lamb2*avir*avir)
            return jnp.exp(lamb2*sv*sv*rs*rs)*rs*rs*rs

        @staticmethod
        def _fused_moments_jax(k1, k2, k3, mu12, mu1, cphi, inv_qtr, inv_F, c,
                               stab, b1, b2, g2, f, avir, sv, MB0, NP0, cnloB,
                               cB1, cB2, inv_nbar, inv_qiso6, ang_q, rad,
                               damp=None, vdg=True, ctr=None):
            """JAX counterpart of `_fused_moments_njit` for a block of pairs,
            with ``ang_q = ang[ang_id]`` and the spline table ``stab`` of
            `_jax_spline_table`. Returns (n_pair, nq, nparams).

            Same values as the numba kernel to rounding, arranged for XLA:
            leg 1 (which only depends on the pair and mu1) is evaluated on
            the (pair, mu1) grid, with its Pdw stored once, the model terms are selected by the static
            ``vdg`` (VDG_infty damping) and ``ctr`` (None, 'EggLeeSco' or
            'IvaPhiNis') of `_jax_kernel_flags`, and reciprocals replace
            most divisions. The rest of the node computation (geometry, AP,
            Pdw of legs 2 and 3, bispectrum) is left to XLA as a single
            fusion of small inputs, which XLA spreads over all threads (a
            fusion that reads large precomputed node arrays is costed as
            memory-bound and gets fewer).

            ``damp``, if given, multiplies the bispectrum at every node before
            the projection; it broadcasts against the (pair, mu1, mu12, phi,
            params) grid (see `PTEmu._jax_bell_sugi`)."""
            BN = BispectrumNum
            L1, L2, L3 = BN._legs_jax(k1, k2, k3, mu12, mu1, cphi, inv_qtr,
                                      inv_F, c, stab, b1, f, cB1, cB2, ctr,
                                      keep=(0,))
            tree = (BN._tree_leg_jax(L1, L2, L3, b1, b2, g2, f)
                    + BN._tree_leg_jax(L2, L3, L1, b1, b2, g2, f)
                    + BN._tree_leg_jax(L3, L1, L2, b1, b2, g2, f))
            U = L1['u'] + L2['u'] + L3['u']
            if ctr == 'EggLeeSco':
                tree = tree*(1.0 + cnloB*f*f*U)
            if vdg:
                tree = tree*BN._winfty_jax(U, f, avir, sv)
            stoch = sum(BN._stoch_leg_jax(L, b1, f, avir, sv, MB0, NP0, cnloB,
                                          vdg, ctr) for L in (L1, L2, L3))
            B = (tree + stoch*inv_nbar)*inv_qiso6
            if damp is not None:
                B = B*damp
            return jnp.einsum('pajbm,qab,pqj->pqm', B, ang_q, rad)

        _fused_moments_jax = staticmethod(jax.jit(
            _fused_moments_jax.__func__, static_argnames=('vdg', 'ctr')))

        @staticmethod
        def _fused_moments_diagrams_jax(k1, k2, k3, mu12, mu1, cphi, inv_qtr,
                                        inv_F, c, stab, f, avir, sv, inv_qiso6,
                                        ang_q, rad, tree_spec, stoch_spec,
                                        damp=None, vdg=True):
            """JAX counterpart of `_fused_moments_diagrams_njit` for a block of
            pairs, with the spline table ``stab`` of `_jax_spline_table`.
            Returns (n_pair, nq, nparams, n_tree) and
            (n_pair, nq, nparams, n_stoch). ``tree_spec`` is a static tuple
            of ``(((a, b), ...), ctr)`` per requested tree diagram and
            ``stoch_spec`` a static tuple of ``(s, ctr)`` per requested
            stochastic diagram (see `_diagram_kernel_tables`); ``vdg`` is as
            in `_fused_moments_jax`, and ``damp`` too.

            Each diagram is a column of cheap arithmetic on per-node
            quantities shared by all columns: the legs (with their Pdw kept
            in memory, see `_keep_jax`), the damping and the stochastic
            weights. XLA computes the columns in separate passes, so
            anything expensive that is not stored is paid once per
            diagram. The node grid is (pair, mu12, mu1, phi, params), so
            that each column is projected by a plain matmul (see below)."""
            BN = BispectrumNum
            zero = jnp.zeros_like(f)
            legs = BN._legs_jax(k1, k2, k3, mu12, mu1, cphi, inv_qtr, inv_F, c,
                                stab, zero, f, zero, zero, None,
                                mu12_major=True)
            U = legs[0]['u'] + legs[1]['u'] + legs[2]['u']
            shape = U.shape          # (pair, mu12, mu1, phi, params)
            cols = []
            if tree_spec:
                winfac = (BN._winfty_jax(U, f, avir, sv)*inv_qiso6 if vdg
                          else inv_qiso6)
                a_need = {a for terms, _ in tree_spec for a, _ in terms}
                b_need = {b for terms, _ in tree_spec for _, b in terms}
                L1, L2, L3 = legs
                pairs = (BN._pair_pieces_jax(L1, L2, L3, f, a_need, b_need),
                         BN._pair_pieces_jax(L2, L3, L1, f, a_need, b_need),
                         BN._pair_pieces_jax(L3, L1, L2, f, a_need, b_need))
                for terms, ctr in tree_spec:
                    d = sum(2.0*A[a]*Z[b] for A, Z in pairs for a, b in terms)
                    d = d*winfac
                    if ctr:
                        d = d*(f*f*U)
                    cols.append(jnp.broadcast_to(d, shape))
            if stoch_spec:
                wp = []
                for L in legs:
                    w = L['pd']*inv_qiso6
                    if vdg:
                        lam = -f*f*L['u']
                        rs = jax.lax.rsqrt(1.0 - lam*avir*avir)
                        w = w*jnp.exp(lam*sv*sv*rs*rs)*rs
                    wp.append(w)
                cols += [jnp.broadcast_to(sum(
                    BN._stoch_part_jax(L, f, s)*w*(f*f*L['u'] if ctr else 1.0)
                    for L, w in zip(legs, wp)), shape)
                    for s, ctr in stoch_spec]
            n_pair, nq = rad.shape[0], rad.shape[1]
            nmu12, nmu1, nphi, npar = shape[1:]
            if not cols:
                empty = jnp.zeros((n_pair, nq, npar, 0))
                return empty, empty
            if damp is not None:
                # damp is on the (pair, mu1, mu12, phi, params) grid
                damp = jnp.swapaxes(jnp.broadcast_to(
                    damp, (n_pair, nmu1, nmu12, nphi, damp.shape[-1])), 1, 2)
                cols = [d*damp for d in cols]
            # Projection of each diagram: on this grid the (mu1, phi) nodes
            # contracted with ang are contiguous, so that is a plain matmul
            # (multithreaded, and without moving the diagram, unlike an
            # einsum over the (pair, mu1, mu12, phi) grid, for which XLA
            # first transposes all the diagrams), followed by the small
            # contraction with rad over mu12
            angT = ang_q.reshape(nq, nmu1*nphi).T

            def project(d):
                if npar == 1:
                    y = (d.reshape(n_pair*nmu12, nmu1*nphi) @ angT
                         ).reshape(n_pair, nmu12, nq)
                    return jnp.einsum('pjq,pqj->pq', y, rad)[..., None]
                y = jnp.einsum('nkm,kq->nqm', d.reshape(
                    n_pair*nmu12, nmu1*nphi, npar), angT
                    ).reshape(n_pair, nmu12, nq, npar)
                return jnp.einsum('pjqm,pqj->pqm', y, rad)
            mom = jnp.stack([project(d) for d in cols], axis=-1)
            n_tree = len(tree_spec)
            return mom[..., :n_tree], mom[..., n_tree:]

        _fused_moments_diagrams_jax = staticmethod(jax.jit(
            _fused_moments_diagrams_jax.__func__,
            static_argnames=('tree_spec', 'stoch_spec', 'vdg')))
