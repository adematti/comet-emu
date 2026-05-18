"""Bispectrum module."""

import numpy as np
import numba as nb
import pickle
from scipy.interpolate import CubicSpline
from scipy.special import factorial, lpmv
from comet.grid import Grid, CtypedGrid

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



class BispectrumNum:
    """Numerical-integration bispectrum multipoles.

    Supports two projection bases, both using shared 5D kernel evaluation.

    - **Sugiyama**: Projects onto ``(l1, l2, L)`` multipoles via a
      ``(mu1, mu12, phi)`` quadrature per ``(k1, k2)`` pair.
    - **Scoccimarro**: Projects onto ``(l, m)`` multipoles via a
      ``(mu1, phi)`` quadrature per ``(k1, k2, k3)`` triangle.

    Supports real-space and redshift-space (VDG_infty / EFT / VDG_infty_ctr).
    The EggLeeSco EFT counterterm is enabled by including ``'EFT'`` or
    ``'VDG_infty_ctr'`` in ``model``; it modifies the SPT tree integrand
    multiplicatively by ``1 + cnloB * sum_j(k_j^2 mu_j^2)``.

    Two inner-loop backends are available:

    - ``backend='numba'`` (default): fused njit kernels.
    - ``backend='jax'``: JAX-traced kernels (requires `jax`).

    """

    def __init__(self, real_space, model, use_Mpc, backend='numba'):
        self.real_space = real_space
        self.model = model
        self.use_Mpc = use_Mpc
        self.nbar = 1.0
        self.cnlo_type = 'EggLeeSco'
        if backend not in ('numba', 'jax'):
            raise ValueError(
                f"backend must be 'numba' or 'jax'; got {backend!r}.")
        if backend == 'jax' and not HAS_JAX:
            raise ImportError(
                "backend='jax' requested but JAX is not installed.")
        self.backend = backend
        self._sugi_quad_cache = {}
        self._sugi_proj_cache = {}
        self._sugi_proj_cache_k3 = {}
        self._scocc_quad_cache = {}
        self._scocc_proj_cache = {}

        self.tree_diagrams = ('B0L_b1b1b1', 'B0L_b1b1', 'B0L_b1',
                              'B0L_b1b1b2', 'B0L_b1b2', 'B0L_b2',
                              'B0L_b1b1g2', 'B0L_b1g2', 'B0L_g2',
                              'B0L_id')
        self.tree_index = {d: i for i, d in enumerate(self.tree_diagrams)}
        self.stoch_diagrams = ('Bnoise_MB0b1b1', 'Bnoise_MB0b1', 'Bnoise_NP0', 'Bnoise_NB0')
        self.stoch_index = {d: i for i, d in enumerate(self.stoch_diagrams)}

    def define_nbar(self, nbar):
        self.nbar = np.copy(nbar)

    def define_units(self, use_Mpc):
        self.use_Mpc = use_Mpc

    def change_cnlo_type(self, type):
        """Switch the EFT counterterm flavour. Mirrors `Bispectrum.change_cnlo_type`."""
        if type in ('EggLeeSco', 'IvaPhiNis'):
            self.cnlo_type = type
        else:
            raise ValueError(
                f"Unknown cnlo_type {type!r}; use 'EggLeeSco' or 'IvaPhiNis'.")

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

  
    def _eval_pdw_legs(self, Pdw_eval, k1_p, k2_p, k3_p, full_shape):
        """Evaluate Pdw at the AP-corrected wavemodes for all three legs
        with a single ``Pdw_eval`` call.

        For single-z (``nparams == 1``) the flat ``[k1_p, k2_p, k3_p]``
        array is fed directly to ``Pdw_eval``.

        For batched-z we instead evaluate ``Pdw_eval`` on a compressed
        100-point kgrid that covers the AP-corrected range and then
        build a per-z `CubicSpline`. 
        """
        nparams = full_shape[-1]
        if nparams == 1:
            N_full = int(np.prod(full_shape))
            k1_flat = np.ascontiguousarray(
                np.broadcast_to(k1_p, full_shape)).ravel()
            k2_flat = np.ascontiguousarray(
                np.broadcast_to(k2_p, full_shape)).ravel()
            k3_flat = np.ascontiguousarray(
                np.broadcast_to(k3_p, full_shape)).ravel()
            Pdw_all = Pdw_eval(np.concatenate([k1_flat, k2_flat, k3_flat]))
            pdw1 = Pdw_all[:N_full].reshape(full_shape)
            pdw2 = Pdw_all[N_full:2*N_full].reshape(full_shape)
            pdw3 = Pdw_all[2*N_full:].reshape(full_shape)
            return pdw1, pdw2, pdw3

        kmin = min(float(np.min(k1_p)), float(np.min(k2_p)),
                   float(np.min(k3_p)))
        kmax = max(float(np.max(k1_p)), float(np.max(k2_p)),
                   float(np.max(k3_p)))
        kgrid = self._kgrid_compression(kmin * 0.99, kmax * 1.01)
        Pdw_grid = Pdw_eval(kgrid)  # shape (nk, nparams)
        pdw1 = np.empty(full_shape)
        pdw2 = np.empty(full_shape)
        pdw3 = np.empty(full_shape)
        # Materialize broadcast views so we can slice the last axis per-z.
        k1_b = np.broadcast_to(k1_p, full_shape)
        k2_b = np.broadcast_to(k2_p, full_shape)
        k3_b = np.broadcast_to(k3_p, full_shape)
        for iz in range(nparams):
            cs = CubicSpline(kgrid, Pdw_grid[:, iz])
            pdw1[..., iz] = cs(k1_b[..., iz])
            pdw2[..., iz] = cs(k2_b[..., iz])
            pdw3[..., iz] = cs(k3_b[..., iz])
        return pdw1, pdw2, pdw3


    # def _tree_bias_coeffs(self, params):
    #     """SPT bias coefficients, shape ``(n_diag_spt, nparams)``."""
    #     b1 = np.atleast_1d(params['b1'])
    #     b2 = np.atleast_1d(params['b2'])
    #     g2 = np.atleast_1d(params['g2'])
    #     ones = np.ones_like(b1)
    #     return np.array([
    #         b1**3,         # B0L_b1b1b1
    #         b1**2,         # B0L_b1b1
    #         b1,            # B0L_b1
    #         b1**2 * b2,    # B0L_b1b1b2
    #         b1 * b2,       # B0L_b1b2
    #         b2,            # B0L_b2
    #         b1**2 * g2,    # B0L_b1b1g2
    #         b1 * g2,       # B0L_b1g2
    #         g2,            # B0L_g2
    #         ones,          # B0L_id
    #     ])

    # def __stoch_bias_coeffs(self, params):
    #     """Stochastic bias coefficients, shape ``(n_diag_stoch, nparams)``.

    #     Mirrors the analytical-path convention (`_update_AP_params` /
    #     `Bell`): the per-leg coefficients carry the `1/nbar` factor and
    #     the `Bnoise_NB0` coefficient carries `1/nbar^2`. Diagram values
    #     themselves only carry the AP volume factor `1/qiso6`.
    #     """
    #     b1 = np.atleast_1d(params['b1'])
    #     MB0 = np.atleast_1d(params['MB0'])
    #     NP0 = np.atleast_1d(params['NP0'])
    #     NB0 = np.atleast_1d(params['NB0'])
    #     return np.array([
    #         b1**2 * MB0,         # Bnoise_MB0b1b1
    #         b1 * (MB0 + NP0),    # Bnoise_MB0b1
    #         NP0 * np.ones_like(b1),         # Bnoise_NP0
    #         NB0 * np.ones_like(b1) / self.nbar,  # Bnoise_NB0  (only (0,0,0))
    #     ]) / self.nbar

    def _eval_sugi_geometry(self, pair, params, nmu1, nmu12, nphi, mu12_transform):
        """Build the full 5D (k1, k2, k3, mu1, mu2, mu3) geometry on the
        Sugiyama ``(mu1, mu12, phi)`` quadrature grid, with AP correction applied.
        Returns broadcast contiguous arrays sized
        ``(n_pair, nmu1, nmu12, nphi, nparams)`` plus the AP volume factor ``qiso6``.
        """
        pair = np.atleast_2d(pair)
        n_pair = pair.shape[0]
        nparams = np.atleast_1d(params['q_lo']).size

        mu1_g, _, mu12_g, _, cphi_g, _, _ = \
            self._sugi_get_quadrature(nmu1, nmu12, nphi, mu12_transform)
        mu1 = mu1_g[..., None]
        cphi = cphi_g[..., None]
        
        k1 = pair[:, 0][:, None, None, None, None]
        k2 = pair[:, 1][:, None, None, None, None]

        if mu12_transform == 'k3':
            t = mu12_g[..., None]              # (1, 1, nmu12, 1, 1)
            k_lo = np.abs(k1 - k2)
            k_hi = k1 + k2
            k3, _ = self._k3_from_t(t, k_lo, k_hi)
            k3 = np.maximum(k3, 1e-30)
            mu12 = (k3*k3 - k1*k1 - k2*k2) / (2.0 * k1 * k2)
            mu12 = np.clip(mu12, -1.0, 1.0)
        else:
            mu12 = mu12_g[..., None]
            k3 = np.sqrt(k1*k1 + k2*k2 + 2.0*k1*k2*mu12)
            k3 = np.maximum(k3, 1e-30)
        sin_mu12 = np.sqrt(np.maximum(1.0 - mu12*mu12, 0.0))
        sin_mu1 = np.sqrt(np.maximum(1.0 - mu1*mu1, 0.0))
        mu2 = mu12*mu1 + sin_mu12*sin_mu1*cphi
        mu3 = -(k1*mu1 + k2*mu2) / k3

        qlo = np.atleast_1d(params['q_lo'])
        qtr = np.atleast_1d(params['q_tr'])
        qiso6 = qlo**2 * qtr**4

        k1_p, mu1_p = self._apply_ap(k1, mu1, qlo, qtr)
        k2_p, mu2_p = self._apply_ap(k2, mu2, qlo, qtr)
        k3_p, mu3_p = self._apply_ap(k3, mu3, qlo, qtr)
        full_shape = (n_pair, nmu1, nmu12, nphi, nparams)

        return (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
                qiso6, full_shape)
     


    def _eval_sugi_5d_bispectrum(self, pair, Pdw_eval, params,
                                 nmu1, nmu12, nphi, mu12_transform):
        """5D bispectrum (no diagram axis) on the
        ``(mu1, mu12, phi)`` quadrature grid.

        Returns
        -------
        B_5d : ndarray, shape (n_pair, nmu1, nmu12, nphi, nparams)
            Bias-weighted VDG bispectrum at each quadrature node, with
            AP volume factor already applied. The constant `NB0` piece is
            *not* included (it is added analytically by `Bell_sugiyama`).
        qiso6 : ndarray, shape (nparams,)
        """
        (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
         qiso6, full_shape) = self._eval_sugi_geometry(
            pair, params, nmu1, nmu12, nphi, mu12_transform)
        pdw1, pdw2, pdw3 = self._eval_pdw_legs(
            Pdw_eval, k1_p, k2_p, k3_p, full_shape)

        b1 = np.ascontiguousarray(np.atleast_1d(params['b1']))
        b2 = np.ascontiguousarray(np.atleast_1d(params['b2']))
        g2 = np.ascontiguousarray(np.atleast_1d(params['g2']))
        f = np.ascontiguousarray(np.atleast_1d(params['f']))
        avirB, sv = self._get_damping_arrays(params)
        MB0 = np.ascontiguousarray(np.atleast_1d(params['MB0']))
        NP0 = np.ascontiguousarray(np.atleast_1d(params['NP0']))
        cnloB, cB1, cB2 = self._get_ctr_arrays(params)
        inv_qiso6 = np.ascontiguousarray(1.0 / qiso6)
        nb = np.atleast_1d(np.asarray(self.nbar, dtype=float)).reshape(-1)
        if nb.size == 1:
            nb = np.broadcast_to(nb, inv_qiso6.shape)
        inv_nbar = np.ascontiguousarray(1.0 / nb)

        if self.backend == 'jax':
            B_5d = np.asarray(self._bispectrum_5d_jax_fused(
                jnp.asarray(k1_p), jnp.asarray(k2_p), jnp.asarray(k3_p),
                jnp.asarray(mu1_p), jnp.asarray(mu2_p), jnp.asarray(mu3_p),
                jnp.asarray(pdw1), jnp.asarray(pdw2), jnp.asarray(pdw3),
                jnp.asarray(b1), jnp.asarray(b2), jnp.asarray(g2),
                jnp.asarray(f), jnp.asarray(avirB), jnp.asarray(sv),
                jnp.asarray(MB0), jnp.asarray(NP0), 
                jnp.asarray(cnloB), jnp.asarray(cB1), jnp.asarray(cB2),
                jnp.asarray(inv_nbar), jnp.asarray(inv_qiso6)))
            B_5d = np.broadcast_to(B_5d, full_shape)
            return B_5d, qiso6

 
        def _full(a):
            return np.ascontiguousarray(np.broadcast_to(a, full_shape))
        k1_b, k2_b, k3_b = _full(k1_p), _full(k2_p), _full(k3_p)
        mu1_b, mu2_b, mu3_b = _full(mu1_p), _full(mu2_p), _full(mu3_p)
        pdw1, pdw2, pdw3 = _full(pdw1), _full(pdw2), _full(pdw3)
        B_5d = np.empty(full_shape, dtype=float)
        self._bispectrum_5d_njit(
            k1_b, k2_b, k3_b, mu1_b, mu2_b, mu3_b,
            pdw1, pdw2, pdw3,
            b1, b2, g2, f, avirB, sv, MB0, NP0,
            cnloB, cB1, cB2,
            inv_nbar, inv_qiso6, B_5d)
        return B_5d, qiso6

    def _eval_sugi_5d_diagrams(self, pair, Pdw_eval, params,
                               nmu1, nmu12, nphi, mu12_transform,
                               tree_keep=None, stoch_keep=None):
        """Build the diagram-stacked 5D bispectrum on the (mu1, mu12, phi)
        quadrature grid.

        Mirrors `_eval_sugi_5d_bispectrum` but emits per-diagram (bias-stripped)
        contributions instead of the bias-collapsed scalar.

        ``tree_keep`` / ``stoch_keep`` are tuples of diagram indices to
        compute (defaults: all 10 SPT diagrams, all 3 stoch integrals).
        Output stacks are compacted along the last axis to ``len(tree_keep)``
        and ``len(stoch_keep)`` respectively.

        Parameters
        ----------
        pair : ndarray, shape (n_pair, 2)
            (k1, k2) pairs.
        Pdw_eval : callable
            Single-shot Pdw evaluator (see `_eval_5d_bispectrum`).
        params : dict
            COMET parameter dict; per-key arrays of length ``nparams``.

        Returns
        -------
        spt_stack, stoch_stack : ndarrays of shape
            ``(n_pair, nmu1, nmu12, nphi, nparams, n_keep)``
        qiso6 : ndarray, shape ``(nparams,)``
            AP volume factor :math:`q_\\parallel^2 q_\\perp^4`.
        """
        if tree_keep is None:
            tree_keep = tuple(range(len(self.tree_diagrams)))
        else:
            tree_keep = tuple(tree_keep)
        if stoch_keep is None:
            stoch_keep = tuple(range(len(self.stoch_diagrams) - 1))
        else:
            stoch_keep = tuple(stoch_keep)

        (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
         qiso6, full_shape) = self._eval_sugi_geometry(
            pair, params, nmu1, nmu12, nphi, mu12_transform)
        pdw1, pdw2, pdw3 = self._eval_pdw_legs(
            Pdw_eval, k1_p, k2_p, k3_p, full_shape)

        f = np.ascontiguousarray(np.atleast_1d(params['f']))
        avirB, sv = self._get_damping_arrays(params)
        cnloB, cB1, cB2 = self._get_ctr_arrays(params)
        inv_qiso6 = np.ascontiguousarray(1.0 / qiso6)

        n_tree_keep = len(tree_keep)
        n_stoch_keep = len(stoch_keep)

        if self.backend == 'jax':
            spt_jax, stoch_jax = self._bispectrum_5d_jax_diagrams(
                jnp.asarray(k1_p), jnp.asarray(k2_p), jnp.asarray(k3_p),
                jnp.asarray(mu1_p), jnp.asarray(mu2_p), jnp.asarray(mu3_p),
                jnp.asarray(pdw1), jnp.asarray(pdw2), jnp.asarray(pdw3),
                jnp.asarray(f), jnp.asarray(avirB), jnp.asarray(sv),
                jnp.asarray(cnloB), jnp.asarray(inv_qiso6),
                tree_keep, stoch_keep)
            spt_stack = np.broadcast_to(
                np.asarray(spt_jax), full_shape + (n_tree_keep,))
            stoch_stack = np.broadcast_to(
                np.asarray(stoch_jax), full_shape + (n_stoch_keep,))
            return spt_stack, stoch_stack, qiso6

        def _full(a):
            return np.ascontiguousarray(np.broadcast_to(a, full_shape))
        k1_b, k2_b, k3_b = _full(k1_p), _full(k2_p), _full(k3_p)
        mu1_b, mu2_b, mu3_b = _full(mu1_p), _full(mu2_p), _full(mu3_p)
        pdw1_b, pdw2_b, pdw3_b = _full(pdw1), _full(pdw2), _full(pdw3)

        tree_col = np.full(len(self.tree_diagrams), -1, dtype=np.int64)
        for k, d in enumerate(tree_keep):
            tree_col[d] = k
        stoch_col = np.full(len(self.stoch_diagrams) - 1, -1, dtype=np.int64)
        for k, d in enumerate(stoch_keep):
            stoch_col[d] = k

        N = int(np.prod(full_shape))
        spt_flat = np.empty((N, max(n_tree_keep, 1)), dtype=float)
        stoch_flat = np.empty((N, max(n_stoch_keep, 1)), dtype=float)
        self._bispectrum_5d_diagrams_njit(
            k1_b, k2_b, k3_b, mu1_b, mu2_b, mu3_b,
            pdw1_b, pdw2_b, pdw3_b,
            f, avirB, sv, cnloB,
            inv_qiso6, tree_col, stoch_col, spt_flat, stoch_flat)
        if n_tree_keep == 0:
            spt_stack = np.empty(full_shape + (0,), dtype=float)
        else:
            spt_stack = spt_flat[:, :n_tree_keep].reshape(
                full_shape + (n_tree_keep,))
        if n_stoch_keep == 0:
            stoch_stack = np.empty(full_shape + (0,), dtype=float)
        else:
            stoch_stack = stoch_flat[:, :n_stoch_keep].reshape(
                full_shape + (n_stoch_keep,))
        return spt_stack, stoch_stack, qiso6

    def _sugi_project_stack(self, stack, ell, nmu1, nmu12, nphi,
                            mu12_transform):
        """Project a ``(n_pair, nmu1, nmu12, nphi, nparams, n_diag)`` stack
        onto each requested ``(l1, l2, L)`` multipole. Returns
        ``{ll: shape (n_pair, nparams, n_diag) array}``."""
        proj_ops = self._sugi_get_proj_ops(
            ell, nmu1, nmu12, nphi, mu12_transform)
        n_pair = stack.shape[0]
        nparams, n_diag = stack.shape[-2], stack.shape[-1]
        flat = stack.reshape(n_pair, nmu1*nmu12*nphi, nparams, n_diag)
        return {tuple(ll): np.einsum('ijcd,j->icd', flat, proj_ops[tuple(ll)],
                                     optimize='optimal') for ll in ell}

    def Bell_Sugi(self, pair, Pdw_eval, params,
                  ell=((0, 0, 0), (2, 0, 2)),
                  nmu1=5, nmu12=12, nphi=5,
                  mu12_transform='quadratic'):
        """Bispectrum Sugiyama multipoles via numerical 3D quadrature.

        Returns
        -------
        dict
            ``{(l1, l2, L): ndarray}`` with shape ``(n_pair,)`` for
            single-z and ``(n_pair, nparams)`` for batched-z.
        """

        ell = tuple(tuple(ll) for ll in ell)
        B_5d, qiso6 = self._eval_sugi_5d_bispectrum(
            pair, Pdw_eval, params, nmu1, nmu12, nphi, mu12_transform)

        n_pair = B_5d.shape[0]
        nparams = qiso6.size
        B_flat = B_5d.reshape(n_pair, nmu1*nmu12*nphi, nparams)
        NB0 = np.ascontiguousarray(np.atleast_1d(params['NB0']))
        inv_nbar = np.ascontiguousarray(np.broadcast_to(1.0 / np.atleast_1d(self.nbar), qiso6.shape))
        Bell_dict = {}
        if mu12_transform == 'k3':
            # Per-pair projection kernel; the dmu12/dt Jacobian is folded
            # into `proj_ops[ll][p, q]` so the contraction is per-pair.
            proj_ops = self._sugi_get_proj_ops_k3(
                ell, pair, nmu1, nmu12, nphi, mu12_transform)
            for ll in ell:
                B = np.einsum('ijc,ij->ic', B_flat, proj_ops[ll],
                              optimize='optimal')
                if ll == (0, 0, 0):
                    B = B + (NB0 / self.nbar**2)[None, :] / qiso6[None, :]
                Bell_dict[ll] = np.squeeze(B)
        else:
            proj_ops = self._sugi_get_proj_ops(
                ell, nmu1, nmu12, nphi, mu12_transform)
            for ll in ell:
                B = np.einsum('ijc,j->ic', B_flat, proj_ops[ll],
                              optimize='optimal')
                if ll == (0, 0, 0):
                    B = B + (NB0 * inv_nbar**2)[None, :] / qiso6[None, :]
                Bell_dict[ll] = np.squeeze(B)
        return Bell_dict

    def BX_ell_Sugi(self, pair, Pdw_eval, params,
                    ell=((0, 0, 0), (2, 0, 2)),
                    nmu1=5, nmu12=12, nphi=5,
                    mu12_transform='quadratic',
                    X_list=None):
        """Diagram-resolved Sugiyama bispectrum multipoles.

        If ``X_list`` is **None** (default), returns
        ``{(l1, l2, L): {diagram_name: ndarray}}`` where each diagram value
        is the contribution stripped of its bias coefficient (same
        convention as `BX_ell`). Single-z output is squeezed to
        ``(n_pair,)``; batched-z output is ``(n_pair, nparams)``.

        If ``X_list`` is provided (a string or iterable of diagram names),
        returns ``{(l1, l2, L): ndarray}`` where the array has shape
        ``(n_pair, nx, nparams)`` for batched-z and ``(n_pair, nx)`` for
        single-z, with the diagram axis ordered to match ``X_list``. Only
        the requested diagrams are projected.

        The `Bnoise_NB0` slice is `1/qiso6` for `(0, 0, 0)` and zero
        otherwise, matching the analytical path.

        `Pdw_eval` is the single-shot Pdw callable consumed by the
        integrator; see `_eval_sugi_5d_diagrams` for its contract.
        """

        if 'VDG_infty' not in self.model:
            raise NotImplementedError(
                "BX_ell_sugiyama is currently implemented only for VDG_infty "
                "models; got model={!r}.".format(self.model))

        ell = tuple(tuple(ll) for ll in ell)
        nb0_name = self.stoch_diagrams[-1]
        tree_set = set(self.tree_diagrams)
        stoch_set = set(self.stoch_diagrams[:-1])

        if X_list is None:
            tree_keep = tuple(range(len(self.tree_diagrams)))
            stoch_keep = tuple(range(len(self.stoch_diagrams) - 1))
        else:
            X_list_local = ([X_list] if isinstance(X_list, str)
                            else list(X_list))
            valid_names = tree_set | stoch_set | {nb0_name}
            for x in X_list_local:
                if x not in valid_names:
                    raise ValueError(
                        "Unknown diagram '{}' in X_list. Valid names: {}."
                        .format(x, sorted(valid_names)))
            tree_keep = tuple(sorted({self.tree_index[x] for x in X_list_local
                                      if x in tree_set}))
            stoch_keep = tuple(sorted({self.stoch_index[x] for x in X_list_local
                                       if x in stoch_set}))

        spt_stack, stoch_stack, qiso6 = self._eval_sugi_5d_diagrams(
            pair, Pdw_eval, params, nmu1, nmu12, nphi, mu12_transform,
            tree_keep=tree_keep, stoch_keep=stoch_keep)

        n_pair = np.atleast_2d(pair).shape[0]
        nparams = qiso6.size
        nb0_value = np.broadcast_to(1.0 / qiso6[None, :], (n_pair, nparams))

        spt_proj = None
        stoch_proj = None
        if tree_keep:
            spt_proj = self._sugi_project_stack(
                spt_stack, ell, nmu1, nmu12, nphi, mu12_transform)
        if stoch_keep:
            stoch_proj = self._sugi_project_stack(
                stoch_stack, ell, nmu1, nmu12, nphi, mu12_transform)

        tree_pos = {orig: k for k, orig in enumerate(tree_keep)}
        stoch_pos = {orig: k for k, orig in enumerate(stoch_keep)}

        if X_list is None:
            BX_ell_dict = {}
            for ll in ell:
                out = {}
                for name in self.tree_diagrams:
                    d = tree_pos[self.tree_index[name]]
                    out[name] = np.squeeze(spt_proj[ll][:, :, d])
                for name in self.stoch_diagrams[:-1]:
                    d = stoch_pos[self.stoch_index[name]]
                    out[name] = np.squeeze(stoch_proj[ll][:, :, d])
                if ll == (0, 0, 0):
                    out[nb0_name] = np.squeeze(nb0_value)
                else:
                    out[nb0_name] = \
                        np.squeeze(np.zeros((n_pair, nparams)))
                BX_ell_dict[ll] = out
            return BX_ell_dict

        nx = len(X_list_local)
        BX_ell_dict = {}
        for ll in ell:
            res = np.empty((n_pair, nx, nparams))
            for ix, name in enumerate(X_list_local):
                if name == nb0_name:
                    if ll == (0, 0, 0):
                        res[:, ix, :] = nb0_value
                    else:
                        res[:, ix, :] = 0.0
                elif name in tree_set:
                    d = tree_pos[self.tree_index[name]]
                    res[:, ix, :] = spt_proj[ll][:, :, d]
                else:
                    d = stoch_pos[self.stoch_index[name]]
                    res[:, ix, :] = stoch_proj[ll][:, :, d]
            if nparams == 1:
                res = res[..., 0]
            BX_ell_dict[ll] = res
        return BX_ell_dict


    def _eval_scocc_geometry(self, tri, params, nmu, nphi):
        """Build the full 5D (k1, k2, k3, mu1, mu2, mu3) geometry on the
        Scoccimarro ``(mu1, phi)`` quadrature grid, with AP correction applied.
        Returns broadcast contiguous arrays sized
        ``(n_tri, nmu, nphi, nparams)`` plus the AP volume factor ``qiso6``.
        """
        tri = np.atleast_2d(tri)
        n_tri = tri.shape[0]
        nparams = np.atleast_1d(params['q_lo']).size

        mu_g, _, cphi_g, _, _, _ = self._scocc_get_quadrature(nmu, nphi)
        # Add trailing nparams axis: (1, nmu, 1, 1), (1, 1, nphi, 1)
        mu1 = mu_g[..., None]
        cphi = cphi_g[..., None]

        k1 = tri[:, 0][:, None, None, None]
        k2 = tri[:, 1][:, None, None, None]
        k3 = tri[:, 2][:, None, None, None]

        # Triangle condition cosine.
        mu12 = (k3*k3 - k1*k1 - k2*k2) / (2.0 * k1 * k2)
        mu12 = np.clip(mu12, -1.0, 1.0)
        sin_mu12 = np.sqrt(np.maximum(1.0 - mu12*mu12, 0.0))
        sin_mu1 = np.sqrt(np.maximum(1.0 - mu1*mu1, 0.0))
        mu2 = mu12*mu1 + sin_mu12*sin_mu1*cphi
        mu3 = -(k1*mu1 + k2*mu2) / np.maximum(k3, 1e-30)

        qlo = np.atleast_1d(params['q_lo'])
        qtr = np.atleast_1d(params['q_tr'])
        qiso6 = qlo**2 * qtr**4

        k1_p, mu1_p = self._apply_ap(k1, mu1, qlo, qtr)
        k2_p, mu2_p = self._apply_ap(k2, mu2, qlo, qtr)
        k3_p, mu3_p = self._apply_ap(k3, mu3, qlo, qtr)

        full_shape = (n_tri, nmu, nphi, nparams)
        return (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
                qiso6, full_shape)

    def _eval_scocc_5d_bispectrum(self, tri, Pdw_eval, params, nmu, nphi):
        """Bias-weighted bispectrum on the Scoccimarro (mu, phi) grid via 5D
        kernel evaluation.

        Internally uses the shared 5D kernel `_bispectrum_5d_njit` evaluated
        on the full (k1, k2, k3, mu1, mu2, mu3) space, then projected onto
        Scoccimarro (mu1, phi) multipoles. Returns ``(B_out, qiso6)`` where
        ``B_out`` has shape ``(n_tri, nmu, nphi, nparams)``.
        """
        (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
         qiso6, full_shape) = self._eval_scocc_geometry(
            tri, params, nmu, nphi)

        pdw1, pdw2, pdw3 = self._eval_pdw_legs(
            Pdw_eval, k1_p, k2_p, k3_p, full_shape)

        b1 = np.ascontiguousarray(np.atleast_1d(params['b1']))
        b2 = np.ascontiguousarray(np.atleast_1d(params['b2']))
        g2 = np.ascontiguousarray(np.atleast_1d(params['g2']))
        f = np.ascontiguousarray(np.atleast_1d(params['f']))
        avir, sv = self._get_damping_arrays(params)
        MB0 = np.ascontiguousarray(np.atleast_1d(params['MB0']))
        NP0 = np.ascontiguousarray(np.atleast_1d(params['NP0']))
        cnloB, cB1, cB2 = self._get_ctr_arrays(params)
        inv_qiso6 = np.ascontiguousarray(1.0 / qiso6)
        nb = np.atleast_1d(np.asarray(self.nbar, dtype=float)).reshape(-1)
        if nb.size == 1:
            nb = np.broadcast_to(nb, inv_qiso6.shape)
        inv_nbar = np.ascontiguousarray(1.0 / nb)

        if self.backend == 'jax':
            B_4d = np.asarray(self._bispectrum_5d_jax_fused(
                jnp.asarray(k1_p), jnp.asarray(k2_p), jnp.asarray(k3_p),
                jnp.asarray(mu1_p), jnp.asarray(mu2_p), jnp.asarray(mu3_p),
                jnp.asarray(pdw1), jnp.asarray(pdw2), jnp.asarray(pdw3),
                jnp.asarray(b1), jnp.asarray(b2), jnp.asarray(g2),
                jnp.asarray(f), jnp.asarray(avir), jnp.asarray(sv),
                jnp.asarray(MB0), jnp.asarray(NP0), 
                jnp.asarray(cnloB), jnp.asarray(cB1), jnp.asarray(cB2),
                jnp.asarray(inv_nbar), jnp.asarray(inv_qiso6)))
            B_4d = np.broadcast_to(B_4d, full_shape)
            return B_4d, qiso6

        def _full(a):
            return np.ascontiguousarray(np.broadcast_to(a, full_shape))
        k1_b, k2_b, k3_b = _full(k1_p), _full(k2_p), _full(k3_p)
        mu1_b, mu2_b, mu3_b = _full(mu1_p), _full(mu2_p), _full(mu3_p)
        pdw1, pdw2, pdw3 = _full(pdw1), _full(pdw2), _full(pdw3)
        B_5d = np.empty(full_shape, dtype=float)
        self._bispectrum_5d_njit(
            k1_b, k2_b, k3_b, mu1_b, mu2_b, mu3_b,
            pdw1, pdw2, pdw3,
            b1, b2, g2, f, avir, sv, MB0, NP0,
            cnloB, cB1, cB2,
            inv_nbar, inv_qiso6, B_5d)
        return B_5d, qiso6

    def _eval_scocc_5d_diagrams(self, tri, Pdw_eval, params, nmu, nphi,
                                tree_keep=None, stoch_keep=None):
        """Per-diagram bias-stripped bispectrum on the Scoccimarro (mu, phi)
        grid via 5D kernel evaluation.

        Internally uses the shared 5D diagram kernel `_bispectrum_5d_diagrams_njit`
        on the full (k1, k2, k3, mu1, mu2, mu3) space, then projects onto
        Scoccimarro multipoles. Returns ``(spt_stack, stoch_stack, qiso6)``
        with shape ``(n_tri, nmu, nphi, nparams, n_keep)``. ``tree_keep`` and
        ``stoch_keep`` select which diagrams to compute (default: all).
        """
        if tree_keep is None:
            tree_keep = tuple(range(len(self.tree_diagrams)))
        else:
            tree_keep = tuple(tree_keep)
        if stoch_keep is None:
            stoch_keep = tuple(range(len(self.stoch_diagrams) - 1))
        else:
            stoch_keep = tuple(stoch_keep)

        (k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
         qiso6, full_shape) = self._eval_scocc_geometry(
            tri, params, nmu, nphi)

        pdw1, pdw2, pdw3 = self._eval_pdw_legs(
            Pdw_eval, k1_p, k2_p, k3_p, full_shape)

        f = np.ascontiguousarray(np.atleast_1d(params['f']))
        avir, sv = self._get_damping_arrays(params)
        cnloB, cB1, cB2 = self._get_ctr_arrays(params)
        inv_qiso6 = np.ascontiguousarray(1.0 / qiso6)

        n_tree_keep = len(tree_keep)
        n_stoch_keep = len(stoch_keep)

        if self.backend == 'jax':
            spt_jax, stoch_jax = self._bispectrum_5d_jax_diagrams(
                jnp.asarray(k1_p), jnp.asarray(k2_p), jnp.asarray(k3_p),
                jnp.asarray(mu1_p), jnp.asarray(mu2_p), jnp.asarray(mu3_p),
                jnp.asarray(pdw1), jnp.asarray(pdw2), jnp.asarray(pdw3),
                jnp.asarray(f), jnp.asarray(avir), jnp.asarray(sv),
                jnp.asarray(cnloB), jnp.asarray(inv_qiso6),
                tree_keep, stoch_keep)
            spt_stack = np.broadcast_to(
                np.asarray(spt_jax), full_shape + (n_tree_keep,))
            stoch_stack = np.broadcast_to(
                np.asarray(stoch_jax), full_shape + (n_stoch_keep,))
            return spt_stack, stoch_stack, qiso6

        def _full(a):
            return np.ascontiguousarray(np.broadcast_to(a, full_shape))
        k1_b, k2_b, k3_b = _full(k1_p), _full(k2_p), _full(k3_p)
        mu1_b, mu2_b, mu3_b = _full(mu1_p), _full(mu2_p), _full(mu3_p)
        pdw1_b, pdw2_b, pdw3_b = _full(pdw1), _full(pdw2), _full(pdw3)

        tree_col = np.full(len(self.tree_diagrams), -1, dtype=np.int64)
        for k, d in enumerate(tree_keep):
            tree_col[d] = k
        stoch_col = np.full(len(self.stoch_diagrams) - 1, -1, dtype=np.int64)
        for k, d in enumerate(stoch_keep):
            stoch_col[d] = k

        N = int(np.prod(full_shape))
        spt_flat = np.empty((N, max(n_tree_keep, 1)), dtype=float)
        stoch_flat = np.empty((N, max(n_stoch_keep, 1)), dtype=float)
        self._bispectrum_5d_diagrams_njit(
            k1_b, k2_b, k3_b, mu1_b, mu2_b, mu3_b,
            pdw1_b, pdw2_b, pdw3_b,
            f, avir, sv, cnloB,
            inv_qiso6, tree_col, stoch_col, spt_flat, stoch_flat)
        if n_tree_keep == 0:
            spt_stack = np.empty(full_shape + (0,), dtype=float)
        else:
            spt_stack = spt_flat[:, :n_tree_keep].reshape(
                full_shape + (n_tree_keep,))
        if n_stoch_keep == 0:
            stoch_stack = np.empty(full_shape + (0,), dtype=float)
        else:
            stoch_stack = stoch_flat[:, :n_stoch_keep].reshape(
                full_shape + (n_stoch_keep,))
        return spt_stack, stoch_stack, qiso6

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

    def Bell_Scocc(self, tri, Pdw_eval, params, ell=((0, 0), (2, 0)),
                   norm='sphharm', nmu=5, nphi=5):
        """Scoccimarro bispectrum multipoles via numerical 2D quadrature.

        Parameters
        ----------
        tri : ndarray, shape (n_tri, 3)
            Triangle wavemodes ``(k1, k2, k3)``.
        Pdw_eval : callable
            Single-shot Pdw evaluator (see `Bell_sugiyama`).
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

        ell = tuple(tuple(ll) for ll in ell)
        B_5d, qiso6 = self._eval_scocc_5d_bispectrum(
            tri, Pdw_eval, params, nmu, nphi)

        n_tri = B_5d.shape[0]
        nparams = qiso6.size
        B_flat = B_5d.reshape(n_tri, nmu*nphi, nparams)
        NB0 = np.ascontiguousarray(np.atleast_1d(params['NB0']))
        inv_nbar = np.ascontiguousarray(np.broadcast_to(1.0 / np.atleast_1d(self.nbar), qiso6.shape))
        proj_ops = self._scocc_get_proj_ops(ell, nmu, nphi)

        Bell_dict = {}
        for ll in ell:
            B = np.einsum('ijc,j->ic', B_flat, proj_ops[ll],
                          optimize='optimal')
            if ll == (0, 0):
                B = B + (NB0 * inv_nbar**2)[None, :] / qiso6[None, :]
            normfact = 1.0
            if norm == 'legendre':
                l, m = ll
                normfact = np.sqrt((2*l+1)/(4*np.pi))
            Bell_dict[ll] = normfact * np.squeeze(B)
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
        the requested diagrams are projected.
        """
        if self.real_space:
            raise NotImplementedError('Numerical quadrature is not implemented in real space yet.')

        if 'VDG_infty' not in self.model:
            raise NotImplementedError(
                "BX_ell_Scocc is currently implemented only for VDG_infty "
                "models; got model={!r}.".format(self.model))

        ell = tuple(tuple(ll) for ll in ell)
        nb0_name = self.stoch_diagrams[-1]
        tree_set = set(self.tree_diagrams)
        stoch_set = set(self.stoch_diagrams[:-1])

        if X_list is None:
            tree_keep = tuple(range(len(self.tree_diagrams)))
            stoch_keep = tuple(range(len(self.stoch_diagrams) - 1))
        else:
            X_list_local = ([X_list] if isinstance(X_list, str)
                            else list(X_list))
            valid_names = tree_set | stoch_set | {nb0_name}
            for x in X_list_local:
                if x not in valid_names:
                    raise ValueError(
                        "Unknown diagram '{}' in X_list. Valid names: {}."
                        .format(x, sorted(valid_names)))
            tree_keep = tuple(sorted({self.tree_index[x] for x in X_list_local
                                      if x in tree_set}))
            stoch_keep = tuple(sorted({self.stoch_index[x] for x in X_list_local
                                       if x in stoch_set}))

        spt_stack, stoch_stack, qiso6 = self._eval_scocc_5d_diagrams(
            tri, Pdw_eval, params, nmu, nphi,
            tree_keep=tree_keep, stoch_keep=stoch_keep)
        proj_ops = self._scocc_get_proj_ops(ell, nmu, nphi)

        n_tri = np.atleast_2d(tri).shape[0]
        nparams = qiso6.size
        nb0_value = np.broadcast_to(
            1.0 / qiso6[None, :], (n_tri, nparams))

        def _normfact(ll):
            if norm == 'legendre':
                l, _ = ll
                return np.sqrt((2*l+1)/(4*np.pi))
            return 1.0

        spt_flat = (spt_stack.reshape(n_tri, nmu*nphi, nparams, len(tree_keep))
                    if tree_keep else None)
        stoch_flat = (stoch_stack.reshape(n_tri, nmu*nphi, nparams,
                                          len(stoch_keep))
                      if stoch_keep else None)

        tree_pos = {orig: k for k, orig in enumerate(tree_keep)}
        stoch_pos = {orig: k for k, orig in enumerate(stoch_keep)}

        if X_list is None:
            BX_ell_dict = {}
            for ll in ell:
                normfact = _normfact(ll)
                spt_proj = normfact * np.einsum(
                    'ijcd,j->icd', spt_flat, proj_ops[ll], optimize='optimal')
                stoch_proj = normfact * np.einsum(
                    'ijcd,j->icd', stoch_flat, proj_ops[ll],
                    optimize='optimal')
                out = {}
                for name in self.tree_diagrams:
                    d = tree_pos[self.tree_index[name]]
                    out[name] = np.squeeze(spt_proj[:, :, d])
                for name in self.stoch_diagrams[:-1]:
                    d = stoch_pos[self.stoch_index[name]]
                    out[name] = np.squeeze(stoch_proj[:, :, d])
                if ll == (0, 0):
                    out[nb0_name] = np.squeeze(nb0_value)
                else:
                    out[nb0_name] = \
                        np.squeeze(np.zeros((n_tri, nparams)))
                BX_ell_dict[ll] = out
            return BX_ell_dict

        nx = len(X_list_local)
        BX_ell_dict = {}
        for ll in ell:
            normfact = _normfact(ll)
            spt_proj = None
            stoch_proj = None
            if tree_keep:
                spt_proj = normfact * np.einsum(
                    'ijcd,j->icd', spt_flat, proj_ops[ll],
                    optimize='optimal')
            if stoch_keep:
                stoch_proj = normfact * np.einsum(
                    'ijcd,j->icd', stoch_flat, proj_ops[ll],
                    optimize='optimal')

            res = np.empty((n_tri, nx, nparams))
            for ix, name in enumerate(X_list_local):
                if name == nb0_name:
                    if ll == (0, 0):
                        res[:, ix, :] = nb0_value
                    else:
                        res[:, ix, :] = 0.0
                elif name in tree_set:
                    d = tree_pos[self.tree_index[name]]
                    res[:, ix, :] = spt_proj[:, :, d]
                else:
                    d = stoch_pos[self.stoch_index[name]]
                    res[:, ix, :] = stoch_proj[:, :, d]
            if nparams == 1:
                res = res[..., 0]
            BX_ell_dict[ll] = res
        return BX_ell_dict


    # def _real_space_bispectrum(self, tri, Pdw_eval, params):
    #     """Real-space bispectrum at the given triangles. Returns the
    #     bias-stripped per-diagram dict matching `BX_ell_*` convention.

    #     Only ``b1b1b1`` (F2), ``b1b1b2``, ``b1b1g2`` (K), ``MB0b1b1`` and
    #     ``NB0`` are nonzero in real space; the others are filled with zeros
    #     so the canonical diagram set is always present.

    #     AP enters only through a uniform k-rescaling and the volume factor
    #     ``1/qiso6``, since there is no LOS in real space.
    #     """
    #     tri = np.atleast_2d(tri)
    #     k1, k2, k3 = tri[:, 0], tri[:, 1], tri[:, 2]
    #     nparams = np.atleast_1d(params['q_lo']).size
    #     n_tri = tri.shape[0]

    #     qlo = np.atleast_1d(params['q_lo'])
    #     qtr = np.atleast_1d(params['q_tr'])
    #     qiso6 = qlo**2 * qtr**4
    #     inv_q6 = 1.0 / qiso6

    #     # In real space the AP transform reduces to ``k_p = k / qtr``.
    #     # Build (n_tri, nparams) AP-corrected k arrays.
    #     k1_p = k1[:, None] / qtr[None, :]
    #     k2_p = k2[:, None] / qtr[None, :]
    #     k3_p = k3[:, None] / qtr[None, :]

    #     # F2 and K kernels per leg from AP-corrected k magnitudes.
    #     def _F2K(ki, kj, kk):
    #         muij = (kk*kk - ki*ki - kj*kj) / (2.0*ki*kj)
    #         muij = np.clip(muij, -1.0, 1.0)
    #         ratio = 0.5*(ki/kj + kj/ki)*muij
    #         F2 = 5.0/7.0 + 2.0/7.0*muij*muij + ratio
    #         K = muij*muij - 1.0
    #         return F2, K
    #     F2_12, K12 = _F2K(k1_p, k2_p, k3_p)
    #     F2_23, K23 = _F2K(k2_p, k3_p, k1_p)
    #     F2_31, K31 = _F2K(k3_p, k1_p, k2_p)

    #     # Single Pdw call covering all three legs across all z.
    #     if nparams == 1:
    #         k_flat = np.concatenate([k1_p.ravel(), k2_p.ravel(),
    #                                  k3_p.ravel()])
    #         pdw_all = Pdw_eval(k_flat)
    #         pdw1 = pdw_all[:n_tri][:, None]
    #         pdw2 = pdw_all[n_tri:2*n_tri][:, None]
    #         pdw3 = pdw_all[2*n_tri:][:, None]
    #     else:
    #         kmin = float(min(k1_p.min(), k2_p.min(), k3_p.min()))
    #         kmax = float(max(k1_p.max(), k2_p.max(), k3_p.max()))
    #         kgrid = self._kgrid_compression(kmin*0.99, kmax*1.01)
    #         pdw_grid = Pdw_eval(kgrid)  # (nk, nparams)
    #         pdw1 = np.empty((n_tri, nparams))
    #         pdw2 = np.empty((n_tri, nparams))
    #         pdw3 = np.empty((n_tri, nparams))
    #         for iz in range(nparams):
    #             cs = CubicSpline(kgrid, pdw_grid[:, iz])
    #             pdw1[:, iz] = cs(k1_p[:, iz])
    #             pdw2[:, iz] = cs(k2_p[:, iz])
    #             pdw3[:, iz] = cs(k3_p[:, iz])

    #     # EggLeeSco EFT counterterm in real space (mu=0): factor reduces to
    #     # ``1 + cnloB * (k_1^2 + k_2^2 + k_3^2)``. Off when cnloB is zero.
    #     cnloB, cB1, cB2 = self._get_ctr_arrays(params)
    #     k_sumsq = (k1_p*k1_p + k2_p*k2_p + k3_p*k3_p)
    #     eft = 1.0 + cnloB[None, :] * k_sumsq

    #     pp12 = pdw1 * pdw2
    #     pp23 = pdw2 * pdw3
    #     pp31 = pdw3 * pdw1
    #     d_b1b1b1 = (2.0*F2_12 * pp12 + 2.0*F2_23 * pp23
    #                 + 2.0*F2_31 * pp31) * eft
    #     d_b1b1b2 = (pp12 + pp23 + pp31) * eft
    #     d_b1b1g2 = (2.0*K12 * pp12 + 2.0*K23 * pp23
    #                 + 2.0*K31 * pp31) * eft
    #     s_mb0b1b1 = (pdw1 + pdw2 + pdw3)

    #     inv_q6_b = inv_q6[None, :]
    #     diagrams = {
    #         'B0L_b1b1b1': d_b1b1b1 * inv_q6_b,
    #         'B0L_b1b1b2': d_b1b1b2 * inv_q6_b,
    #         'B0L_b1b1g2': d_b1b1g2 * inv_q6_b,
    #         'Bnoise_MB0b1b1': s_mb0b1b1 * inv_q6_b / self.nbar,
    #         'Bnoise_NB0': np.broadcast_to(
    #             inv_q6_b, (n_tri, nparams)).copy(),
    #     }
    #     zero = np.zeros((n_tri, nparams))
    #     for name in list(self.tree_diagrams) + list(self.stoch_diagrams):
    #         if name not in diagrams:
    #             diagrams[name] = zero.copy()
    #     return diagrams

    # def _real_space_bell_scocc(self, tri, Pdw_eval, params, ell):
    #     """Scoccimarro multipoles in real space: only ``(0, 0)`` is
    #     nonzero; the rest are zero by symmetry. Diagrams already carry
    #     the 1/qiso6 and 1/nbar factors that match the RSD path.
    #     """
    #     diagrams = self._real_space_bispectrum(tri, Pdw_eval, params)
    #     b1 = np.atleast_1d(params['b1'])[None, :]
    #     b2 = np.atleast_1d(params['b2'])[None, :]
    #     g2 = np.atleast_1d(params['g2'])[None, :]
    #     MB0 = np.atleast_1d(params['MB0'])[None, :]
    #     NB0 = np.atleast_1d(params['NB0'])[None, :]
    #     B = (b1*b1*b1 * diagrams['B0L_b1b1b1']
    #          + b1*b1 * b2 * diagrams['B0L_b1b1b2']
    #          + b1*b1 * g2 * diagrams['B0L_b1b1g2']
    #          + b1*b1 * MB0 * diagrams['Bnoise_MB0b1b1']
    #          + NB0 / self.nbar * diagrams['Bnoise_NB0'])
    #     zero = np.zeros_like(B)
    #     result = {}
    #     for ll in ell:
    #         ll = tuple(ll)
    #         result[ll] = np.squeeze(B) if ll == (0, 0) else np.squeeze(zero)
    #     return result

    # def _real_space_bxell_scocc(self, tri, Pdw_eval, params, ell):
    #     """Diagram-resolved real-space Scoccimarro: only ``(0, 0)`` is
    #     populated."""
    #     diagrams = self._real_space_bispectrum(tri, Pdw_eval, params)
    #     squeezed = {k: np.squeeze(v) for k, v in diagrams.items()}
    #     zero_dict = {k: np.zeros_like(v) for k, v in squeezed.items()}
    #     out = {}
    #     for ll in ell:
    #         ll = tuple(ll)
    #         out[ll] = squeezed if tuple(ll) == (0, 0) else zero_dict
    #     return out

    @staticmethod
    def _apply_ap(k, mu, qlo, qtr):
        F = qlo / qtr
        fac = np.sqrt(1.0 + mu**2 * (1.0/F**2 - 1.0))
        return k / qtr * fac, mu / F / fac

    @staticmethod
    def _sph_harm(l, m, costheta, phi):
        """Complex spherical harmonic Y_l^m(theta, phi) using Condon-Shortley sign.
        Matches the convention used in the original external implementation."""
        norm = np.sqrt(factorial(l - abs(m)) / factorial(l + abs(m)))
        norm = norm * (-1.0)**(0.5 * (m - abs(m)))
        return norm * lpmv(abs(m), l, costheta) * np.exp(1j * m * phi)
    
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

    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _bispectrum_5d_njit(
            k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
            pdw1, pdw2, pdw3,
            b1, b2, g2, f, avir, sv, MB0, NP0,
            cnloB, cB1, cB2,
            inv_nbar, inv_qiso6, out):
        """Fused bias-weighted VDG bispectrum on the 5D quadrature grid. 
        """
        flat = out.reshape(-1)
        k1f = k1_p.reshape(-1)
        k2f = k2_p.reshape(-1)
        k3f = k3_p.reshape(-1)
        m1f = mu1_p.reshape(-1)
        m2f = mu2_p.reshape(-1)
        m3f = mu3_p.reshape(-1)
        p1f = pdw1.reshape(-1)
        p2f = pdw2.reshape(-1)
        p3f = pdw3.reshape(-1)
        b1f = b1.reshape(-1)
        b2f = b2.reshape(-1)
        g2f = g2.reshape(-1)
        ff_ = f.reshape(-1)
        avf = avir.reshape(-1)
        svf = sv.reshape(-1)
        mbf = MB0.reshape(-1)
        npf = NP0.reshape(-1)
        cnf = cnloB.reshape(-1)
        cB1f = cB1.reshape(-1)
        cB2f = cB2.reshape(-1)
        inbf = inv_nbar.reshape(-1)
        iqf = inv_qiso6.reshape(-1)
        nparams = b1f.size
        N = flat.size
        for q in nb.prange(N):
            p = q % nparams
            bb1 = b1f[p]; bb2 = b2f[p]; gg2 = g2f[p]
            ff = ff_[p]; av = avf[p]; sv_p = svf[p]
            mb0 = mbf[p]; np0 = npf[p]; cnl = cnf[p]
            ccB1 = cB1f[p]; ccB2 = cB2f[p]
            iqs = iqf[p]; inb = inbf[p]
            k1 = k1f[q]; k2 = k2f[q]; k3 = k3f[q]
            mu1 = m1f[q]; mu2 = m2f[q]; mu3 = m3f[q]
            pd1 = p1f[q]; pd2 = p2f[q]; pd3 = p3f[q]

            mu1_sq = mu1*mu1; mu2_sq = mu2*mu2; mu3_sq = mu3*mu3
            Z1_1 = bb1 + ff*mu1_sq - ccB1*(k1*k1)*mu1_sq - ccB2*(k1*k1)*mu1_sq*mu1_sq
            Z1_2 = bb1 + ff*mu2_sq - ccB1*(k2*k2)*mu2_sq - ccB2*(k2*k2)*mu2_sq*mu2_sq
            Z1_3 = bb1 + ff*mu3_sq - ccB1*(k3*k3)*mu3_sq - ccB2*(k3*k3)*mu3_sq*mu3_sq

            # tree leg 12 (i=1, j=2, k=3, muk = -mu3)
            muij = (k3*k3 - k1*k1 - k2*k2) / (2.0*k1*k2)
            muij2 = muij*muij
            ratio = 0.5*(k1/k2 + k2/k1)*muij
            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
            K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
            muk = -mu3
            Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k3*muk*((mu1/k1)*Z1_2 + (mu2/k2)*Z1_1)
            tree = 2.0*Z1_1*Z1_2*Z2 * pd1*pd2

            # tree leg 23
            muij = (k1*k1 - k2*k2 - k3*k3) / (2.0*k2*k3)
            muij2 = muij*muij
            ratio = 0.5*(k2/k3 + k3/k2)*muij
            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
            K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
            muk = -mu1
            Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k1*muk*((mu2/k2)*Z1_3 + (mu3/k3)*Z1_2)
            tree += 2.0*Z1_2*Z1_3*Z2 * pd2*pd3

            # tree leg 31
            muij = (k2*k2 - k3*k3 - k1*k1) / (2.0*k3*k1)
            muij2 = muij*muij
            ratio = 0.5*(k3/k1 + k1/k3)*muij
            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
            K2 = bb1*F2 + 0.5*bb2 + gg2*(muij2 - 1.0)
            muk = -mu2
            Z2 = K2 + ff*muk*muk*G2 + 0.5*ff*k2*muk*((mu3/k3)*Z1_1 + (mu1/k1)*Z1_3)
            tree += 2.0*Z1_3*Z1_1*Z2 * pd3*pd1

            kmu1_sq = k1*k1*mu1_sq
            kmu2_sq = k2*k2*mu2_sq
            kmu3_sq = k3*k3*mu3_sq
            lamb2 = -0.5*ff*ff*(kmu1_sq + kmu2_sq + kmu3_sq)
            denom = 1.0 - lamb2*av*av
            winfty = np.exp(lamb2*sv_p*sv_p/denom) / denom**1.5
            eft = 1.0 + cnl*ff*ff*(kmu1_sq + kmu2_sq + kmu3_sq)
            tree = tree * eft

            lam1 = -ff*ff*k1*k1*mu1_sq
            den1 = 1.0 - lam1*av*av
            st1 = np.exp(lam1*sv_p*sv_p/den1)/den1**1.5 * (bb1*mb0 + ff*np0*mu1_sq) * Z1_1
            st1 = st1 * (1.0 + cnl*ff*ff*kmu1_sq)
            lam2 = -ff*ff*k2*k2*mu2_sq
            den2 = 1.0 - lam2*av*av
            st2 = np.exp(lam2*sv_p*sv_p/den2)/den2**1.5 * (bb1*mb0 + ff*np0*mu2_sq) * Z1_2
            st2 = st2 * (1.0 + cnl*ff*ff*kmu2_sq)
            lam3 = -ff*ff*k3*k3*mu3_sq
            den3 = 1.0 - lam3*av*av
            st3 = np.exp(lam3*sv_p*sv_p/den3)/den3**1.5 * (bb1*mb0 + ff*np0*mu3_sq) * Z1_3
            st3 = st3 * (1.0 + cnl*ff*ff*kmu3_sq)
            stoch = (st1*pd1 + st2*pd2 + st3*pd3) * inb

            flat[q] = (tree*winfty + stoch) * iqs

    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _bispectrum_5d_diagrams_njit(
            k1_p, k2_p, k3_p, mu1_p, mu2_p, mu3_p,
            pdw1, pdw2, pdw3,
            f, avir, sv, cnloB,
            inv_qiso6, tree_col, stoch_col, out_spt, out_stoch):
        """Per-diagram, bias-stripped bispectrum on the 5D grid.

        ``tree_col`` (length 10) and ``stoch_col`` (length 3) map each
        diagram index to its output column in ``out_spt`` / ``out_stoch``;
        a value of ``-1`` means the diagram is not requested and is
        skipped (no work, no write).
        """
        k1f = k1_p.reshape(-1)
        k2f = k2_p.reshape(-1)
        k3f = k3_p.reshape(-1)
        m1f = mu1_p.reshape(-1)
        m2f = mu2_p.reshape(-1)
        m3f = mu3_p.reshape(-1)
        p1f = pdw1.reshape(-1)
        p2f = pdw2.reshape(-1)
        p3f = pdw3.reshape(-1)
        ff_ = f.reshape(-1)
        avf = avir.reshape(-1)
        svf = sv.reshape(-1)
        cnf = cnloB.reshape(-1)
        iqf = inv_qiso6.reshape(-1)
        nparams = ff_.size
        N = out_spt.shape[0]

        c0 = tree_col[0]; c1 = tree_col[1]; c2 = tree_col[2]
        c3 = tree_col[3]; c4 = tree_col[4]; c5 = tree_col[5]
        c6 = tree_col[6]; c7 = tree_col[7]; c8 = tree_col[8]
        c9 = tree_col[9]
        cs0 = stoch_col[0]; cs1 = stoch_col[1]; cs2 = stoch_col[2]
        any_tree = (c0 >= 0 or c1 >= 0 or c2 >= 0 or c3 >= 0 or c4 >= 0
                    or c5 >= 0 or c6 >= 0 or c7 >= 0 or c8 >= 0 or c9 >= 0)
        any_stoch = (cs0 >= 0 or cs1 >= 0 or cs2 >= 0)

        for q in nb.prange(N):
            p = q % nparams
            ff = ff_[p]; av = avf[p]; sv_p = svf[p]; cnl = cnf[p]
            iqs = iqf[p]
            k1 = k1f[q]; k2 = k2f[q]; k3 = k3f[q]
            mu1 = m1f[q]; mu2 = m2f[q]; mu3 = m3f[q]
            pd1 = p1f[q]; pd2 = p2f[q]; pd3 = p3f[q]

            f2 = ff*ff
            f3 = f2*ff
            mu1_sq = mu1*mu1
            mu2_sq = mu2*mu2
            mu3_sq = mu3*mu3
            kmu1_sq = k1*k1*mu1_sq
            kmu2_sq = k2*k2*mu2_sq
            kmu3_sq = k3*k3*mu3_sq

            if any_tree:
                pp12 = pd1*pd2
                pp23 = pd2*pd3
                pp31 = pd3*pd1

                lamb2 = -0.5*f2*(kmu1_sq + kmu2_sq + kmu3_sq)
                denom = 1.0 - lamb2*av*av
                winfty = np.exp(lamb2*sv_p*sv_p/denom) / denom**1.5
                eft = 1.0 + cnl*(kmu1_sq + kmu2_sq + kmu3_sq)
                winfac = winfty * iqs * eft

                spt0 = 0.0; spt1 = 0.0; spt2 = 0.0
                spt3 = 0.0; spt4 = 0.0; spt5 = 0.0
                spt6 = 0.0; spt7 = 0.0; spt8 = 0.0; spt9 = 0.0

                for leg in range(3):
                    if leg == 0:
                        ki = k1; kj = k2; kk = k3
                        mui = mu1; muj = mu2; muk = mu3
                        mui_sq = mu1_sq; muj_sq = mu2_sq; muk_sq = mu3_sq
                        pp = pp12
                    elif leg == 1:
                        ki = k2; kj = k3; kk = k1
                        mui = mu2; muj = mu3; muk = mu1
                        mui_sq = mu2_sq; muj_sq = mu3_sq; muk_sq = mu1_sq
                        pp = pp23
                    else:
                        ki = k3; kj = k1; kk = k2
                        mui = mu3; muj = mu1; muk = mu2
                        mui_sq = mu3_sq; muj_sq = mu1_sq; muk_sq = mu2_sq
                        pp = pp31

                    muij = (kk*kk - ki*ki - kj*kj) / (2.0*ki*kj)
                    muij2 = muij*muij
                    s_ij = mui_sq + muj_sq
                    p_ij = mui_sq * muj_sq
                    fkm = ff*kk*muk
                    a_i = fkm * mui/ki
                    a_j = fkm * muj/kj

                    need_F2 = (c0 >= 0 or c1 >= 0 or c2 >= 0)
                    need_G2 = (c1 >= 0 or c2 >= 0 or c9 >= 0)
                    need_Kmu = (c6 >= 0 or c7 >= 0 or c8 >= 0)
                    F2 = 0.0; G2 = 0.0; Kmu = 0.0
                    if need_F2 or need_G2:
                        ratio = 0.5*(ki/kj + kj/ki)*muij
                        if need_F2:
                            F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
                        if need_G2:
                            G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
                    if need_Kmu:
                        Kmu = muij2 - 1.0

                    if c0 >= 0:
                        spt0 += (2.0*F2 - a_i - a_j) * pp
                    if c1 >= 0:
                        spt1 += (2.0*F2*ff*s_ij + 2.0*G2*ff*muk_sq
                                 - a_i*ff*(mui_sq + 2.0*muj_sq)
                                 - a_j*ff*(2.0*mui_sq + muj_sq)) * pp
                    if c2 >= 0:
                        spt2 += (2.0*F2*f2*p_ij + 2.0*G2*f2*muk_sq*s_ij
                                 - a_i*f2*(2.0*p_ij + muj_sq*muj_sq)
                                 - a_j*f2*(mui_sq*mui_sq + 2.0*p_ij)) * pp
                    if c3 >= 0:
                        spt3 += pp
                    if c4 >= 0:
                        spt4 += ff*s_ij * pp
                    if c5 >= 0:
                        spt5 += f2*p_ij * pp
                    if c6 >= 0:
                        spt6 += 2.0*Kmu * pp
                    if c7 >= 0:
                        spt7 += 2.0*Kmu*ff*s_ij * pp
                    if c8 >= 0:
                        spt8 += 2.0*Kmu*f2*p_ij * pp
                    if c9 >= 0:
                        spt9 += (2.0*G2*f3*muk_sq*p_ij
                                 - a_i*f3*mui_sq*muj_sq*muj_sq
                                 - a_j*f3*muj_sq*mui_sq*mui_sq) * pp

                if c0 >= 0: out_spt[q, c0] = spt0 * winfac
                if c1 >= 0: out_spt[q, c1] = spt1 * winfac
                if c2 >= 0: out_spt[q, c2] = spt2 * winfac
                if c3 >= 0: out_spt[q, c3] = spt3 * winfac
                if c4 >= 0: out_spt[q, c4] = spt4 * winfac
                if c5 >= 0: out_spt[q, c5] = spt5 * winfac
                if c6 >= 0: out_spt[q, c6] = spt6 * winfac
                if c7 >= 0: out_spt[q, c7] = spt7 * winfac
                if c8 >= 0: out_spt[q, c8] = spt8 * winfac
                if c9 >= 0: out_spt[q, c9] = spt9 * winfac

            if any_stoch:
                if cs0 >= 0 or cs1 >= 0 or cs2 >= 0:
                    lam1 = -f2*kmu1_sq
                    den1 = 1.0 - lam1*av*av
                    W1 = np.exp(lam1*sv_p*sv_p/den1)/den1**1.5
                    lam2 = -f2*kmu2_sq
                    den2 = 1.0 - lam2*av*av
                    W2 = np.exp(lam2*sv_p*sv_p/den2)/den2**1.5
                    lam3 = -f2*kmu3_sq
                    den3 = 1.0 - lam3*av*av
                    W3 = np.exp(lam3*sv_p*sv_p/den3)/den3**1.5

                if cs0 >= 0:
                    out_stoch[q, cs0] = (W1*pd1 + W2*pd2 + W3*pd3) * iqs
                if cs1 >= 0:
                    out_stoch[q, cs1] = (ff*mu1_sq*W1*pd1
                                         + ff*mu2_sq*W2*pd2
                                         + ff*mu3_sq*W3*pd3) * iqs
                if cs2 >= 0:
                    out_stoch[q, cs2] = (f2*mu1_sq*mu1_sq*W1*pd1
                                         + f2*mu2_sq*mu2_sq*W2*pd2
                                         + f2*mu3_sq*mu3_sq*W3*pd3) * iqs

    if HAS_JAX:
        @staticmethod
        def _bispectrum_5d_jax_fused(
                k1, k2, k3, mu1, mu2, mu3,
                pdw1, pdw2, pdw3,
                b1, b2, g2, f, avir, sv, MB0, NP0,
                cnloB, cB1, cB2,
                inv_nbar, inv_qiso6):
            """JAX-traced fused bispectrum on the broadcast grid.
            """
            mu1_sq = mu1*mu1
            mu2_sq = mu2*mu2
            mu3_sq = mu3*mu3
            Z1_1 = b1 + f*mu1_sq - cB1*(k1*k1)*mu1_sq - cB2*(k1*k1)*mu1_sq*mu1_sq
            Z1_2 = b1 + f*mu2_sq - cB1*(k2*k2)*mu2_sq - cB2*(k2*k2)*mu2_sq*mu2_sq
            Z1_3 = b1 + f*mu3_sq - cB1*(k3*k3)*mu3_sq - cB2*(k3*k3)*mu3_sq*mu3_sq

            def _leg(ki, kj, kk, mui, muj, muk, pp, Z1_i, Z1_j):
                muij = (kk*kk - ki*ki - kj*kj) / (2.0*ki*kj)
                muij2 = muij*muij
                ratio = 0.5*(ki/kj + kj/ki)*muij
                F2 = 5.0/7.0 + 2.0/7.0*muij2 + ratio
                G2 = 3.0/7.0 + 4.0/7.0*muij2 + ratio
                K2 = b1*F2 + 0.5*b2 + g2*(muij2 - 1.0)
                muk_neg = -muk
                Z2 = (K2 + f*muk_neg*muk_neg*G2
                      + 0.5*f*kk*muk_neg*((mui/ki)*Z1_j + (muj/kj)*Z1_i))
                return 2.0*Z1_i*Z1_j*Z2 * pp

            tree = (_leg(k1, k2, k3, mu1, mu2, mu3, pdw1*pdw2, Z1_1, Z1_2)
                    + _leg(k2, k3, k1, mu2, mu3, mu1, pdw2*pdw3, Z1_2, Z1_3)
                    + _leg(k3, k1, k2, mu3, mu1, mu2, pdw3*pdw1, Z1_3, Z1_1))

            kmu1_sq = k1*k1*mu1_sq
            kmu2_sq = k2*k2*mu2_sq
            kmu3_sq = k3*k3*mu3_sq
            lamb2 = -0.5*f*f*(kmu1_sq + kmu2_sq + kmu3_sq)
            denom = 1.0 - lamb2*avir*avir
            winfty = jnp.exp(lamb2*sv*sv/denom) / denom**1.5
            eft = 1.0 + cnloB*f*f*(kmu1_sq + kmu2_sq + kmu3_sq)

            def _stoch_leg(ki, mui, mui_sq, Z1_i, pdw_i):
                lam = -f*f*ki*ki*mui_sq
                den = 1.0 - lam*avir*avir
                W = jnp.exp(lam*sv*sv/den) / den**1.5
                return W * (b1*MB0 + f*NP0*mui_sq) * Z1_i * pdw_i * (1.0 + cnloB*f*f*ki*ki*mui_sq)

            stoch = (_stoch_leg(k1, mu1, mu1_sq, Z1_1, pdw1)
                     + _stoch_leg(k2, mu2, mu2_sq, Z1_2, pdw2)
                     + _stoch_leg(k3, mu3, mu3_sq, Z1_3, pdw3)) * inv_nbar
            return (tree*eft*winfty + stoch) * inv_qiso6

        _bispectrum_5d_jax_fused = staticmethod(
            jax.jit(_bispectrum_5d_jax_fused.__func__))

        @staticmethod
        def _bispectrum_5d_jax_diagrams(
                k1, k2, k3, mu1, mu2, mu3,
                pdw1, pdw2, pdw3,
                f, avir, sv, cnloB, inv_qiso6,
                tree_keep, stoch_keep):
            """JAX-traced per-diagram bispectrum on the broadcast grid.

            ``tree_keep`` and ``stoch_keep`` are static tuples of diagram
            indices (subsets of ``range(10)`` and ``range(3)``) to compute.
            Only those diagrams are evaluated and stacked; JAX JIT
            specializes per (tree_keep, stoch_keep) combination.
            """
            mu1_sq = mu1*mu1
            mu2_sq = mu2*mu2
            mu3_sq = mu3*mu3
            f2 = f*f
            f3 = f2*f

            kmu1_sq = k1*k1*mu1_sq
            kmu2_sq = k2*k2*mu2_sq
            kmu3_sq = k3*k3*mu3_sq

            tree_set = set(tree_keep)

            if tree_set:
                lamb2 = -0.5*f2*(kmu1_sq + kmu2_sq + kmu3_sq)
                denom = 1.0 - lamb2*avir*avir
                winfty = jnp.exp(lamb2*sv*sv/denom) / denom**1.5
                eft = 1.0 + cnloB*(kmu1_sq + kmu2_sq + kmu3_sq)
                winfac = winfty * inv_qiso6 * eft

                need_F2 = bool(tree_set & {0, 1, 2})
                need_G2 = bool(tree_set & {1, 2, 9})
                need_Kmu = bool(tree_set & {6, 7, 8})

                def _leg(ki, kj, kk, mui, muj, muk,
                         mui_sq, muj_sq, muk_sq, pp):
                    muij = (kk*kk - ki*ki - kj*kj) / (2.0*ki*kj)
                    muij2 = muij*muij
                    s_ij = mui_sq + muj_sq
                    p_ij = mui_sq * muj_sq
                    fkm = f*kk*muk
                    a_i = fkm * mui/ki
                    a_j = fkm * muj/kj

                    ratio = (0.5*(ki/kj + kj/ki)*muij
                             if (need_F2 or need_G2) else 0.0)
                    F2 = (5.0/7.0 + 2.0/7.0*muij2 + ratio) if need_F2 else 0.0
                    G2 = (3.0/7.0 + 4.0/7.0*muij2 + ratio) if need_G2 else 0.0
                    Kmu = (muij2 - 1.0) if need_Kmu else 0.0

                    out = {}
                    if 0 in tree_set:
                        out[0] = (2.0*F2 - a_i - a_j) * pp
                    if 1 in tree_set:
                        out[1] = (2.0*F2*f*s_ij + 2.0*G2*f*muk_sq
                                  - a_i*f*(mui_sq + 2.0*muj_sq)
                                  - a_j*f*(2.0*mui_sq + muj_sq)) * pp
                    if 2 in tree_set:
                        out[2] = (2.0*F2*f2*p_ij + 2.0*G2*f2*muk_sq*s_ij
                                  - a_i*f2*(2.0*p_ij + muj_sq*muj_sq)
                                  - a_j*f2*(mui_sq*mui_sq + 2.0*p_ij)) * pp
                    if 3 in tree_set:
                        out[3] = pp
                    if 4 in tree_set:
                        out[4] = f*s_ij * pp
                    if 5 in tree_set:
                        out[5] = f2*p_ij * pp
                    if 6 in tree_set:
                        out[6] = 2.0*Kmu * pp
                    if 7 in tree_set:
                        out[7] = 2.0*Kmu*f*s_ij * pp
                    if 8 in tree_set:
                        out[8] = 2.0*Kmu*f2*p_ij * pp
                    if 9 in tree_set:
                        out[9] = (2.0*G2*f3*muk_sq*p_ij
                                  - a_i*f3*mui_sq*muj_sq*muj_sq
                                  - a_j*f3*muj_sq*mui_sq*mui_sq) * pp
                    return out

                l12 = _leg(k1, k2, k3, mu1, mu2, mu3,
                           mu1_sq, mu2_sq, mu3_sq, pdw1*pdw2)
                l23 = _leg(k2, k3, k1, mu2, mu3, mu1,
                           mu2_sq, mu3_sq, mu1_sq, pdw2*pdw3)
                l31 = _leg(k3, k1, k2, mu3, mu1, mu2,
                           mu3_sq, mu1_sq, mu2_sq, pdw3*pdw1)
                spt_list = [(l12[d] + l23[d] + l31[d]) * winfac
                            for d in tree_keep]
                spt_stack = jnp.stack(spt_list, axis=-1)
            else:
                spt_shape = (k1*mu1*pdw1*f*avir*sv*cnloB*inv_qiso6).shape
                spt_stack = jnp.zeros(spt_shape + (0,))

            stoch_set = set(stoch_keep)
            if stoch_set:
                def _stoch_W(ki, mui_sq):
                    lam = -f2*ki*ki*mui_sq
                    den = 1.0 - lam*avir*avir
                    return jnp.exp(lam*sv*sv/den) / den**1.5
                W1 = _stoch_W(k1, mu1_sq)
                W2 = _stoch_W(k2, mu2_sq)
                W3 = _stoch_W(k3, mu3_sq)
                stoch_terms = {}
                if 0 in stoch_set:
                    stoch_terms[0] = (W1*pdw1 + W2*pdw2 + W3*pdw3) * inv_qiso6
                if 1 in stoch_set:
                    stoch_terms[1] = (f*mu1_sq*W1*pdw1
                                      + f*mu2_sq*W2*pdw2
                                      + f*mu3_sq*W3*pdw3) * inv_qiso6
                if 2 in stoch_set:
                    stoch_terms[2] = (f2*mu1_sq*mu1_sq*W1*pdw1
                                      + f2*mu2_sq*mu2_sq*W2*pdw2
                                      + f2*mu3_sq*mu3_sq*W3*pdw3) * inv_qiso6
                stoch_stack = jnp.stack(
                    [stoch_terms[d] for d in stoch_keep], axis=-1)
            else:
                stoch_shape = (k1*mu1*pdw1*f*avir*sv*inv_qiso6).shape
                stoch_stack = jnp.zeros(stoch_shape + (0,))

            return spt_stack, stoch_stack

        _bispectrum_5d_jax_diagrams = staticmethod(
            jax.jit(_bispectrum_5d_jax_diagrams.__func__,
                    static_argnums=(14, 15)))

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
        elif mu12_transform == 'k3':
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

    def _sugi_get_proj_ops(self, ell, nmu1, nmu12, nphi, mu12_transform):
        """Compute and cache the (l1, l2, L) Sugiyama projection operators.

        Each entry maps an `(l1, l2, L)` tuple to a 1D array of length
        nmu1*nmu12*nphi, containing the real part of the projection kernel
        multiplied by integration weights. Inner product with the raveled
        5D bispectrum yields the multipole.
        """
        ell_key = tuple(sorted(set(tuple(int(x) for x in ll) for ll in ell)))
        key = (nmu1, nmu12, nphi, mu12_transform, ell_key)
        cached = self._sugi_proj_cache.get(key)
        if cached is not None:
            return cached

        try:
            from sympy.physics.wigner import wigner_3j
        except ImportError as exc:
            raise ImportError(
                "Sugiyama multipoles require `sympy` for Wigner-3j "
                "coefficients. Install it via `pip install sympy`."
            ) from exc

        mu1_g, w_mu1_g, mu12_g, w_mu12_g, _, phi_g, w_phi_g = \
            self._sugi_get_quadrature(nmu1, nmu12, nphi, mu12_transform)
        weights = (w_mu1_g * w_mu12_g * w_phi_g).squeeze()

        proj = {}
        for ll in ell_key:
            l1, l2, L = ll
            h = float(wigner_3j(l1, l2, L, 0, 0, 0).evalf())
            if h == 0.0:
                proj[ll] = np.zeros(nmu1 * nmu12 * nphi)
                continue
            op = np.zeros((1, nmu1, nmu12, nphi), dtype=complex)
            for M in range(-L, L+1):
                w3j = float(wigner_3j(l1, l2, L, 0, -M, M).evalf())
                if w3j == 0.0:
                    continue
                y1 = self._sph_harm(l2, -M, mu12_g, 0.0)
                y2 = self._sph_harm(L, M, mu1_g, -phi_g)
                op = op + w3j * y1 * y2
            op = op.squeeze() * weights
            prefactor = h * (2*l1 + 1) * (2*l2 + 1) * (2*L + 1) / (8.0*np.pi)
            proj[ll] = np.real(op.ravel() * prefactor)

        self._sugi_proj_cache[key] = proj
        return proj

    def _sugi_get_proj_ops_k3(self, ell, pair, nmu1, nmu12, nphi,
                              mu12_transform='k3'):
        """Per-pair projection operators for the per-pair-k3 quadratures.

        Each entry maps an `(l1, l2, L)` tuple to an array of shape
        `(n_pair, nmu1*nmu12*nphi)`. The per-pair Jacobian ``dmu12/dt``
        is folded into the operator together with the underlying
        quadrature weights.

        Result is cached on the pair contents and shape so that repeated
        calls with the same ``(pair, nmu1, nmu12, nphi, mu12_transform,
        ell)`` reuse the operator. 
        """
        pair = np.atleast_2d(pair)
        ell_key = tuple(sorted(set(tuple(int(x) for x in ll) for ll in ell)))
        cache_key = (pair.shape, pair.tobytes(),
                     nmu1, nmu12, nphi, mu12_transform, ell_key)
        cached = self._sugi_proj_cache_k3.get(cache_key)
        if cached is not None:
            return cached

        try:
            from sympy.physics.wigner import wigner_3j
        except ImportError as exc:
            raise ImportError(
                "Sugiyama multipoles require `sympy` for Wigner-3j "
                "coefficients. Install it via `pip install sympy`."
            ) from exc

        mu1_g, w_mu1_g, t_g, w_t_g, _, phi_g, w_phi_g = \
            self._sugi_get_quadrature(nmu1, nmu12, nphi, mu12_transform)

        n_pair = pair.shape[0]
        k1 = pair[:, 0][:, None, None, None]
        k2 = pair[:, 1][:, None, None, None]
        k_lo = np.abs(k1 - k2)
        k_hi = k1 + k2
        # `t_g` has shape (1, 1, nmu12, 1) from `_sugi_get_quadrature`.
        k3_pair, dk3_dt = self._k3_from_t(t_g, k_lo, k_hi)
        k3_pair = np.maximum(k3_pair, 1e-30)
        mu12_pair = np.clip(
            (k3_pair*k3_pair - k1*k1 - k2*k2) / (2.0 * k1 * k2), -1.0, 1.0)
        # dmu12/dt = (dk3/dt) * dmu12/dk3 = (dk3/dt) * k3/(k1 k2)
        jac = dk3_dt * k3_pair / (k1 * k2)
        weight_pair = (w_mu1_g * w_t_g * w_phi_g * jac).reshape(
            n_pair, nmu1*nmu12*nphi)

        proj = {}
        for ll in ell_key:
            l1, l2, L = ll
            h = float(wigner_3j(l1, l2, L, 0, 0, 0).evalf())
            if h == 0.0:
                proj[ll] = np.zeros((n_pair, nmu1*nmu12*nphi))
                continue
            op = np.zeros((n_pair, nmu1, nmu12, nphi), dtype=complex)
            for M in range(-L, L+1):
                w3j = float(wigner_3j(l1, l2, L, 0, -M, M).evalf())
                if w3j == 0.0:
                    continue
                y1 = self._sph_harm(l2, -M, mu12_pair, 0.0)
                y2 = self._sph_harm(L, M, mu1_g, -phi_g)
                op = op + w3j * y1 * y2
            prefactor = h * (2*l1 + 1) * (2*l2 + 1) * (2*L + 1) / (8.0*np.pi)
            proj[ll] = np.real(op.reshape(n_pair, -1) * weight_pair
                               * prefactor)
        self._sugi_proj_cache_k3[cache_key] = proj
        return proj
    
    def clear_caches(self):
        """Clear all internal caches."""
        self._sugi_quad_cache.clear()
        self._sugi_proj_cache.clear()
        self._sugi_proj_cache_k3.clear()
        self._scocc_quad_cache.clear()
        self._scocc_proj_cache.clear()

    @staticmethod
    def _kgrid_compression(kmin, kmax, nk=100):
        kcenter, power = 0.65, 1.5
        croot = lambda x: np.sign(x) * np.abs(x)**(1.0 / power)
        qmin = croot(np.log10(kmin) + kcenter, )
        qmax = croot(np.log10(kmax) + kcenter, )
        s = qmin + (qmax - qmin) * np.linspace(0.0, 1.0, nk)
        return 10.0**(np.sign(s) * np.abs(s)**power - kcenter)

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