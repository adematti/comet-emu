"""Main PTEmu module."""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.integrate import quad_vec
from scipy.integrate import quad,dblquad
from scipy.special import eval_legendre
from astropy.io import fits
import pickle
from comet.cosmology import Cosmology
from comet.data import MeasuredData
from comet.tables import Tables
from comet.bispectrum import Bispectrum
import os

base_dir = os.path.join(os.path.dirname(__file__))


class PTEmu:
    r"""Main class for the emulator of the power spectrum multipoles.

    The emulator makes use of evolution mapping (`Sanchez 2020
    <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511>`_,
    `Sanchez et al 2021 <https://arxiv.org/abs/2108.12710>`_,) to compress
    the information of evolution parameters :math:`\mathbf{\Theta_{e}}`
    (e.g. :math:`h,\,\Omega_\mathrm{K},\,w_0,\,w_\mathrm{a},\,A_\mathrm{s},\,
    \ldots`) into the single quantity :math:`\sigma_{12}`, defined as the rms
    fluctuation of the linear density contrast :math:`\delta` within spheres
    of radius :math:`R=8\,\mathrm{Mpc}`.

    This parameter, together with the parameters affecting the shape of the
    power spectrum :math:`\mathbf{\Theta_{s}}` (e.g.
    :math:`\omega_\mathrm{b},\,\omega_\mathrm{c},\,n_\mathrm{s}`), and
    the linear growth rate :math:`f`, are used as base of the emulator.

    The redshift-dependency of the multipoles can also be treated similarly to
    the impact that different evolution parameters have on the power spectrum,
    that is, by a simple rescaling of the amplitude of the power spectrum in
    order to match the desired value of :math:`\sigma_{12}`.

    Internally to the emulator, the pair :math:`\left[k,P(k)\right]` is
    expressed in :math:`\left[\mathrm{Mpc}^{-1},\mathrm{Mpc}^3\right]` units,
    since this is the only set of units for which the evolution parameter
    degeneracy is present. If the user wishes to use the more conventional
    unit set :math:`\left[h\,\mathrm{Mpc}^{-1},h^{-3}\,\mathrm{Mpc}^3\right]`,
    they can do so by specifying it in the proper class attribute flag. In this
    case, the input/output are converted into :math:`\mathrm{Mpc}` units
    before being used/returned.

    Geometrical distortions (AP corrections) are included a posteriori without
    the need of including them in the emulation. This process is carried out
    by first reconstructing the full anisotropic 2d galaxy power spectrum
    :math:`P_\mathrm{gg}(k,\mu)`, summing up all the even multipoles up to
    :math:`\ell=6`, applying distortions to :math:`k` and :math:`\mu`, and then
    projecting again over the Legendre polynomials.
    """

    def __init__(self, model, use_Mpc=True, bias_basis='EggScoSmi'):
        r"""Class constructor.

        Parameters
        ----------
        model: str
            Identifier of the selected model.
        use_Mpc: bool, optional
            Flag that determines if the input and output quantities are
            specified in :math:`\mathrm{Mpc}` (**True**) or
            :math:`h^{-1}\mathrm{Mpc}` (**False**) units. Defaults to **True**.
        """
        self.bias_basis = bias_basis

        if self.bias_basis == 'EggScoSmi':
            self.bias_params_list = ['b1', 'b2', 'g2', 'g21', 'c0', 'c2', 'c4',
                                     'cnlo', 'NP0', 'NP20', 'NP22', 'NB0',
                                     'MB0']
        elif self.bias_basis == 'AssBauGre':
            self.bias_params_list = ['b1', 'b2', 'bG2', 'bGam3', 'c0', 'c2',
                                     'c4', 'cnlo', 'NP0', 'NP20', 'NP22',
                                     'NB0', 'MB0']
        elif self.bias_basis == 'AmiGleKok':
            self.bias_params_list = ['b1t', 'b2t', 'b3t', 'b4t', 'c0', 'c2',
                                     'c4', 'cnlo', 'NP0', 'NP20', 'NP22',
                                     'NB0', 'MB0']
        else:
            print('Warning. Bias basis not recognised, defaulting to '
                  '"EggScoSmi".')

        self.RSD_params_list = []
        self.de_model_params_list = {
            'lambda': ['h', 'As', 'Ok', 'z'],
            'w0': ['h', 'As', 'Ok', 'w0', 'z'],
            'w0wa': ['h', 'As', 'Ok', 'w0', 'wa', 'z']}

        self.n_diagrams = 19
        self.diagrams_emulated = ['P0L_b1b1', 'PNL_b1', 'PNL_id',
                                  'Pctr_c0', 'Pctr_c2', 'Pctr_c4',
                                  'Pctr_b1b1cnlo', 'Pctr_b1cnlo', 'Pctr_cnlo',
                                  'P1L_b1b1', 'P1L_b1b2', 'P1L_b1g2',
                                  'P1L_b1g21', 'P1L_b2b2', 'P1L_b2g2',
                                  'P1L_g2g2', 'P1L_b2', 'P1L_g2', 'P1L_g21']
        self.diagrams_all = ['P0L_b1b1', 'PNL_b1', 'PNL_id', 'Pctr_c0',
                             'Pctr_c2', 'Pctr_c4', 'Pctr_b1b1cnlo',
                             'Pctr_b1cnlo', 'Pctr_cnlo', 'P1L_b1b1',
                             'P1L_b1b2', 'P1L_b1g2', 'P1L_b1g21', 'P1L_b2b2',
                             'P1L_b2g2', 'P1L_g2g2', 'P1L_b2', 'P1L_g2',
                             'P1L_g21', 'Pnoise_NP0', 'Pnoise_NP20',
                             'Pnoise_NP22']

        self.use_Mpc = use_Mpc
        self.nbar = 1.0  # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc

        self.training = {}

        self.emu = {}
        self.cosmo = Cosmology(0.3, 67.0)  # Initialise with arbitrary values

        self.Pk_lin = None
        self.Pk_ratios = {0: None, 2: None, 4: None}

        self.Pell_spline = {}
        self.Pell_min = {}
        self.Pell_max = {}
        self.neff_min = {}
        self.neff_max = {}

        self.PX_ell_spline = {X: {} for X in self.diagrams_all}
        self.PX_ell_min = {X: {} for X in self.diagrams_all}
        self.PX_ell_max = {X: {} for X in self.diagrams_all}
        self.X_neff_min = {X: {} for X in self.diagrams_all}
        self.X_neff_max = {X: {} for X in self.diagrams_all}
        self.PX_ell_list = {}

        self.k_table_min = {}
        self.k_table_max = {}

        self.data = {}

        self.splines_up_to_date = False
        self.dw_spline_up_to_date = False
        self.X_splines_up_to_date = {X: False for X in self.diagrams_all}
        self.emu_params_updated = False

        self.chi2_decomposition = None

        try:
            self.load_emulator_data(
                fname=base_dir+'/data_dir/tables/{}.fits'.format(model))
        except Exception:
            print('Table file for this model not found. Initialise '
                  'with `load_emulator_data`')
        try:
            self.load_emulator(
                fname_base=base_dir+'/data_dir/models/{}'.format(model))
        except Exception:
            print('Emulator files for this model not found. Initialise with '
                  '`load_emulator`, or train the emulator first, '
                  'if necessary.')

    def init_params_dict(self):
        r"""Initialize params dictionary.

        Sets up the internal class attribute which stores the complete list
        of model parameters. This includes cosmological parameters as well as
        biases, noises, counterterms, and other nuisance parameters.
        """
        self.params = {p: 0.0 for p in self.params_list +
                       self.bias_params_list +
                       self.de_model_params_list['w0wa']}
        self.params['w0'] = -1.0
        self.params['q_tr'] = 1.0
        self.params['q_lo'] = 1.0

    def load_emulator_data(self, fname):
        r"""Load tables of the emulator.

        Loads a fits file, reads the tables and stores them as class
        attributes, as instances of the **Tables** class. Additionally sets up
        the internal dictionary that stores the full list of model parameters,
        by calling **init_params_dict**. Determine if the emulator is for real-
        or redshift-space, checking if the growth rate :math:`f` is part of the
        parameter sample or not.

        Parameters
        ----------
        fname: str
            Name of the output fits file to read from.
        """
        hdul = fits.open(fname)

        self.params_shape_list = ([
            hdul['PARAMS_SHAPE'].header['TTYPE{}'.format(n+1)]
            for n in range(hdul['PARAMS_SHAPE'].header['TFIELDS'])])
        self.params_list = ([
            hdul['PARAMS_FULL'].header['TTYPE{}'.format(n+1)]
            for n in range(hdul['PARAMS_FULL'].header['TFIELDS'])])
        self.real_space = False if 'f' in self.params_list else True
        self.emu_LCDM_params = ({
            p: hdul['PRIMARY'].header['TRAINING:{}'.format(p)]
            for p in ['wc', 'wb', 'ns', 'h', 'As', 'z']})
        self.params_shape_ranges = {}
        self.params_ranges = {}
        for p in self.params_shape_list:
            min = hdul['PARAMS_SHAPE'].header['MIN:{}'.format(p)]
            max = hdul['PARAMS_SHAPE'].header['MAX:{}'.format(p)]
            self.params_shape_ranges[p] = [min, max]
        for p in self.params_list:
            min = hdul['PARAMS_FULL'].header['MIN:{}'.format(p)]
            max = hdul['PARAMS_FULL'].header['MAX:{}'.format(p)]
            self.params_ranges[p] = [min, max]
        self.init_params_dict()

        self.training['SHAPE'] = Tables(self.params_shape_list)
        self.training['FULL'] = Tables(self.params_list)

        self.k_table = hdul['K_TABLE'].data['bins']
        self.nk = self.k_table.shape[0]
        self.nkloop = sum(self.k_table > hdul['K_TABLE'].header['k1loop'])
        self.RSD_model = hdul['MODEL_FULL'].header['RSD_model']

        if self.RSD_model == 'VDG_infty':
            self.RSD_params_list.append('avir')
            self.params['avir'] = 0.0

        self.training['SHAPE'].assign_samples(hdul['PARAMS_SHAPE'])
        self.training['SHAPE'].assign_table(hdul['MODEL_SHAPE'],
                                            self.nk, self.nkloop)
        self.training['FULL'].assign_samples(hdul['PARAMS_FULL'])
        self.training['FULL'].assign_table(hdul['MODEL_FULL'],
                                           self.nk, self.nkloop)
        if not self.real_space:
            self.s12_for_P6 = hdul['MODEL_Pell6'].header['SIG12']
            self.P6 = hdul['MODEL_Pell6'].data['P_all']

        self.Bisp = Bispectrum(self.real_space, self.use_Mpc)

    def load_emulator(self, fname_base, data_type=None):
        r"""Load the emulator from pickle file.

        Loads an emulator object from a file (pickle format) and adds it to the
        internal dictionary containing the emulators.

        Parameters
        ----------
        fname_base: str
            Root name of the input pickle file.
        data_type: str, optional
            Type of the table which refers to the input emulator. If **None**,
            it loads the emulators for all the tables that are stored as class
            attributes. Defaults to **None**.
        """
        if data_type is None:
            ell_train = [0, 2, 4] if not self.real_space else [0]
            for dt in ['PL', 's12']:
                self.emu[dt] = pickle.load(
                    open('{}_{}.pickle'.format(fname_base, dt), "rb"))
            if self.RSD_model == 'VDG_infty':
                self.emu['sv'] = pickle.load(
                    open('{}_{}.pickle'.format(fname_base, 'sv'), "rb"))
            for ell in ell_train:
                self.emu[ell] = pickle.load(
                    open('{}_ratios_ell{}.pickle'.format(fname_base, ell),
                         "rb"))
        else:
            data_type = [data_type] if not isinstance(data_type, list) \
                else data_type
            for dt in data_type:
                if dt in ['PL', 's12', 'sv']:
                    self.emu[dt] = pickle.load(
                        open('{}_{}.pickle'.format(fname_base, dt), "rb"))
                else:
                    self.emu[dt] = pickle.load(
                        open('{}_ratios_ell{}.pickle'.format(fname_base, dt),
                             "rb"))

    def define_units(self, use_Mpc):
        r"""Define units for the power spectrum and number density.

        Sets the internal class attribute **use_Mpc**, clears all the data
        objects (if defined), and resets the number density to 1 in the units
        corresponding to the input flag. The number density value can be
        subsequently explicitly changed calling **define_nbar**.

        Parameters
        ----------
        use_Mpc: bool
            Flag that determines if the input and output quantities are
            specified in :math:`\mathrm{Mpc}` (**True**) or
            :math:`h^{-1}\,\mathrm{Mpc}` (**False**) units.
        """
        if use_Mpc != self.use_Mpc:
            self.use_Mpc = use_Mpc
            self.nbar = 1.0  # units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc
            for obs_id in self.data.keys():
                self.data[obs_id].clear_data()
            self.splines_up_to_date = False
            self.dw_spline_up_to_date = False
            self.Bisp.define_units(self.use_Mpc)
            self.Bisp.define_nbar(self.nbar)
            nbar_unit = '(1/Mpc)^3' if self.use_Mpc else '(h/Mpc)^3'
            print("Number density resetted to nbar = 1 {}. Data set (if "
                  "defined) cleared.".format(nbar_unit))

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
        self.Bisp.define_nbar(self.nbar)
        self.splines_up_to_date = False

    def define_data_set(self, obs_id, **kwargs):
        r"""Define data sample.

        If the identifier of the data sample is not present in the internal
        data dictionary, it assigns a new **MeasuredData** object to it.
        Otherwise it updates the already existing entry.

        Parameters
        ----------
        obs_id: str
            Identifier of the data sample.
        **kwargs: dict
            Dictionary of keyword arguments (check docs of **MeasuredData**
            class for the list of allowed keyword arguments).
        """
        if obs_id not in self.data.keys():
            self.data[obs_id] = MeasuredData(**kwargs)
        else:
            self.data[obs_id].update(**kwargs)

    def define_fiducial_cosmology(self, HDm_fid=None, params_fid=None,
                                  de_model='lambda'):
        r"""Define fiducial cosmology.

        Sets the internal attributes of the class to store the parameters of
        the fiducial cosmology, required for the calculation of the AP
        corrections.

        Parameters
        ----------
        HDm_fid: list or numpy.ndarray, optional
            List containing the fiducial expansion factor :math:`H(z)` and
            angular diameter distance :math:`D_\mathrm{A}(z)`, in the units
            defined by the class attribute **use_Mpc**. If **None**, this
            method expects to find a dictionary containing the parameters
            of the fiducial cosmology (see **params_fid** below). Defaults to
            **None**.
        params_fid: dict, optional
            Dictionary containing the parameters of the fiducial cosmology,
            used to compute the expansion factor :math:`H(z)` and angular
            diameter distance :math:`D_\mathrm{A}(z)`, in the units defined
            by the class attribute **use_Mpc**. Defaults to **None**.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen form the list [`"lambda"`, `"w0"`, `"w0wa"`].
            Defaults to `"lambda"`.
        """
        if HDm_fid is not None:
            self.H_fid = HDm_fid[0]
            self.Dm_fid = HDm_fid[1]
        else:
            Om0 = (params_fid['wc']+params_fid['wb'])/params_fid['h']**2
            H0 = params_fid['h']*100.0
            Ok0 = 0.0 if 'Ok' not in params_fid else params_fid['Ok']
            if de_model == 'lambda':
                w0 = -1.0
                wa = 0.0
            elif de_model == 'w0':
                w0 = params_fid['w0']
                wa = 0.0
            elif de_model == 'w0wa':
                w0 = params_fid['w0']
                wa = params_fid['wa']
            self.cosmo.update_cosmology(Om0, H0, Ok0=Ok0, de_model=de_model,
                                        w0=w0, wa=wa)
            self.h_fid = params_fid['h']
            self.H_fid = self.cosmo.Hz(params_fid['z'])
            self.Dm_fid = \
                self.cosmo.comoving_transverse_distance(params_fid['z'])

    def update_params(self, params, de_model=None):
        r"""Update parameters of the emulator.

        Sets the internal attributes of the class to store the parameters
        of the emulator, based on the input argument, and resets to **None**
        the internal dictionary containing the model ingredients.

        Parameters
        ----------
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        """
        def check_ranges(params_list):
            for p in params_list:
                if not self.params_ranges[p][0] <= self.params[p] \
                    <= self.params_ranges[p][1]:
                        print('Warning! Leaving emulator range' + \
                              'for parameter {}'.format(p))

        try:
            if de_model is None and self.use_Mpc:
                emu_params_updated = any([params[p] != self.params[p] for p
                                          in self.params_list])
                for p in self.params_list:
                    self.params[p] = params[p]
                self.params['As'] = 0.0
                self.params['z'] = 0.0
                check_ranges(self.params_list)
            elif de_model is None and not self.use_Mpc:
                emu_params_updated = any([params[p] != self.params[p] for p
                                          in self.params_list+['h']])
                for p in self.params_list+['h']:
                    self.params[p] = params[p]
                self.params['As'] = 0.0
                self.params['z'] = 0.0
                check_ranges(self.params_list)
            else:
                expected_params = self.params_shape_list \
                                  + self.de_model_params_list[de_model]
                if 'Ok' not in params:
                    expected_params.remove('Ok')
                emu_params_updated = any([params[p] != self.params[p] for p
                                          in expected_params])
                for p in expected_params:
                    self.params[p] = params[p]
                check_ranges(self.params_shape_list)
        except KeyError:
            print('Not all required parameter values have been defined.')

        if emu_params_updated:
            self.Pk_ratios = {0: None, 2: None, 4: None}
            self.splines_up_to_date = False
            self.dw_spline_up_to_date = False
            self.X_splines_up_to_date = {X: False for X
                                         in self.diagrams_all}
            self.chi2_decomposition = None

        for p in self.bias_params_list + self.RSD_params_list:
            if p in params.keys():
                self.params[p] = params[p]
            else:
                self.params[p] = 0.0

        if self.RSD_model == 'VDG_infty':
            self.params['cnlo'] = 0.0

        if self.bias_basis == 'AssBauGre':
            self.params['g2'] = self.params['bG2']
            self.params['g21'] = -4.0/7.0 * (self.params['bG2']
                                             + self.params['bGam3'])
        elif self.bias_basis == 'AmiGleKok':
            self.params['b1'] = self.params['b1t']
            self.params['b2'] = 2.0 * (-self.params['b1t'] + self.params['b2t']
                                       + self.params['b4t'])
            self.params['g2'] = -2.0/7.0 * (self.params['b1t']
                                            - self.params['b2t'])
            self.params['g21'] = -2.0/147.0 * (11*self.params['b1t']
                                               - 18*self.params['b2t']
                                               + 9*self.params['b3t'])

        return emu_params_updated

    def update_AP_params(self, params, de_model=None, q_tr_lo=None):
        r"""Update AP parameters.

        Sets the internal attributes of the class to store the AP parameters.

        Parameters
        ----------
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of expansion factors and
            angular diameter distance). Defaults to **None**.
        """
        if de_model is not None and q_tr_lo is None:
            Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
            H0 = 100.0*self.params['h']
            self.cosmo.update_cosmology(
                Om0=Om0, H0=H0, Ok0=self.params['Ok'],
                de_model=de_model, w0=self.params['w0'], wa=self.params['wa'])
            self.params['q_lo'] = \
                self.H_fid/self.cosmo.Hz(self.params['z'])
            self.params['q_tr'] = self.cosmo.comoving_transverse_distance(
                self.params['z'])/self.Dm_fid
            if not self.use_Mpc:
                self.params['q_lo'] *= (self.params['h']/self.h_fid)
                self.params['q_tr'] *= (self.params['h']/self.h_fid)
        elif de_model is not None:
            self.params['q_lo'] = q_tr_lo[1]
            self.params['q_tr'] = q_tr_lo[0]
        elif (de_model is None and
              'q_lo' in params and
              'q_tr' in params):
            self.params['q_lo'] = params['q_lo']
            self.params['q_tr'] = params['q_tr']

    def get_bias_coeff(self):
        r"""Get bias coefficients for the emulated terms.

        Each term of the :math:`P_{\ell}` expansion is multiplied by a
        combination of bias parameters. This method returns such
        combinations in an array format.

        Returns
        -------
        params_comb: numpy.ndarray
            Combinations of bias parameters that multiply each term of the
            expansion of the multipole of order :math:`\ell`. The output
            corresponds to

            .. math::
                :nowrap:

                    \begin{flalign*}
                        & P_{\delta\delta}^\mathrm{tree} \rightarrow b_1^2 \\
                        & P_{\delta\theta}^\mathrm{tree+1\mbox{-}loop} \
                        \rightarrow b_1 \\
                        & P_{\theta\theta}^\mathrm{tree+1\mbox{-}loop} \
                        \rightarrow 1 \\
                        & P_{\mathrm{ctr},k^2} \rightarrow \
                        [c_0,\: c_2,\: c_4] \\
                        & P_{\mathrm{ctr},k^4} \rightarrow \
                        [b_1^2c_\mathrm{nlo},\: b_1c_\mathrm{nlo},\: \
                        c_\mathrm{nlo}] \\
                        & P_{\delta\delta}^\mathrm{1\mbox{-}loop} \
                        \rightarrow b_1^2 \\
                        & P_{b_\mathrm{X}b_\mathrm{Y}} \rightarrow [b_1b_2, \
                        \: b_1\gamma_2,\: b_1\gamma_{21},\: \
                        b_2^2,\: b_2\gamma_2,\: \gamma_2^2,\: b_2,\: \
                        \gamma_2,\: \gamma_{21}]
                    \end{flalign*}

        """
        b1 = self.params['b1']
        b2 = self.params['b2']
        g2 = self.params['g2']
        g21 = self.params['g21']
        c0 = self.params['c0'] if self.use_Mpc \
            else self.params['c0']/self.params['h']**2
        c2 = self.params['c2'] if self.use_Mpc \
            else self.params['c2']/self.params['h']**2
        c4 = self.params['c4'] if self.use_Mpc \
            else self.params['c4']/self.params['h']**2
        cnlo = self.params['cnlo'] if self.use_Mpc \
            else self.params['cnlo']/self.params['h']**4
        b1sq = b1**2

        return np.array([b1sq, b1, 1., c0, c2, c4, b1sq*cnlo, b1*cnlo,
                         cnlo, b1sq, b1*b2, b1*g2, b1*g21, b2**2,
                         b2*g2, g2**2, b2, g2, g21])

    def get_bias_coeff_for_P6(self):
        r"""Get bias coefficients for the emulated terms of the octopole.

        Differently from the lower-order multipoles :math:`P_{0,2,4}`, the
        shape parameters of :math:`P_6` are kept fixed to the best values
        from Planck 2018 (TT+TE+EE+lowE+lensing), while each of the terms is
        rescaled by the current value of the growth rate :math:`f` and
        :math:`\sigma_{12}`. Each term of the :math:`P_6` expansion is
        therefore multiplied by a combination of growth rate and bias
        parameters. This method returns such combinations in an array format.

        Returns
        -------
        params_comb: numpy.ndarray
            Combinations of bias parameters that multiply each term of the
            expansion of the multipole of order 6. The output corresponds to

            .. math::
                :nowrap:

                    \begin{flalign*}
                    &P^\mathrm{tree}\rightarrow[b_1^2,\: fb_1,\: f^2] \\
                    &P^\mathrm{1\mbox{-}loop}\rightarrow[b_1^2,\: fb_1^2,\
                    \: f^2b_1^2,\: fb_1,\: f^2b_1,\: f^3b_1,\: f^2,\: \
                    f^3,\: f^4, \\
                    &\hspace{2.3cm} b_1b_2,\: fb_1b_2,\: b_1\gamma_2, \
                    \: fb_1\gamma_2,\: b_1\gamma_{21},\: b_2^2,\: \
                    b_2\gamma_2, \\
                    &\hspace{2.3cm} \gamma_2^2,\: fb_2,\: f^2b_2,\: \
                    f\gamma_2,\: f^2\gamma_2,\: f\gamma_{21}] \\
                    &P_{\mathrm{ctr},k^4}\rightarrow[f^4b_1^2 \
                    c_\mathrm{nlo},\: f^5b_1c_\mathrm{nlo},\: \
                    f^6c_\mathrm{nlo}]
                    \end{flalign*}
        """
        b1 = self.params['b1']
        b2 = self.params['b2']
        g2 = self.params['g2']
        g21 = self.params['g21']
        cnlo = self.params['cnlo'] if self.use_Mpc \
            else self.params['cnlo']/self.params['h']**4
        f = self.params['f']

        b1sq = b1**2
        b1f = b1*f
        f2 = f**2
        f3 = f**3
        f4 = f**4

        bb_tree = np.array([b1sq, b1*f, f2])
        bb_loop = np.array([b1sq, b1sq*f, b1sq*f2, b1f, b1f*f, b1f*f2,
                            f2, f3, f4, b1*b2, b1f*b2, b1*g2, b1f*g2, b1*g21,
                            b2**2, b2*g2, g2**2, b2*f, b2*f2, g2*f, g2*f2,
                            g21*f])
        bb_k4ctr = np.array([b1sq*f4, b1f*f4, f2*f4])*cnlo

        s12ratio = (self.params['s12']/self.s12_for_P6)**2
        bb_tree *= s12ratio
        bb_loop *= s12ratio**2
        bb_k4ctr *= s12ratio

        return np.hstack([bb_tree, bb_loop, bb_k4ctr])

    def get_bias_coeff_for_chi2_decomposition(self):
        r"""Get bias coefficients for the :math:`\chi^2` tables.

        In order to speed up the evaluation of the likelihood, the total
        :math:`\chi^2` is factorised into separate contributions scaling with
        different combinations of the bias and shot-noise parameters (the
        latter are expressed in units of the sample mean number density
        :math:`\bar{n}`). This method returns such combinations in an array
        format.

        Returns
        -------
        params_comb: numpy.ndarray
            Combinations of bias and noise parameters that multiply each term
            of the factorisation of the total :math:`\chi^2` into individual
            terms. The output correpsonds to

            .. math::
                :nowrap:

                    \begin{flalign*}
                        & P_{\delta\delta}^\mathrm{tree} \rightarrow b_1^2 \\
                        & P_{\delta\theta}^\mathrm{tree+1\mbox{-}loop} \
                        \rightarrow b_1 \\
                        & P_{\theta\theta}^\mathrm{tree+1\mbox{-}loop} \
                        \rightarrow 1 \\
                        & P_{\mathrm{ctr},k^2} \rightarrow [c_0,\: c_2,\: \
                        c_4] \\
                        & P_{\mathrm{ctr},k^4} \rightarrow \
                        [b_1^2c_\mathrm{nlo},\: b_1c_\mathrm{nlo},\: \
                        c_\mathrm{nlo}] \\
                        & P_{\delta\delta}^\mathrm{1\mbox{-}loop} \
                        \rightarrow b_1^2 \\
                        & P_{b_\mathrm{X}b_\mathrm{Y}} \rightarrow [b_1b_2, \
                        \: b_1\gamma_2,\: b_1\gamma_{21},\: \
                        b_2^2,\: b_2\gamma_2,\: \gamma_2^2,\: b_2,\: \
                        \gamma_2,\: \gamma_{21}] \\
                        & P_\mathrm{noise} \rightarrow [N_0/\bar{n},\: \
                        N_{20}/\bar{n},\: N_{22}/\bar{n}]
                    \end{flalign*}
        """
        b1 = self.params['b1']
        b2 = self.params['b2']
        g2 = self.params['g2']
        g21 = self.params['g21']
        c0 = self.params['c0'] if self.use_Mpc \
            else self.params['c0']/self.params['h']**2
        c2 = self.params['c2'] if self.use_Mpc \
            else self.params['c2']/self.params['h']**2
        c4 = self.params['c4'] if self.use_Mpc \
            else self.params['c4']/self.params['h']**2
        cnlo = self.params['cnlo'] if self.use_Mpc \
            else self.params['cnlo']/self.params['h']**4
        N0 = self.params['NP0'] if self.use_Mpc \
            else self.params['NP0']/self.params['h']**3
        N20 = self.params['NP20'] if self.use_Mpc \
            else self.params['NP20']/self.params['h']**5
        N22 = self.params['NP22'] if self.use_Mpc \
            else self.params['NP22']/self.params['h']**5
        b1sq = b1**2

        return np.array([b1sq, b1, 1., c0, c2, c4, b1sq*cnlo, b1*cnlo, cnlo,
                         b1sq, b1*b2, b1*g2, b1*g21, b2**2, b2*g2, g2**2, b2,
                         g2, g21, N0/self.nbar, N20/self.nbar, N22/self.nbar])

    def eval_emulator(self, params, ell, de_model=None):
        r"""Evaluate the emulators for the different terms.

        Sets up the internal parameters of the class, and evaluate the
        emulators for the various ingredients of the model, that are then
        stored as class attributes.

        The list of emulated quantities comprises the linear power spectrum
        :math:`P_\mathrm{L}(k)` (function of the shape parameters
        :math:`\mathbf{\Theta_{s}}`), the value of :math:`\sigma_{12}`
        (function of the shape parameters :math:`\mathbf{\Theta_{s}}`), and all
        the integral tables consisting of ratios between individual
        contributions to the one-loop galaxy power spectrum and the linear one
        (function of shape parameters :math:`\mathbf{\Theta_{s}}`, the growth
        rate :math:`f`, and :math:`\sigma_{12}`). For the `VDG_infty` model, an
        additional emulator is evaluated to obtain the value of the pairwise
        velocity dispersion, i.e. :math:`\sigma_\mathrm{v}`.

        Parameters
        ----------
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: list or numpy.ndarray
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`) and
            hexadecapole (:math:`\ell=4`).
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        """
        emu_params_updated = self.update_params(params, de_model=de_model)
        params_shape = np.array(
            [self.params[p] for p in self.params_shape_list])

        if de_model is None:
            params_all = np.array([self.params[p] for p in self.params_list])

            if self.Pk_lin is None or emu_params_updated:
                sigma12 = self.training['SHAPE'].transform_inv(
                    self.emu['s12'].predict(params_shape[None, :])[0][0],
                    's12')
                self.Pk_lin = self.training['SHAPE'].transform_inv(
                    self.emu['PL'].predict(params_shape[None, :])[0][0], 'PL')
                self.Pk_lin *= (self.params['s12']/sigma12)**2

                if self.RSD_model == 'VDG_infty':
                    self.params['sv'] = self.training['SHAPE'].transform_inv(
                        self.emu['sv'].predict(params_shape[None, :])[0][0],
                        'sv')[0]
                    self.params['sv'] *= self.params['s12']/sigma12
                    if not self.use_Mpc:
                        self.params['sv'] *= self.params['h']

            for m in ell:
                if self.Pk_ratios[m] is None or emu_params_updated:
                    self.Pk_ratios[m] = self.training['FULL'].transform_inv(
                        self.emu[m].predict(params_all[None, :])[0][0], m)
        else:
            if self.Pk_lin is None or emu_params_updated:
                sigma12 = self.training['SHAPE'].transform_inv(
                    self.emu['s12'].predict(params_shape[None, :])[0][0],
                    's12')
                self.Pk_lin = self.training['SHAPE'].transform_inv(
                    self.emu['PL'].predict(params_shape[None, :])[0][0], 'PL')

                # compute growth factors corresponding to fiducial and target
                # parameters + growth rate
                Om0_fid = (self.params['wc']+self.params['wb']) \
                    / self.emu_LCDM_params['h']**2
                H0_fid = 100.0*self.emu_LCDM_params['h']
                self.cosmo.update_cosmology(Om0=Om0_fid, H0=H0_fid)
                Dfid = self.cosmo.growth_factor(self.emu_LCDM_params['z'])

                Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
                H0 = 100.0*self.params['h']
                self.cosmo.update_cosmology(
                    Om0=Om0, H0=H0, Ok0=self.params['Ok'],
                    de_model=de_model, w0=self.params['w0'],
                    wa=self.params['wa'])
                D, f = self.cosmo.growth_factor(self.params['z'],
                                                get_growth_rate=True)

                # rescale linear power spectrum and sigma12
                amplitude_scaling = np.sqrt(
                    self.params['As']/self.emu_LCDM_params['As'])*D/Dfid
                self.Pk_lin *= amplitude_scaling**2
                self.params['s12'] = sigma12[0]*amplitude_scaling
                self.params['f'] = f

                for p in list(set(['s12','f']) & set(self.params_list)):
                    if not self.params_ranges[p][0] <= self.params[p] \
                        <= self.params_ranges[p][1]:
                            print('Warning! Leaving emulator range' + \
                                  'for parameter {}'.format(p))

                if self.RSD_model == 'VDG_infty':
                    self.params['sv'] = self.training['SHAPE'].transform_inv(
                        self.emu['sv'].predict(params_shape[None, :])[0][0],
                        'sv')[0]
                    self.params['sv'] *= amplitude_scaling
                    if not self.use_Mpc:
                        self.params['sv'] *= self.params['h']

            params_all = np.array([self.params[p] for p in self.params_list],
                                  dtype=object)

            for m in ell:
                if self.Pk_ratios[m] is None or emu_params_updated:
                    self.Pk_ratios[m] = self.training['FULL'].transform_inv(
                        self.emu[m].predict(params_all[None, :])[0][0], m)

    def W_kurt(self, k, mu):
        r"""Large scale limit of the velocity difference generating function.

        Method used exclusively if the `VDG_infty` model is specified.

        In the large scale limit, :math:`r\rightarrow\infty`, the velocity
        difference generating function :math:`W_\mathrm{G}` becomes
        scale-independent, with a gaussian limit given by

        .. math::
            W_\infty(\lambda)=e^{-\lambda^2\sigma_\mathrm{v}^2},

        where :math:`\lambda=fk\mu`, and :math:`\sigma_\mathrm{v}` is the
        pairwise velocity dispersion. This method returns a modified version
        of the gaussian limit, which also allows for non-zero kurtosis of the
        pairwise velocity distribution,

        .. math::
            W_\infty(\lambda)=\frac{1}{\sqrt(1+a_\mathrm{vir}^2\lambda^2)}
            e^{-\frac{\lambda^2\sigma_\mathrm{v}^2}
            {1+a_\mathrm{vir}^2\lambda^2}},

        where :math:`a_\mathrm{vir}` is a free parameter of the model, that can
        be specified in the list of model parameters when instantiating or
        updating the class.

        Parameters
        ----------
        k: float
            Value of the wavemode :math:`k`.
        mu: float
            Value of the cosine :math:`\mu` of the angle between
            the pair separation and the line of sight.

        Returns
        -------
        Winfty: float
            Value of the pairwise velocity generating function in the large
            scale limit.
        """
        t1 = (self.params['f']*k*mu)**2
        t2 = 1.0 + t1*self.params['avir']**2
        return 1.0/np.sqrt(t2)*np.exp(-t1*self.params['sv']**2/t2)

    def build_Pell_spline(self, Pell, ell):
        r"""Build spline object for power spectrum multipoles.

        Generates a cubic spline object for the specified power spectrum
        multipole, including the computation of effective indexes for the low-
        and high-:math:`k` tails of the multipole, and stores it as class
        attribute.

        Parameters
        ----------
        Pell: list or numpy.ndarray
            Array containing the power spectrum multipole of order
            :math:`\ell`, evaluated at the wavemodes defined by the class
            attribute **k_table**.
        ell: int
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        """
        id_min = 0 if not ell == 6 else self.nk-self.nkloop
        if self.use_Mpc:
            self.Pell_spline[ell] = UnivariateSpline(self.k_table, Pell,
                                                     k=3, s=0)
            self.Pell_min[ell] = Pell[id_min]
            self.Pell_max[ell] = Pell[-1]
            self.k_table_min[ell] = self.k_table[id_min]
            self.k_table_max[ell] = self.k_table[-1]
            dlP_min = np.log10(np.abs(Pell[id_min+2]/Pell[id_min]))
            dlP_max = np.log10(np.abs(Pell[-1]/Pell[-3]))
            dlk_min = np.log10(self.k_table[id_min+2]/self.k_table[id_min])
            dlk_max = np.log10(self.k_table[-1]/self.k_table[-3])
            self.neff_min[ell] = dlP_min/dlk_min
            self.neff_max[ell] = dlP_max/dlk_max
        else:
            Pell *= self.params['h']**3
            self.Pell_spline[ell] = UnivariateSpline(
                self.k_table/self.params['h'], Pell, k=3, s=0)
            self.Pell_min[ell] = Pell[id_min]
            self.Pell_max[ell] = Pell[-1]
            self.k_table_min[ell] = self.k_table[id_min]/self.params['h']
            self.k_table_max[ell] = self.k_table[-1]/self.params['h']
            dlP_min = np.log10(np.abs(Pell[id_min+2]/Pell[id_min]))
            dlP_max = np.log10(np.abs(Pell[-1]/Pell[-3]))
            dlk_min = np.log10(self.k_table[id_min+2]/self.k_table[id_min])
            dlk_max = np.log10(self.k_table[-1]/self.k_table[-3])
            self.neff_min[ell] = dlP_min/dlk_min
            self.neff_max[ell] = dlP_max/dlk_max

    def build_Pdw_spline(self, Pdw):
        r"""Build spline object for multipoles of linear de-wiggled power
        spectrum.

        Generates a cubic spline object for the linear de-wiggled power
        spectrum, including the computation of effective indexes for the low-
        and high-:math:`k` tails, and stores it as class attribute.

        Parameters
        ----------
        Pdw: list or numpy.ndarray
            Array containing the de-wiggled linear power spectrum evaluated at
            the wavemodes defined by the class attribute **k_table**.
        """
        id_min = 0
        if self.use_Mpc:
            self.Pdw_spline = UnivariateSpline(self.k_table, Pdw, k=3, s=0)
            self.Pdw_min = Pdw[id_min]
            self.Pdw_max = Pdw[-1]
            self.k_table_min[0] = self.k_table[id_min]
            self.k_table_max[0] = self.k_table[-1]
            dlP_min = np.log10(np.abs(Pdw[id_min+2]/Pdw[id_min]))
            dlP_max = np.log10(np.abs(Pdw[-1]/Pdw[-3]))
            dlk_min = np.log10(self.k_table[id_min+2]/self.k_table[id_min])
            dlk_max = np.log10(self.k_table[-1]/self.k_table[-3])
            self.neff_dw_min = dlP_min/dlk_min
            self.neff_dw_max = dlP_max/dlk_max
        else:
            Pdw *= self.params['h']**3
            self.Pdw_spline = UnivariateSpline(self.k_table/self.params['h'],
                                               Pdw, k=3, s=0)
            self.Pdw_min = Pdw[id_min]
            self.Pdw_max = Pdw[-1]
            self.k_table_min[0] = self.k_table[id_min]/self.params['h']
            self.k_table_max[0] = self.k_table[-1]/self.params['h']
            dlP_min = np.log10(np.abs(Pdw[id_min+2]/Pdw[id_min]))
            dlP_max = np.log10(np.abs(Pdw[-1]/Pdw[-3]))
            dlk_min = np.log10(self.k_table[id_min+2]/self.k_table[id_min])
            dlk_max = np.log10(self.k_table[-1]/self.k_table[-3])
            self.neff_dw_min = dlP_min/dlk_min
            self.neff_dw_max = dlP_max/dlk_max

    def eval_Pell_spline(self, k, ell):
        r"""Evaluate the spline of the specified power spectrum multipole.

        Calls the spline object stored as class attribute for the power
        spectrum multipole of given order :math:`\ell` on the input wavemodes
        :math:`k`. The called interpolator results in a cubic spline or in a
        power-law extrapolation, depending if the value of :math:`k` is within
        or outside the original boundary spcified by the training table.

        Parameters
        ----------
        k: numpy.ndarray
            Values of the requested wavemodes :math:`k`.
        ell: int
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).

        Returns
        -------
        spline: numpy.ndarray
            Interpolated power spectrum multipole of order :math:`\ell` at the
            requested wavemodes :math:`k`.
        """
        mask_low = k < self.k_table_min[ell]
        mask_high = k > self.k_table_max[ell]
        spline = np.hstack(
            [self.Pell_min[ell] *
             (k[mask_low]/self.k_table_min[ell])**self.neff_min[ell],
             self.Pell_spline[ell](
                k[np.invert(mask_low) & np.invert(mask_high)]),
             self.Pell_max[ell] *
             (k[mask_high]/self.k_table_max[ell])**self.neff_max[ell]]
            )
        return spline

    def eval_Pdw_spline(self, k):
        r"""Evaluate the spline of the linear de-wiggled power spectrum.

        Calls the spline object stored as class attribute for the linear
        de-wiggled power spectrum on the input wavemodes :math:`k`. The called
        interpolator results in a cubic spline or in a power-law extrapolation,
        depending if the value of :math:`k` is within or outside the original
        boundary spcified by the training table.

        Parameters
        ----------
        k: numpy.ndarray
            Values of the requested wavemodes :math:`k`.

        Returns
        -------
        spline: numpy.ndarray
            Interpolated linear de-wiggled power spectrum at the requested
            wavemodes :math:`k`.
        """
        mask_low = k < self.k_table_min[0]
        mask_high = k > self.k_table_max[0]
        spline = np.hstack(
            [self.Pdw_min *
             (k[mask_low]/self.k_table_min[0])**self.neff_dw_min,
             self.Pdw_spline(k[np.invert(mask_low) & np.invert(mask_high)]),
             self.Pdw_max *
             (k[mask_high]/self.k_table_max[0])**self.neff_dw_max]
            )
        return spline

    def PL(self, k, params, de_model=None):
        r"""Compute the linear power spectrum predictions.

        Evaluates the emulator calling **eval_emulator**, and returns the
        linear power spectrum :math:`P_\mathrm{L}(k)` at the specified
        wavemodes.

        Parameters
        ----------
        k: float or numpy.ndarray
            Value of the requested wavemodes :math:`k`.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.

        Returns
        -------
        PL: float or numpy.ndarray
            Linear power spectrum :math:`P_\mathrm{L}(k)` evaluated at the
            input wavemodes :math:`k`.
        """
        self.eval_emulator(params, ell=[], de_model=de_model)

        if self.use_Mpc:
            PL_spline = UnivariateSpline(self.k_table, self.Pk_lin, k=3, s=0)
        else:
            PL_spline = UnivariateSpline(self.k_table/self.params['h'],
                                         self.Pk_lin*self.params['h']**3,
                                         k=3, s=0)

        return PL_spline(k)

    def Pdw_2d(self, k, mu, params, de_model=None, ell_for_recon=None):
        r"""Compute the anisotropic leading order IR-resummed power spectrum.

        Evaluates the emulator calling **eval_emulator**, and returns the
        anisotropic leading order IR-resummed power spectrum
        :math:`P_\mathrm{IR-res}^\mathrm{LO}(k,\mu)`, defined as

        .. math::
            P_\mathrm{IR-res}^\mathrm{LO}(k,\mu) = P_\mathrm{nw}(k) + \
            e^{-k^2\Sigma^2(f,\mu)}P_\mathrm{w}(k),

        where :math:`P_\mathrm{nw}` and :math:`P_\mathrm{w}` are the no-wiggle
        and wiggle-only component of the linear matter power spectrum, and
        :math:`\Sigma(f,\mu)` is the anisotropic BAO damping factor due to
        infrared modes.

        Notice how this function does not include the leading order Kaiser
        effect due to the impact of the velocity field on the amplitude of
        the power spectrum.

        Parameters
        ----------
        k: float or numpy.ndarray
            Value of the requested wavemodes :math:`k`.
        mu: float or numpy.ndarray
            Value of the cosine :math:`\mu` of the angle between
            the pair separation and the line of sight.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        Pdw_2d: numpy.ndarray
            Leading-order infrared resummed power spectrum
            :math:`P_\mathrm{IR-res}^\mathrm{LO}(k,\mu)` evaluated at the
            input wavemodes :math:`k` and angles :math:`\mu`.
        """
        if ell_for_recon is None:
            ell_for_recon = [0, 2, 4, 6] if not self.real_space else [0]
        ell_eval_emu = ell_for_recon.copy()
        try:
            ell_eval_emu.remove(6)
        except Exception:
            pass
        self.eval_emulator(params, ell=ell_eval_emu, de_model=de_model)

        Pdw_ell = np.zeros([self.nk, len(ell_for_recon)])
        for i, ell in enumerate(ell_for_recon):
            if ell != 6:
                Pdw_ell[:, i] = self.Pk_ratios[ell][:self.nk]
            else:
                Pdw_ell[:, i] = self.P6[:, 0]
        Pdw_ell[:, :len(ell_eval_emu)] = (Pdw_ell[:, :len(ell_eval_emu)].T *
                                          self.Pk_lin).T

        Pdw_spline = {}
        for i, ell in enumerate(ell_for_recon):
            if self.use_Mpc:
                Pdw_spline[ell] = UnivariateSpline(self.k_table, Pdw_ell[:, i],
                                                   k=3, s=0)
            else:
                Pdw_spline[ell] = UnivariateSpline(
                    self.k_table/self.params['h'],
                    Pdw_ell[:, i]*self.params['h']**3,
                    k=3, s=0)

        Pdw_2d = 0.0
        for ell in ell_for_recon:
            Pdw_2d += np.outer(Pdw_spline[ell](k), eval_legendre(ell, mu))

        return Pdw_2d

    def Pdw(self, k, params, de_model=None, ell_for_recon=None):
        r"""Compute the real space leading order IR-resummed power spectrum.

        Evaluates the emulator calling **eval_emulator**, and returns the
        real space (:math:`\mu = 0`) leading order IR-resummed power spectrum
        :math:`P_\mathrm{IR-res}^\mathrm{LO}(k,\mu)`, defined as

        .. math::
            P_\mathrm{IR-res}^\mathrm{LO}(k,\mu) = P_\mathrm{nw}(k) + \
            e^{-k^2\Sigma^2(f,\mu)}P_\mathrm{w}(k),

        where :math:`P_\mathrm{nw}` and :math:`P_\mathrm{w}` are the no-wiggle
        and wiggle-only component of the linear matter power spectrum, and
        :math:`\Sigma(f,\mu)` is the anisotropic BAO damping factor due to
        infrared modes.

        Notice how this function does not include the leading order Kaiser
        effect due to the impact of the velocity field on the amplitude of
        the power spectrum.

        Parameters
        ----------
        k: float or numpy.ndarray
            Value of the requested wavemodes :math:`k`.
        mu: float or numpy.ndarray
            Value of the cosine :math:`\mu` of the angle between
            the pair separation and the line of sight.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        Pdw_2d: numpy.ndarray
            Leading-order infrared resummed power spectrum
            :math:`P_\mathrm{IR-res}^\mathrm{LO}(k,\mu)` evaluated at the
            input wavemodes :math:`k` and angles :math:`\mu`.
        """
        if ell_for_recon is None:
            ell_for_recon = [0, 2, 4, 6] if not self.real_space else [0]
        ell_eval_emu = ell_for_recon.copy()
        try:
            ell_eval_emu.remove(6)
        except Exception:
            pass
        self.eval_emulator(params, ell=ell_eval_emu, de_model=de_model)

        if not self.dw_spline_up_to_date:
            Pdw_ell = np.zeros([self.nk, len(ell_for_recon)])
            for i, ell in enumerate(ell_for_recon):
                if ell != 6:
                    Pdw_ell[:, i] = self.Pk_ratios[ell][:self.nk]
                else:
                    Pdw_ell[:, i] = self.P6[:, 0]
            Pdw_ell[:, :len(ell_eval_emu)] = (Pdw_ell[:, :len(ell_eval_emu)].T *
                                              self.Pk_lin).T

            Pdw = 0.0
            for i, ell in enumerate(ell_for_recon):
                Pdw += Pdw_ell[:,i]*eval_legendre(ell, 0.0)
            self.build_Pdw_spline(Pdw)
            self.dw_spline_up_to_date = True

        Pdw = self.eval_Pdw_spline(k)
        return Pdw

    def Pell_fid_ktable(self, params, ell, de_model=None):
        r"""Compute the power spectrum multipoles at the training wavemodes.

        Returns the specified multipole at a fixed :math:`k` grid
        corresponding to the wavemodes used to train the emulator (without
        the need to recur to a spline interpolation in :math:`k`). The output
        power spectrum multipole is not corrected for AP distortions. Used
        for validation purposes.

        Parameters
        ----------
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.

        Returns
        -------
        Pell: numpy.ndarray
            Power spectrum multipole of order :math:`\ell` at the fixed
            :math:`k` grid used for the training of the emulator.
        """
        ell = [ell] if not isinstance(ell, list) else ell
        ell_eval_emu = ell.copy()
        try:
            ell_eval_emu.remove(6)
        except Exception:
            pass
        self.eval_emulator(params, ell_eval_emu, de_model=de_model)

        bij = self.get_bias_coeff()

        Pell = np.zeros([self.nk, len(ell)])
        for i, m in enumerate(ell):
            if m != 6:
                Pk_bij = np.zeros([self.nk, self.n_diagrams])
                Pk_bij[:, :9] = np.multiply(
                    self.Pk_ratios[m][:9*self.nk].reshape((9, self.nk)),
                    self.Pk_lin).T
                Pk_bij[(self.nk-self.nkloop):, 9:19] = np.multiply(
                    self.Pk_ratios[m][9*self.nk:].reshape((10, self.nkloop)),
                    self.Pk_lin[(self.nk-self.nkloop):]).T

                Pell[:, i] = np.dot(bij, Pk_bij.T)

                # add shot noise
                if m == 0:
                    N0 = self.params['NP0'] if self.use_Mpc \
                        else self.params['NP0']/self.params['h']**3
                    N20 = self.params['NP20'] if self.use_Mpc \
                        else self.params['NP20']/self.params['h']**5
                    Pell[:, i] += (np.ones_like(self.k_table)*N0/self.nbar +
                                   self.k_table**2*N20/self.nbar)
                elif m == 2:
                    N22 = self.params['NP22'] if self.use_Mpc \
                        else self.params['NP22']/self.params['h']**5
                    Pell[:, i] += self.k_table**2*N22/self.nbar
            else:
                bij_for_P6 = self.get_bias_coeff_for_P6()
                Pell[:, i] = np.dot(bij_for_P6, self.P6.T)

        return Pell

    def Pell(self, k, params, ell, de_model=None, q_tr_lo=None,
             W_damping=None, ell_for_recon=None):
        r"""Compute the power spectrum multipoles.

        Main method to compute the galaxy power spectrum multipoles.
        Returns the specified multipole at the given wavemodes :math:`k`.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the multipoles. If a list
            is passed, it has to match the size of `ell`, and in that case
            each wavemode refer to a given multipole.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int or list
            Specific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        Pell_dict: dict
            Dictionary containing all the requested power spectrum multipoles
            of order :math:`\ell` at the specified :math:`k`.
        """
        if ell_for_recon is None:
            ell_for_recon = [0, 2, 4, 6] if not self.real_space else [0]

        def P2d(q, mu):
            t = 0.0
            for m in ell_for_recon:
                t += eval_legendre(m, mu) * self.eval_Pell_spline(q, m)
            return t

        if self.RSD_model == 'EFT':

            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['q_lo']**2 +
                                (1.0 - mu2)/self.params['q_tr']**2)
                kp = k*APfac
                mup = mu/self.params['q_lo']/APfac
                return np.outer(P2d(kp, mup), eval_legendre(ell, mu))

        elif self.RSD_model == 'VDG_infty':
            if W_damping is None:
                W_damping = self.W_kurt

            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['q_lo']**2 +
                                (1.0 - mu2)/self.params['q_tr']**2)
                kp = k*APfac
                mup = mu/self.params['q_lo']/APfac
                P2d_damped = P2d(kp, mup) * W_damping(kp, mup)
                return np.outer(P2d_damped, eval_legendre(ell, mu))

        else:
            raise ValueError('Unsupported RSD model.')

        ell = [ell] if not isinstance(ell, list) else ell

        if isinstance(k, list):
            if len(k) != len(ell):
                raise ValueError("If 'k' is given as a list, it must match the"
                                 " length of 'ell'.")
            else:
                k_list = k
                k = np.unique(np.hstack(k_list))
        else:
            k_list = [k]*len(ell)

        params_updated = [params[p] != self.params[p] for p in params.keys()]
        params_nonzero = [x for x in self.bias_params_list +
                          self.RSD_params_list if self.params[x] != 0]

        if (any(params_updated) or
                any(p not in params.keys() for p in params_nonzero) or
                not self.splines_up_to_date):
            Pell = self.Pell_fid_ktable(params, ell=ell_for_recon,
                                        de_model=de_model)
            for i, m in enumerate(ell_for_recon):
                self.build_Pell_spline(Pell[:, i], m)
            self.splines_up_to_date = True
            # self.X_splines_up_to_date = {X: False for X in self.diagrams_all}
            # self.chi2_decomposition = None

        self.update_AP_params(params, de_model=de_model,
                              q_tr_lo=q_tr_lo)
        q3 = self.params['q_tr']**2 * self.params['q_lo']

        Pell_model = quad_vec(integrand, 0.0, 1.0)[0]
        Pell_model *= (2.0*np.array(ell)+1.0) / q3

        Pell_dict = {}
        for i, m in enumerate(ell):
            ids = np.intersect1d(k, k_list[i], return_indices=True)[1]
            Pell_dict['ell{}'.format(m)] = Pell_model[ids, i]

        return Pell_dict

    def Pell_fixed_cosmo_boost(self, k, params, ell, de_model=None,
                               q_tr_lo=None, W_damping=None,
                               ell_for_recon=None):
        r"""Compute the power spectrum multipoles (fast for fixed cosmology).

        Main method to compute the galaxy power spectrum multipoles.
        Returns the specified multipole at the given wavemodes :math:`k`.
        Differently from **Pell**, if the cosmology has not been varied from
        the last call, this method simply reconstruct the final multipoles by
        multiplying the stored model ingredients (which, at fixed cosmology
        are the same) by the new bias parameters.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the multipoles. If a list
            is passed, it has to match the size of `ell`, and in that case
            each wavemode refer to a given multipole.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int or list
            pecific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        Pell_dict: dict
            Dictionary containing all the requested power spectrum multipoles
            of order :math:`\ell` at the specified :math:`k`.
        """
        ell = [ell] if not isinstance(ell, list) else ell

        if isinstance(k, list):
            if len(k) != len(ell):
                raise ValueError("If 'k' is given as a list, it must match the"
                                 " length of 'ell'.")
            else:
                k_list = k
                k = np.unique(np.hstack(k_list))
        else:
            k_list = [k]*len(ell)

        if de_model is None and self.use_Mpc:
            check_params = self.params_list + self.RSD_params_list
        elif de_model is None and not self.use_Mpc:
            check_params = self.params_list + ['h'] + self.RSD_params_list
        else:
            check_params = self.params_shape_list \
                           + self.de_model_params_list[de_model] \
                           + self.RSD_params_list
            if 'Ok' not in params:
                check_params.remove('Ok')

        if (any(params[p] != self.params[p] for p in check_params) or
                not all(self.X_splines_up_to_date.values())):
            self.PX_ell_list = {
                'ell{}'.format(m): np.zeros(
                    [k_list[i].shape[0], len(self.diagrams_all)])
                for i, m in enumerate(ell)}
            for i, X in enumerate(self.diagrams_all):
                PX_ell = self.PX_ell(k_list, params, ell, X, de_model=de_model,
                                     q_tr_lo=q_tr_lo,
                                     W_damping=W_damping,
                                     ell_for_recon=ell_for_recon)
                for m in PX_ell.keys():
                    self.PX_ell_list[m][:, i] = PX_ell[m]

        for p in self.bias_params_list:
            if p in params.keys():
                self.params[p] = params[p]
            else:
                self.params[p] = 0.0
        # self.splines_up_to_date = False
        # self.dw_spline_up_to_date = False
        bX = self.get_bias_coeff_for_chi2_decomposition()

        Pell_dict = {}
        for i, m in enumerate(ell):
            Pell_dict['ell{}'.format(m)] = np.dot(
                self.PX_ell_list['ell{}'.format(m)], bX)

        return Pell_dict

    def Pell_convolved(self, k, params, ell, obs_id, de_model=None,
                       q_tr_lo=None, W_damping=None, ell_for_recon=None):
        r"""Compute the convolved power spectrum multipoles.

        Main method to compute the galaxy power spectrum multipoles convolved
        with a specific data window function.
        Returns the specified multipole at the given wavemodes :math:`k`.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the multipoles. If a list
            is passed, it has to match the size of `ell`, and in that case
            each wavemode refer to a given multipole.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int or list
            pecific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        obs_id: str
            Identifier of the data sample. Necessary to obtain access to the
            particular window function of the sample.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        Pell_dict: dict
            Dictionary containing all the requested power spectrum multipoles
            of order :math:`\ell` at the specified :math:`k`.
        """
        ell_for_mixing_matrix = [0, 2, 4] if not self.real_space else [0]
        Pell = self.Pell(self.data[obs_id].bins_mixing_matrix[:, 1], params,
                         ell_for_mixing_matrix, de_model, q_tr_lo,
                         W_damping, ell_for_recon)
        Pell_list = np.hstack([Pell['ell{}'.format(m)] for m
                               in ell_for_mixing_matrix])
        Pell_convolved = np.dot(self.data[obs_id].W_mixing_matrix, Pell_list)

        ell = [ell] if not isinstance(ell, list) else ell

        if isinstance(k, list):
            if len(k) != len(ell):
                raise ValueError("If 'k' is given as a list, it must match the"
                                 " length of 'ell'.")
            else:
                k_list = k
                k = np.unique(np.hstack(k_list))
        else:
            k_list = [k]*len(ell)

        Pell_dict = {}
        if k != self.data[obs_id].bins_mixing_matrix[:, 0]:
            for i, m in enumerate(ell):
                spline = UnivariateSpline(
                    self.data[obs_id].bins_mixing_matrix[:, 0],
                    Pell_convolved[:, i], k=3, s=0)
                Pell_dict['ell{}'.format(m)] = spline(k_list[i])
        else:
            for i, m in enumerate(ell):
                ids = np.intersect1d(k, k_list[i], return_indices=True)[1]
                Pell_dict['ell{}'.format(m)] = Pell_convolved[ids, i]

        return Pell_dict

    def PX(self, k, mu, params, X, de_model=None):
        r"""Compute the individual contribution X to the galaxy power spectrum.

        Returns the individual anisotropic contribution X to the galaxy power
        spectrum :math:`P_\mathrm{gg}(k,\mu)`.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the X contribution.
        mu: float or list or numpy.ndarray
            Cosinus :math:`\mu` between the pair separation and the line of
            sight at which to evaluate the X contribution.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        X: str
            Identifier of the contribution to the galaxy power spectrum. Can
            be chosen from the list [`"P0L_b1b1"`, `"PNL_b1"`, `"PNL_id"`,
            `"Pctr_c0"`, `"Pctr_c2"`, `"Pctr_c4"`,
            `"Pctr_b1b1cnlo"`, `"Pctr_b1cnlo"`, `"Pctr_cnlo"`,
            `"P1L_b1b1"`, `"P1L_b1b2"`, `"P1L_b1g2"`, `"P1L_b1g21"`,
            `"P1L_b2b2"`, `"P1L_b2g2"`, `"P1L_g2g2"`, `"P1L_b2"`, `"P1L_g2"`,
            `"P1L_g21"`].
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.

        Returns
        -------
        PX_2d: numpy.ndarray
            2-d array containing the X contribution to the galaxy power
            spectrum at the specified :math:`k` and :math:`\mu`.
        """
        ids = None
        for n, diagram in enumerate(self.diagrams_emulated):
            if diagram == X:
                if n < 9:
                    ids = [n*self.nk, (n+1)*self.nk]
                else:
                    ids = [9*self.nk + (n-9)*self.nkloop,
                           9*self.nk + (n-8)*self.nkloop]

        if ids is not None:
            ell_for_recon = [0, 2, 4] if not self.real_space else [0]
            self.eval_emulator(params, ell=ell_for_recon, de_model=de_model)

            PX_ell = np.zeros([self.nk, 3])
            for i, ell in enumerate(ell_for_recon):
                PX_ell[self.nk - (ids[1]-ids[0]):, i] = \
                    self.Pk_ratios[ell][ids[0]:ids[1]]
            PX_ell = (PX_ell.T*self.Pk_lin).T

            PX_spline = {}
            for i, ell in enumerate(ell_for_recon):
                if self.use_Mpc:
                    PX_spline[ell] = UnivariateSpline(self.k_table,
                                                      PX_ell[:, i],
                                                      k=3, s=0)
                else:
                    PX_spline[ell] = UnivariateSpline(
                        self.k_table/self.params['h'],
                        PX_ell[:, i]*self.params['h']**3,
                        k=3, s=0)

            PX_2d = 0.0
            for ell in ell_for_recon:
                PX_2d += np.outer(PX_spline[ell](k), eval_legendre(ell, mu))
        else:
            raise ValueError('{}: invalid identifier.'.format(X))

        return PX_2d

    def PX_ell6_novir_noAP(self, X):
        r"""Compute the individual contribution X to the octopole.

        Returns the individual contribution X to the octopole
        :math:`P_6(k)` of the training sample, multiplying it by the
        correspondent bias and growth coefficients.

        Parameters
        ----------
        X: str
            Identifier of the contribution to the octopole of the galaxy power
            spectrum. Can be chosen from the list [`"P0L_b1b1"`, `"PNL_b1"`,
            `"PNL_id"`, `"Pctr_b1b1cnlo"`, `"Pctr_b1cnlo"`,
            `"Pctr_cnlo"`, `"P1L_b1b1"`, `"P1L_b1b2"`, `"P1L_b1g2"`,
            `"P1L_b1g21"`, `"P1L_b2b2"`, `"P1L_b2g2"`, `"P1L_g2g2"`,
            `"P1L_b2"`, `"P1L_g2"`, `"P1L_g21"`].

        Returns
        -------
        P6X: numpy.ndarray
            Array containing the X contribution to the octopole :math:`P_6(k)`.
        """
        s12ratio = (self.params['s12']/self.s12_for_P6)**2
        s12ratio_sq = s12ratio**2
        f = self.params['f']
        if X == 'P0L_b1b1':
            P6X = self.P6[:, 0]*s12ratio
        elif X == 'PNL_b1':
            fvec = np.array([f*s12ratio, f*s12ratio_sq, f**2*s12ratio_sq,
                             f**3*s12ratio_sq])
            P6X = np.dot(self.P6[:, [1, 6, 7, 8]], fvec)
        elif X == 'PNL_id':
            f2 = f**2
            fvec = np.array([f2*s12ratio, f2*s12ratio_sq, f*f2*s12ratio_sq,
                             f2**2*s12ratio_sq])
            P6X = np.dot(self.P6[:, [2, 9, 10, 11]], fvec)
        elif X == 'P1L_b1b1':
            fvec = np.array([1, f, f**2])
            P6X = np.dot(self.P6[:, [3, 4, 5]], fvec)*s12ratio_sq
        elif X == 'P1L_b1b2':
            fvec = np.array([1, f])
            P6X = np.dot(self.P6[:, [12, 13]], fvec)*s12ratio_sq
        elif X == 'P1L_b1g2':
            fvec = np.array([1, f])
            P6X = np.dot(self.P6[:, [14, 15]], fvec)*s12ratio_sq
        elif X == 'P1L_b1g21':
            P6X = self.P6[:, 16]*s12ratio_sq
        elif X == 'P1L_b2b2':
            P6X = self.P6[:, 17]*s12ratio_sq
        elif X == 'P1L_b2g2':
            P6X = self.P6[:, 18]*s12ratio_sq
        elif X == 'P1L_g2g2':
            P6X = self.P6[:, 19]*s12ratio_sq
        elif X == 'P1L_b2':
            fvec = np.array([f, f**2])
            P6X = np.dot(self.P6[:, [20, 21]], fvec)*s12ratio_sq
        elif X == 'P1L_g2':
            fvec = np.array([f, f**2])
            P6X = np.dot(self.P6[:, [22, 23]], fvec)*s12ratio_sq
        elif X == 'P1L_g21':
            P6X = f*self.P6[:, 24]*s12ratio_sq
        elif X == 'Pctr_b1b1cnlo':
            P6X = f**4*self.P6[:, 25]*s12ratio
        elif X == 'Pctr_b1cnlo':
            P6X = f**5*self.P6[:, 26]*s12ratio
        elif X == 'Pctr_cnlo':
            P6X = f**6*self.P6[:, 27]*s12ratio
        return P6X

    def PX_ell(self, k, params, ell, X, de_model=None, q_tr_lo=None,
               W_damping=None, ell_for_recon=None):
        r"""Get the individual contribution to the power spectrum multipoles.

        Computes the individual contribution X to the galaxy power spectrum
        multipoles. Returns the contribution to the specified multipole at the
        specified wavemodes :math:`k`.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the multipoles. If a list
            is passed, it has to match the size of `ell`, and in that case
            each wavemode refer to a given multipole.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int or list
            pecific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        X: str
            Identifier of the contribution to the galaxy power spectrum. Can
            be chosen from the list [`"P0L_b1b1"`, `"PNL_b1"`, `"PNL_id"`,
            `"Pctr_c0"`, `"Pctr_c2"`, `"Pctr_c4"`,
            `"Pctr_b1b1cnlo"`, `"Pctr_b1cnlo"`, `"Pctr_cnlo"`,
            `"P1L_b1b1"`, `"P1L_b1b2"`, `"P1L_b1g2"`, `"P1L_b1g21"`,
            `"P1L_b2b2"`, `"P1L_b2g2"`, `"P1L_g2g2"`, `"P1L_b2"`, `"P1L_g2"`,
            `"P1L_g21"`, `"Pnoise_NP0"`, `"Pnoise_NP20"`, `"Pnoise_NP22"`].
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        PX_ell_dict: dict
            Dictionary containing the contributions to all the requested power
            spectrum multipoles of order :math:`\ell` at the specified
            :math:`k`.
        """
        if ell_for_recon is None:
            ell_for_recon = [0, 2, 4, 6] if not self.real_space else [0]
        ell_eval_emu = ell_for_recon.copy()
        try:
            ell_eval_emu.remove(6)
        except Exception:
            pass

        def P2d(q, mu):
            t = 0.0
            for m in ell_for_recon:
                t += eval_legendre(m, mu) * self.PX_ell_spline[X][m](q)
            return t

        if self.RSD_model == 'EFT':

            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['q_lo']**2 +
                                (1.0 - mu2)/self.params['q_tr']**2)
                kp = k*APfac
                mup = mu/self.params['q_lo']/APfac
                return np.outer(P2d(kp, mup), eval_legendre(ell, mu))

        elif self.RSD_model == 'VDG_infty':
            if W_damping is None:
                W_damping = self.W_kurt

            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['q_lo']**2 +
                                (1.0 - mu2)/self.params['q_tr']**2)
                kp = k*APfac
                mup = mu/self.params['q_lo']/APfac
                P2d_damped = P2d(kp, mup) * W_damping(kp, mup)
                return np.outer(P2d_damped, eval_legendre(ell, mu))

        else:
            raise ValueError('Unsupported RSD model.')

        ell = [ell] if not isinstance(ell, list) else ell

        if isinstance(k, list):
            if len(k) != len(ell):
                raise ValueError("If 'k' is given as a list, it must match the"
                                 " length of 'ell'.")
            else:
                k_list = k
                k = np.unique(np.hstack(k_list))
        else:
            k_list = [k]*len(ell)

        PX_ell = np.zeros([self.nk, len(ell_for_recon)])
        X_emu = X
        if X_emu in self.diagrams_emulated:
            for n, diagram in enumerate(self.diagrams_emulated):
                if diagram == X_emu:
                    if n < 9:
                        ids = [n*self.nk, (n+1)*self.nk]
                    else:
                        ids = [9*self.nk + (n-9)*self.nkloop,
                               9*self.nk + (n-8)*self.nkloop]

            self.eval_emulator(params, ell=ell_eval_emu, de_model=de_model)
            if X_emu in ['Pctr_c0', 'Pctr_c2', 'Pctr_c4']:
                for i, m in enumerate(ell_eval_emu):
                    PX_ell[self.nk - (ids[1]-ids[0]):, i] = \
                        self.Pk_ratios[m][ids[0]:ids[1]]
            else:
                for i, m in enumerate(ell_for_recon):
                    if m != 6:
                        PX_ell[self.nk - (ids[1]-ids[0]):, i] = \
                            self.Pk_ratios[m][ids[0]:ids[1]]
                    else:
                        PX_ell[:, i] = self.PX_ell6_novir_noAP(X_emu)
            PX_ell[:, :len(ell_eval_emu)] = (PX_ell[:, :len(ell_eval_emu)].T *
                                             self.Pk_lin).T
        else:
            if X_emu == 'Pnoise_NP0':
                PX_ell[:, 0] = np.ones_like(self.k_table)
            elif X_emu == 'Pnoise_NP20':
                PX_ell[:, 0] = self.k_table**2
            elif X_emu == 'Pnoise_NP22' and len(ell_for_recon) > 1:
                PX_ell[:, 1] = self.k_table**2

        for i, m in enumerate(ell_for_recon):
            if self.use_Mpc:
                self.PX_ell_spline[X][m] = UnivariateSpline(self.k_table,
                                                            PX_ell[:, i],
                                                            k=3, s=0)
            else:
                self.PX_ell_spline[X][m] = UnivariateSpline(
                    self.k_table/self.params['h'],
                    PX_ell[:, i]*self.params['h']**3,
                    k=3, s=0)

        self.X_splines_up_to_date[X] = True

        self.update_AP_params(params, de_model=de_model,
                              q_tr_lo=q_tr_lo)
        q3 = self.params['q_tr']**2 * self.params['q_lo']

        PX_ell_model = quad_vec(integrand, 0.0, 1.0)[0]
        PX_ell_model *= (2.0*np.array(ell)+1.0) / q3

        PX_ell_dict = {}
        for i, m in enumerate(ell):
            ids = np.intersect1d(k, k_list[i], return_indices=True)[1]
            PX_ell_dict['ell{}'.format(m)] = PX_ell_model[ids, i]

        return PX_ell_dict

    def Bell(self, tri, params, ell, de_model=None, kfun=None,
             q_tr_lo=None, ell_for_recon=None):
        ell = [ell] if not isinstance(ell, list) else ell
        if tri.ndim == 1:
            tri = tri[None,:]
        tri_sorted = np.flip(np.sort(tri, axis=1), axis=1)
        if np.any(tri != tri_sorted):
            tri = tri_sorted
            print('Warning. Triangle configurations sorted such that '
                  'k1 >= k2 >= k3.')

        if not np.all(self.Bisp.tri == tri):
            if kfun is None:
                kfun = tri[0,0]
                print('kfun not specified. Using kfun = {}'.format(kfun))
            self.Bisp.set_tri(tri, kfun)

        Pdw = self.Pdw(self.Bisp.tri_unique, params, de_model=de_model,
                       ell_for_recon=ell_for_recon)

        if self.real_space:
            neff = None
        else:
            neff = self.Bisp.tri_unique * self.Pdw_spline.derivative(n=1)(
                       self.Bisp.tri_unique) / Pdw

        self.update_AP_params(params, de_model=de_model,
                              q_tr_lo=q_tr_lo)

        Bell_dict = self.Bisp.Bell(Pdw, neff, self.params, ell)
        return Bell_dict

    def Avg_covariance(self, l1, l2, k, Pl, sigma_d, avg_los=3):

        def kxx_int(th, ph,l1, l2, k, b, inv_nbar, Pl, sigma_d):

            costh2 = np.sin(th)*np.cos(ph)
            costh1 = np.cos(th)
            sinth1 = np.sin(th)

            f= self.cosmo.growth_rate(self.params["z"])
            beta = f/b

            func = sinth1*(b**2 * (1 + beta*costh1**2) * \
                    (1 + beta*costh2**2) * Pl + inv_nbar * np.exp(
                    -k**2 * ( costh1**2 + costh2**2) * f**2*sigma_d**2/2
                    ))**2 * eval_legendre(l1, costh1) \
                    *eval_legendre(l2, costh2)

            return func

        def kll_int(th,l1, l2, k, b, inv_nbar, Pl, sigma_d):

            sinth1 = np.sin(th)
            costh1 = np.cos(th)

            f= self.cosmo.growth_rate(self.params["z"])
            beta = f/b

            func = 2*np.pi* sinth1 * ( b**2*(1 + beta*costh1**2)**2*Pl \
                    +  inv_nbar )**2 * eval_legendre(l1, costh1)* \
                    eval_legendre(l2,costh1)
            return func

        kxx_l1l2 = np.asarray([ dblquad(kxx_int, 0, 2*np.pi,
            lambda ph: 0,
            lambda th:
            1*np.pi,
            args=(l1,l2, k[i],
                    self.params["b1"],
                    1/self.nbar,
                    Pl[i],
                    sigma_d)
                    )[0]
            for i in range(len(k)) ])

        kll_l1l2  = np.asarray([ quad(kll_int,0, np.pi,
                  args=(l1,
                        l2,
                        k[i],
                        self.params["b1"],
                        1/self.nbar,
                        Pl[i],
                        sigma_d))[0]
                  for i in range(len(k)) ])

        if avg_los==3:
            return (1+2*(kxx_l1l2/kll_l1l2))/3

        elif avg_los==2:
            return (1+(kxx_l1l2/kll_l1l2))/2

    def Gaussian_covariance(self, l1, l2, k, dk, Pell, volume,
                            Nmodes=None, avg_cov=False, avg_los=3):
        r"""Compute the gaussian covariance of the power spectrum multipoles.

        Returns the gaussian covariance predictions for the specified power
        spectrum multipoles (of order :math:`\ell_1` and :math:`\ell_2`), at
        the specified wavemodes :math:`k`, and for the given volume.

        Parameters
        ----------
        l1: int
            Order of first power spectrum multipole.
        l2: int
            Order of second power spectrum multipole.
        k: numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the gaussian covariance.
        dk: float
            Width of the :math:`k` bins.
        Pell: dict
            Dictionary containing the monopole, quadrupole and hexadecapole
            of the power spectrum.
        volume: float
            Reference volume to be used in the calculation of the gaussian
            covariance.
        Nmodes: numpy.ndarray, optional
            Number of modes contained in each :math:`k` bin. If not provided,
            its calculation is carried out based on the value of :math:`k` and
            :math:`\mathrm{d}k`. Defaults to **None**.

        Returns
        -------
        cov: numpy.ndarray
            Gaussian covariance of the selected power spectrum multipoles.
        """
        if Nmodes is None:
            Nmodes = volume/3.0/(2.0*np.pi**2)*((k+dk/2.0)**3 - (k-dk/2.0)**3)

        if not self.real_space:
            P0 = Pell['ell0']
            P2 = Pell['ell2']
            P4 = Pell['ell4']

            if l1 == l2 == 0:
                cov = P0**2 + 1.0/5.0*P2**2 + 1.0/9.0*P4**2
            elif l1 == 0 and l2 == 2:
                cov = (2.0*P0*P2 + 2.0/7.0*P2**2 + 4.0/7.0*P2*P4 +
                       100.0/693.0*P4**2)
            elif l1 == l2 == 2:
                cov = (5.0*P0**2 + 20.0/7.0*P0*P2 + 20.0/7.0*P0*P4 +
                       15.0/7.0*P2**2 + 120.0/77.0*P2*P4 + 8945.0/9009.0*P4**2)
            elif l1 == 0 and l2 == 4:
                cov = (2.0*P0*P4 + 18.0/35.0*P2**2 + 40.0/77.0*P2*P4 +
                       162.0/1001.0*P4**2)
            elif l1 == 2 and l2 == 4:
                cov = (36.0/7.0*P0*P2 + 200.0/77.0*P0*P4 + 108.0/77.0*P2**2 +
                       3578.0/1001.0*P2*P4 + 900.0/1001.0*P4**2)
            elif l1 == l2 == 4:
                cov = (9.0*P0**2 + 360.0/77.0*P0*P2 + 2916.0/1001.0*P0*P4 +
                       16101.0/5005.0*P2**2 + 3240.0/1001.0*P2*P4 +
                       42849.0/17017.0*P4**2)
        else:
            cov = Pell['ell0']**2

        if avg_cov:
            Pl = self.PL(k,self.params, de_model="lambda")
            sigma_d = np.sqrt(quad(self.PL, np.min(k), np.max(k) ,
                        args=(self.params, "lambda") )[0] / (6*np.pi**2) )
            avg = self.Avg_covariance(l1, l2, k, Pl, sigma_d, avg_los)
        else:
            avg=1.

        cov *= 2.0/Nmodes*avg

        return cov

    def Pell_covariance(self, k, params, ell, dk, de_model=None,
                        q_tr_lo=None, W_damping=None,
                        volume=None, zmin=None, zmax=None,
                        fsky=15000.0/(360.0**2/np.pi), volfac=1.0,
                        avg_cov=False, avg_los=3):
        r"""Compute the gaussian covariance of the power spectrum multipoles.

        Generates the selected power spectrum multipoles for the specified set
        of parameters, and returns their gaussian covariance predictions at
        the specified wavemodes :math:`k`.

        Parameters
        ----------
        k: float or list or numpy.ndarray
            Wavemodes :math:`k` at which to evaluate the multipoles. If a list
            is passed, it has to match the size of `ell`, and in that case
            each wavemode refer to a given multipole.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        ell: int or list
            pecific multipole order :math:`\ell`.
            Can be chosen from the list [0,2,4,6], whose entries correspond to
            monopole (:math:`\ell=0`), quadrupole (:math:`\ell=2`),
            hexadecapole (:math:`\ell=4`) and octopole (:math:`\ell=6`).
        dk: float
            Width of the :math:`k` bins.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        volume: float, optional
            Reference volume to be used in the calculation of the gaussian
            covariance. Defaults to **None**.
        zmin: float, optional
            Minimum redshift of the volume used in the calculation of the
            gaussian covariance. Defaults to **None**.
        zmin: float, optional
            Maximum redshift of the volume used in the calculation of the
            gaussian covariance. Defaults to **None**.
        fsky: float, optional
            Sky fraction of the volume used in the calculation of the gaussian
            covariance (in units of radians). Defaults to
            :math:`15000\mathrm{deg}^2`.
        volfac: float, optional
            Rescaling volume fraction. Defaults to :math:`1`.

        Returns
        -------
        cov: numpy.ndarray
            Gaussian covariance of the selected power spectrum multipoles.
        """
        ell = [ell] if not isinstance(ell, list) else ell
        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the "
                             "length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        nbins = [x.shape[0] for x in k]
        cov = np.zeros([sum(nbins), sum(nbins)])

        k_all = np.unique(np.hstack(k))
        ell_for_cov = [0, 2, 4] if not self.real_space else 0
        Pell = self.Pell(k_all, params, ell=ell_for_cov, de_model=de_model,
                         q_tr_lo=q_tr_lo, W_damping=W_damping)
        Pell['ell0'] += 1.0/self.nbar

        if de_model is not None and volume is None:
            Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
            H0 = 100.0*self.params['h']
            self.cosmo.update_cosmology(Om0=Om0, H0=H0, Ok0=self.params['Ok'],
                                        de_model=de_model,
                                        w0=self.params['w0'],
                                        wa=self.params['wa'])
            volume = volfac*self.cosmo.comoving_volume(zmin, zmax, fsky)
            if not self.use_Mpc:
                volume *= self.params['h']**3
        elif de_model is None and volume is None:
            raise ValueError("If no dark energy model is specified, a value "
                             "for the volume must be provided.")

        for i, l1 in enumerate(ell):
            for j, l2 in enumerate(ell):
                if j >= i:
                    kij, id1, id2 = np.intersect1d(k[i], k[j],
                                                   return_indices=True)
                    ids_ij = np.intersect1d(k_all, kij, return_indices=True)[1]
                    cov_l1l2 = self.Gaussian_covariance(
                        l1, l2, k_all, dk, Pell, volume,
                        avg_cov=avg_cov, avg_los=avg_los)[ids_ij]
                    cov[sum(nbins[:i]):sum(nbins[:i+1]),
                        sum(nbins[:j]):sum(nbins[:j+1])][id1, id2] = cov_l1l2
                else:
                    cov[sum(nbins[:i]):sum(nbins[:i+1]),
                        sum(nbins[:j]):sum(nbins[:j+1])] = \
                        cov[sum(nbins[:j]):sum(nbins[:j+1]),
                            sum(nbins[:i]):sum(nbins[:i+1])].T

        return cov

    def chi2(self, obs_id, params, kmax, de_model=None, q_tr_lo=None,
             W_damping=None, chi2_decomposition=False, ell_for_recon=None):
        r"""Compute the :math:`\chi^2 for the given configurations`.

        Generates the selected power spectrum multipoles for the specified set
        of parameters, and returns the :math:`\chi^2` evaluated with the
        specified :math:`k_\mathrm{max}` for the specified data sample.

        Parameters
        ----------
        obs_id: str
            Identifier of the data sample.
        params: dict
            Dictionary containing the list of total model parameters which are
            internally used by the emulator. The keyword/value pairs of the
            dictionary specify the names and the values of the parameters,
            respectively.
        kmax: float or list
            Maximum wavemode up to which the :math:`\chi^2` is computed. If a
            float is passed, this is used for all the multipoles,
            else each value refers to a given multipoles.
        de_model: str, optional
            String that determines the dark energy equation of state. Can be
            chosen from the list [`"lambda"`, `"w0"`, `"w0wa"`] to work with
            the standard cosmological parameters, or be left undefined to use
            only :math:`\sigma_{12}`. Defaults to **None**.
        q_tr_lo: list or numpy.ndarray, optional
            List containing the user-provided AP parameters, in the form
            :math:`(q_\perp, q_\parallel)`. If provided, prevents
            computation from correct formulas (ratios of angular diameter
            distances and expansion factors wrt to the corresponding quantities
            of the fiducial cosmology). Defaults to **None**.
        W_damping: Callable[[float, float], float], optional
            Function returning the shape of the pairwise velocity generating
            function in the large scale limit, :math:`r\rightarrow\infty`. The
            function accepts two floats as arguments, corresponding to the
            wavemode :math:`k` and the cosinus of the angle between pair
            separation and line of sight :math:`\mu`, and returns a float. This
            function is used only with the **VDG_infty** model. If **None**, it
            uses the free kurtosis distribution defined by **W_kurt**.
            Defaults to **None**.
        chi2_decomposition: bool, optional
            Flag to determine if the :math:`\chi^2` is computed using the fast
            :math:`\chi^2` decomposition (**True**) or not (**False**).
            Defaults to **False**.
        ell_for_recon: list, optional
            List of :math:`\ell` values used for the reconstruction of the
            2d leading-order IR-resummed power spectrum. If **None**, all the
            even multipoles up to :math:`\ell=6` are used in the
            reconstruction. Defaults to **None**.

        Returns
        -------
        chi2: float
            Value of the :math:`\chi^2`.
        """
        obs_id = [obs_id] if not isinstance(obs_id, list) else obs_id

        if not isinstance(kmax, dict):
            kmax_dict = {}
            for oi in obs_id:
                kmax_dict[oi] = kmax
            kmax = kmax_dict

        for oi in obs_id:
            if (not self.data[oi].kmax_is_set or
                (self.data[oi].kmax != kmax[oi] and self.data[oi].kmax !=
                    [kmax[oi] for i in range(self.data[oi].n_ell)])):
                        self.data[oi].set_kmax(kmax[oi])
                        if self.data[oi].stat == 'bispectrum':
                            self.Bisp.set_tri(self.data[oi].bins_kmax[0],
                                              self.data[oi].kfun)
                        self.chi2_decomposition = None
            else:
                if self.data[oi].stat == 'bispectrum' and \
                    self.Bisp.tri is None:
                        self.Bisp.set_tri(self.data[oi].bins_kmax[0],
                                          self.data[oi].kfun)
            if self.data[oi].stat == 'bispectrum':
                chi2_decomposition = False # currently only implemented for Pk

        ell = {}
        for oi in obs_id:
            ell[oi] = [2*m for m in range(self.data[oi].n_ell)
                       if self.data[oi].nbins[m] > 0]

        if not chi2_decomposition:
            chi2 = 0.0
            for oi in obs_id:
                if self.data[oi].stat == 'powerspectrum':
                    Pell = self.Pell(self.data[oi].bins_kmax, params, ell[oi],
                                     de_model=de_model, q_tr_lo=q_tr_lo,
                                     W_damping=W_damping,
                                     ell_for_recon=ell_for_recon)
                    Pell_list = np.hstack([Pell[m] for m in Pell.keys()])

                    diff = Pell_list - self.data[oi].signal_kmax
                    chi2 += diff @ self.data[oi].inverse_cov_kmax @ diff.T
                elif self.data[oi].stat == 'bispectrum':
                    Pdw = self.Pdw(self.Bisp.tri_unique,
                                   params, de_model=de_model,
                                   ell_for_recon=ell_for_recon)
                    if self.real_space:
                        neff = None
                    else:
                        neff = self.Bisp.tri_unique * \
                               self.Pdw_spline.derivative(n=1)(
                                   self.Bisp.tri_unique) / Pdw
                    Bell = self.Bisp.Bell(Pdw, neff, self.params, ell[oi])
                    Bell_list = np.hstack([Bell[m] for m in Bell.keys()])

                    diff = Bell_list - self.data[oi].signal_kmax
                    Ldiff = self.data[oi].inverse_cov_kmax_cholesky @ diff
                    chi2 += np.sum(Ldiff**2)
        else:
            chi2 = 0.0
            for oi in obs_id:
                # check if cosmological + RSD parameters have changed, if so,
                # re-evaluate chi2 decomposition
                if de_model is None and self.use_Mpc:
                    check_params = self.params_list + self.RSD_params_list
                elif de_model is None and not self.use_Mpc:
                    check_params = self.params_list + ['h'] + \
                                   self.RSD_params_list
                else:
                    check_params = self.params_shape_list \
                                   + self.de_model_params_list[de_model] \
                                   + self.RSD_params_list
                    if 'Ok' not in params:
                        check_params.remove('Ok')

                if (any(params[p] != self.params[p] for p in check_params) or
                        self.chi2_decomposition is None):
                    PX_ell_list = np.zeros([sum(self.data[oi].nbins),
                                            len(self.diagrams_all)])
                    for i, X in enumerate(self.diagrams_all):
                        PX_ell = self.PX_ell(self.data[oi].bins_kmax,
                                             params, ell[oi], X,
                                             de_model=de_model,
                                             q_tr_lo=q_tr_lo,
                                             W_damping=W_damping,
                                             ell_for_recon=ell_for_recon)
                        PX_ell_list[:, i] = np.hstack([PX_ell[m] for m
                                                       in PX_ell.keys()])

                    self.chi2_decomposition = {}
                    self.chi2_decomposition['DD'] = self.data[oi].SN_kmax
                    self.chi2_decomposition['XD'] = PX_ell_list.T \
                        @ self.data[oi].inverse_cov_kmax \
                        @ self.data[oi].signal_kmax
                    self.chi2_decomposition['XX'] = PX_ell_list.T \
                        @ self.data[oi].inverse_cov_kmax @ PX_ell_list

                for p in self.bias_params_list:
                    if p in params.keys():
                        self.params[p] = params[p]
                    else:
                        self.params[p] = 0.
                self.splines_up_to_date = False
                self.dw_spline_up_to_date = False

                bX = self.get_bias_coeff_for_chi2_decomposition()
                chi2 += (bX @ self.chi2_decomposition['XX'] @ bX -
                         2*bX @ self.chi2_decomposition['XD'] +
                         self.chi2_decomposition['DD'])

        return chi2
