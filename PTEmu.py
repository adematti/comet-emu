import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.special import beta, betainc, eval_legendre
from astropy.io import fits
import GPy
import pickle
from pyDOE import *


class Cosmo:
    def __init__(self, Om0, H0, Ok0=0, Or0=0, de_model='lambda', w0=-1, wa=0):
        self.Om0 = Om0
        self.Ok0 = Ok0
        self.Or0 = Or0
        self.H0 = H0
        self.Ode0 = 1. - self.Om0 - self.Ok0 - self.Or0

        self.hubble_distance = 2.998E5/self.H0

        self.de_model = de_model
        self.w0 = w0
        self.wa = wa

        self.flat = True if self.Ok0 == 0 else False
        self.relspecies = False if self.Or0 == 0 else True


    def update_cosmology(self, Om0, H0, Ok0=0, Or0=0, de_model='lambda', w0=-1, wa=0):
        self.Om0 = Om0
        self.Ok0 = Ok0
        self.Or0 = Or0
        self.H0 = H0
        self.Ode0 = 1. - self.Om0 - self.Ok0 - self.Or0

        self.hubble_distance = 2.998E5/self.H0

        self.de_model = de_model
        self.w0 = w0
        self.wa = wa

        self.flat = True if self.Ok0 == 0 else False
        self.relspecies = False if self.Or0 == 0 else True


    def DE_z(self, z):
        if self.de_model == 'lambda':
            de_z = 1.
        elif self.de_model == 'w0':
            de_z = (1.+z)**(3.*(1.+self.w0))
        elif self.de_model == 'w0wa':
            a = 1./(1.+z)
            de_z = a**(-3.*(1.+self.w0+self.wa))*np.exp(-3.*self.wa*(1.-a))
        return de_z


    def wz(self, z):
        if self.de_model == 'lambda':
            w = -1.*np.ones_like(z)
        elif self.de_model == 'w0':
            w = self.w0*np.ones_like(z)
        elif self.de_model == 'w0wa':
            w = self.w0 + self.wa*z/(1.+z)
        return w


    def Ez(self, z):
        ainv = 1.+z
        Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
        if not self.flat:
            Ez2 += self.Ok0*ainv**2
        if self.relspecies:
            Ez2 += self.Or0*ainv**4
        return np.sqrt(Ez2)


    def one_over_Ez(self, z):
        return 1./self.Ez(z)


    def Hz(self, z):
        return self.H0*self.Ez(z)


    def Om(self, z):
        return self.Om0*(1.+z)**3/(self.Ez(z))**2


    def Ode(self, z):
        return self.Ode0/(self.Ez(z))**2*self.DE_z(z)


    def comoving_transverse_distance(self, z):
        r = quad(self.one_over_Ez, 0, z)[0]
        if self.flat:
            dm = r
        elif self.Ok0 > 0:
            sqrt_Ok0 = np.sqrt(self.Ok0)
            dm = np.sinh(sqrt_Ok0*r)/sqrt_Ok0
        else:
            sqrt_Ok0 = np.sqrt(-self.Ok0)
            dm = np.sin(sqrt_Ok0*r)/sqrt_Ok0
        return self.hubble_distance*dm


    def angular_diameter_distance(self, z):
        return self.comoving_transverse_distance(z)/(1.+z)


    def growth_factor(self, z, get_growth_rate=False):
        def Ez_for_D(z):
            ainv = 1.+z
            Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
            if not self.flat:
                Ez2 += self.Ok0*ainv**2
            if self.relspecies:
                Ez2 += self.Or0
            return np.sqrt(Ez2)

        def integrand(z):
            return (1.+z)/(Ez_for_D(z))**3

        def growth_factor_from_ODE(z_eval):
            def derivatives_D(a, y):
                z  = 1./a - 1.
                D  = y[0]
                Dp = y[1]

                wa   = self.wz(z)
                Oma  = self.Om(z)
                Odea = self.Ode(z)

                u1 = -(2. - 0.5*(Oma + (3.*wa+1.)*Odea))/a
                u2 = 1.5*Oma/a**2

                return [Dp, u1*Dp + u2*D]

            a_eval = np.array([1./(1. + z_eval)])
            a_min = np.fmin(a_eval, 1E-4)*0.99
            a_max = a_eval*1.01

            dic = solve_ivp(derivatives_D, (a_min, a_max), [a_min, 1.0],
                            t_eval=a_eval, atol=1E-6, rtol=1E-6, vectorized=True)
            D  = dic['y'][0,:]

            if (dic['status'] != 0) or (D.shape[0] != a_eval.shape[0]):
                raise Exception('The calculation of the growth factor failed.')

            if get_growth_rate:
                Dp = dic['y'][1,:]
                f = a_eval*Dp/D
                return [D, f]
            else:
                return D

        if self.de_model == 'lambda':
            if self.flat and not self.relspecies:
                a3 = 1./(1+z)**3
                Dz = 5./6*betainc(5./6,2./3,self.Ode0*a3/(self.Om0+self.Ode0*a3)) \
                   *(self.Om0/self.Ode0)**(1./3)*np.sqrt(1 + self.Om0/(self.Ode0*a3)) \
                   *beta(5./6,2./3)
            else:
                # integrate integral expression
                Dz = 2.5*self.Om0*Ez_for_D(z)*quad(integrand,z,np.inf)[0]
            if get_growth_rate:
                Omz = self.Om(z)
                f = -1. - Omz/2 + self.Ode(z) + 2.5*Omz/Dz/(1.+z)
                Dz = [Dz, f]
        else:
            # do full differential equation integration
            Dz = growth_factor_from_ODE(z)

        return Dz


    def growth_rate(self, z):
        def Ez_for_D(z):
            ainv = 1.+z
            Ez2 = self.Om0*ainv**3 + self.Ode0*self.DE_z(z)
            if not self.flat:
                Ez2 += self.Ok0*ainv**2
            if self.relspecies:
                Ez2 += self.Or0
            return np.sqrt(Ez2)

        if self.de_model == 'lambda':
            Omz = self.Om(z)
            f = -1. - Omz/2 + self.Ode(z) + 2.5*Omz/self.growth_factor(z)/(1.+z)
        else:
            f = self.growth_factor(z, get_growth_rate=True)[1]
        return f


    def comoving_volume(self, zmin, zmax, fsky):
        def differential_comoving_volume(z):
            dm = self.comoving_transverse_distance(z)
            return self.hubble_distance*dm**2/self.Ez(z)

        return fsky*4*np.pi*quad(differential_comoving_volume, zmin, zmax)[0]



class Tables:
    def __init__(self, params, validation=False):
        self.params = params
        self.n_params = len(params)
        self.param_ranges = None
        self.validation = validation
        self.model = None
        self.model_transformed = None

        self.n_diagrams = 19
        self.names_diagrams = ['P0L_b1b1', 'PNL_b1', 'PNL_id', 'P1L_b1b1', 'P1L_b1b2',
                               'P1L_b1g2','P1L_b1g21','P1L_b2b2','P1L_b2g2','P1L_g2g2',
                               'P1L_b2','P1L_g2','P1L_g21','Pctr_c0','Pctr_c2','Pctr_c4',
                               'Pctr_b1b1cnlo', 'Pctr_b1cnlo', 'Pctr_cnlo']


    def set_param_ranges(self, ranges):
        if self.param_ranges is None:
            self.param_ranges = {}
        for p in self.params:
            self.param_ranges[p] = ranges[p]


    def generate_samples(self, ranges, n_samples, n_trials=0):
        self.set_param_ranges(ranges)
        self.n_samples = n_samples

        if self.validation:
            self.samples = np.random.rand(self.n_samples, self.n_params)
        else:
            self.samples = lhs(self.n_params, samples=self.n_samples, criterion='center')
            dist = cdist(self.samples, self.samples, metric='euclidean')
            min_dist = np.amin(dist[dist > 0])
            for n in range(n_trials):
                samples_new = lhs(self.n_params, samples=self.n_samples, criterion='center')
                dist = cdist(samples_new, samples_new, metric='euclidean')
                min_dist_new = np.amin(dist[dist > 0])
                if (min_dist_new > min_dist):
                    min_dist = min_dist_new
                    self.samples = samples_new

        for n,p in enumerate(self.params):
            self.samples[:, n] = self.samples[:, n] * (self.param_ranges[p][1] - self.param_ranges[p][0]) \
                               + self.param_ranges[p][0]


    def save_samples(self, fname):
        np.savetxt(fname, self.samples)


    def load_samples(self, fname):
        self.samples = np.loadtxt(fname)
        self.n_samples = self.samples.shape[0]


    def assign_samples(self, samples_hdu):
        self.n_samples = samples_hdu.header['NAXIS2']
        if self.validation:
            self.samples = np.zeros([self.n_samples,samples_hdu.header['TFIELDS']])
            for i,p in enumerate([samples_hdu.header['TTYPE{}'.format(n+1)] for n in range(samples_hdu.header['TFIELDS'])]):
                self.samples[:,i] = samples_hdu.data[p]
        else:
            self.samples = np.zeros([self.n_samples,self.n_params])
            for i,p in enumerate(self.params):
                self.samples[:,i] = samples_hdu.data[p]


    def assign_table(self, table_hdu, nk, nkloop):
        self.nk     = nk
        self.nkloop = nkloop

        if 'MODEL_SHAPE' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            for TYPE in [table_hdu.header['TTYPE{}'.format(i+1)] for i in range(table_hdu.header['TFIELDS'])]:
                self.model[TYPE] = table_hdu.data[TYPE]
                if self.model[TYPE].ndim == 1:
                    self.model[TYPE]= self.model[TYPE][:,None]
            if not self.validation:
                self.transform_emulator_data(data_type=list(self.model.keys()))
        elif 'MODEL_FULL' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            self.model['PL'] = table_hdu.data['PL']
            for ell in [0,2,4]:
                for diagram in self.names_diagrams:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    if diagram == 'PNL_b1':
                        # combine tree-level, one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P0L_b1_ell{}'.format(ell)] \
                                                 + table_hdu.data['P1L_b1_ell{}'.format(ell)] \
                                                 + table_hdu.data['Pk2corr_b1_ell{}'.format(ell)]
                    elif diagram == 'PNL_id':
                        # combine tree-level, one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P0L_id_ell{}'.format(ell)] \
                                                 + table_hdu.data['P1L_id_ell{}'.format(ell)] \
                                                 + table_hdu.data['Pk2corr_id_ell{}'.format(ell)]
                    elif diagram == 'P1L_b1b1':
                        # combine one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P1L_b1b1_ell{}'.format(ell)] \
                                                 + table_hdu.data['Pk2corr_b1b1_ell{}'.format(ell)]
                    else:
                        self.model[diagram_full] = table_hdu.data[diagram_full]
            if not self.validation:
                self.transform_emulator_data()
            else:
                self.convert_dictionary()
        else:
            raise KeyError('HDU table does not contain valid identifiers.')


    def get_flip_and_offset(self, table):
        flip = []
        offset_list = []
        for i in range(table.shape[0]):
            idmax = np.abs(table[i,:]).argmax()
            flip.append(np.sign(table[i,idmax]))
            offset_list.append(np.abs(np.amin(flip[-1]*table[i,:])))
        if len(offset_list) == 0:
            max_offset = 0.
        else:
            max_offset = max(offset_list)*1.1
        return flip, max_offset


    def transform(self, table, data_type):
        if data_type not in ['PL','s12','sv']:
            self.flip[data_type], self.offset[data_type] = self.get_flip_and_offset(table.T)
        else:
            self.flip[data_type]    = np.ones(table.shape[1])
            self.offset[data_type]  = 0.
        temp = np.log10(self.flip[data_type]*table+self.offset[data_type])
        self.mean[data_type] = np.mean(temp, axis=0)
        self.std[data_type]  = np.std(temp, axis=0)
        return (temp-self.mean[data_type])/self.std[data_type]


    def transform_inv(self, table, data_type):
        return (10**(table*self.std[data_type] + self.mean[data_type]) - self.offset[data_type]) \
            *self.flip[data_type]


    def transform_emulator_data(self, data_type=None):
        if data_type is not None:
            data_type = [data_type] if not isinstance(data_type,list) else data_type
            for dt in data_type:
                if self.model_transformed is None:
                    self.model_transformed = {}
                    self.mean   = {}
                    self.std    = {}
                    self.flip   = {}
                    self.offset = {}
                self.model_transformed[dt] = self.transform(self.model[dt], data_type=dt)
        else:
            self.model_transformed = {}
            self.mean   = {}
            self.std    = {}
            self.flip   = {}
            self.offset = {}
            for ell in [0,2,4]:
                temp = np.zeros([self.n_samples, 7*self.nk + 10*self.nkloop])
                cnt = 0
                # only include the cell_ell counterterm (the others are negligible without AP)
                for diagram in ['P0L_b1b1', 'PNL_b1', 'PNL_id','Pctr_c{}'.format(ell),
                                'Pctr_b1b1cnlo','Pctr_b1cnlo','Pctr_cnlo']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, cnt*self.nk:(cnt+1)*self.nk] = self.model[diagram_full]/self.model['PL']
                    cnt += 1
                cnt = 0
                for diagram in ['P1L_b1b1', 'P1L_b1b2','P1L_b1g2','P1L_b1g21','P1L_b2b2',
                                'P1L_b2g2','P1L_g2g2','P1L_b2','P1L_g2','P1L_g21']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, 7*self.nk+cnt*self.nkloop:7*self.nk+(cnt+1)*self.nkloop] \
                        = self.model[diagram_full][:,(self.nk-self.nkloop):] \
                        / self.model['PL'][:,(self.nk-self.nkloop):]
                    cnt += 1

                self.model_transformed[ell] = self.transform(temp, data_type=ell)


    def convert_dictionary(self):
        temp = np.zeros([self.n_samples, self.nk, 1+self.n_diagrams*3])
        temp[:,:,0] = self.model['PL']
        for ell in [0,2,4]:
            for cnt,diagram in enumerate(self.names_diagrams):
                diagram_full = '{}_ell{}'.format(diagram, ell)
                temp[:, :, 1+cnt+self.n_diagrams*int(ell/2)] = self.model[diagram_full]
                cnt += 1
        self.model = temp


    def GPy_model(self, data_type):
        kernel = GPy.kern.RBF(input_dim=self.n_params,
                              variance=np.var(self.model_transformed[data_type]),
                              lengthscale=np.ones(self.n_params), ARD=True)
        return GPy.models.GPRegression(
            self.samples, self.model_transformed[data_type], kernel)



class PTEmu:
    def __init__(
        self, params, use_Mpc=True, fname_base=None,
        emu_LCDM_params={'wc':0.11544,'wb':0.0222191,'ns':0.9632,
                         'h':0.695,'As':2.2078559,'z':1.0}
        ):

        self.params_shape_list    = [key for key,val in params.items() \
                                     if 'SHAPE' in val]
        self.params_list          = [p for p in params.keys()]
        self.bias_params_list     = ['b1','b2','g2','g21','c0','c2','c4','cnlo',
                                     'N0','N20','N22']
        self.RSD_params_list      = []
        self.de_model_params_list = {'lambda':['h','As','Ok','z'],
                                     'w0':['h','As','Ok','w0','z'],
                                     'w0wa':['h','As','Ok','w0','wa','z']}

        self.params             = {p:0. for p in self.params_list + \
            self.bias_params_list + self.de_model_params_list['w0wa']}
        self.params['w0']       = -1
        self.params['alpha_tr'] = 1
        self.params['alpha_lo'] = 1
        self.emu_LCDM_params    = emu_LCDM_params

        self.use_Mpc  = use_Mpc
        self.nbar     = 1. # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc

        self.n_diagrams = 19

        self.training   = {}
        self.validation = {}

        self.training['SHAPE']    = Tables(self.params_shape_list)
        self.training['FULL']     = Tables(self.params_list)
        self.validation['SHAPE']  = Tables(self.params_shape_list, validation=True)
        self.validation['FULL']   = Tables(self.params_list, validation=True)

        self.emu   = {}
        self.cosmo = Cosmo(0.3, 67) # Initialise with arbitrary values

        self.Pk_lin    = None
        self.Pk_ratios = {0:None, 2:None, 4:None}
        self.Pell_spline = {}

        self.splines_up_to_date = False
        self.emu_params_updated = False
        self.kmax_is_set        = False

        if fname_base is not None:
            self.load_emulator_data(fname='{}.fits'.format(fname_base))
            self.load_emulator(fname_base=fname_base)


    def generate_samples(self, type, ranges, n_samples, n_trials=0, validation=False):
        if validation:
            self.validation[type].generate_samples(ranges, n_samples, n_trials)
        else:
            self.training[type].generate_samples(ranges, n_samples, n_trials)


    def save_samples(self, type, fname, validation=False):
        if validation:
            self.validation[type].save_samples(fname)
        else:
            self.training[type].save_samples(fname)


    def load_emulator_data(self, fname, validation=False):
        hdul = fits.open(fname)

        # check that parameter match and abort if not!
        if not validation:
            params_shape_fits = [hdul['PARAMS_SHAPE'].header['TTYPE{}'.format(n+1)] for n in range(hdul['PARAMS_SHAPE'].header['TFIELDS'])]
            if not set(self.params_shape_list) == set(params_shape_fits):
                raise KeyError('Fits table list of shape parameters does not match.')
            params_full_fits = [hdul['PARAMS_FULL'].header['TTYPE{}'.format(n+1)] for n in range(hdul['PARAMS_FULL'].header['TFIELDS'])]
            if not set(self.params_list) == set(params_full_fits):
                raise KeyError('Fits table list of all parameters does not match.')

        self.k_table   = hdul['K_TABLE'].data['bins']
        self.nk        = self.k_table.shape[0]
        self.nkloop    = sum(self.k_table > hdul['K_TABLE'].header['k1loop'])
        self.RSD_model = hdul['MODEL_FULL'].header['RSD_model']

        if self.RSD_model == 'VIR':
            self.RSD_params_list.append('avir')
            self.params['avir']  = 0

        if validation:
            self.validation['FULL'].model = None
            self.validation['FULL'].assign_samples(hdul['PARAMS_FULL'])
            self.validation['FULL'].assign_table(hdul['MODEL_FULL'], self.nk, self.nkloop)
        else:
            self.training['SHAPE'].assign_samples(hdul['PARAMS_SHAPE'])
            self.training['SHAPE'].assign_table(hdul['MODEL_SHAPE'], self.nk, self.nkloop)
            self.training['FULL'].assign_samples(hdul['PARAMS_FULL'])
            self.training['FULL'].assign_table(hdul['MODEL_FULL'], self.nk, self.nkloop)


    def train_emulator(self, max_f_eval=1000, num_restarts=5, data_type=None):
        if data_type is None:
            ell_train = [0,2,4] if 'f' in self.params_list else [0]
            self.emu['PL'] = self.training['SHAPE'].GPy_model('PL')
            self.emu['s12'] = self.training['SHAPE'].GPy_model('s12')
            if self.RSD_model == 'VIR':
                self.emu['sv'] = self.training['SHAPE'].GPy_model('sv')
            for ell in ell_train:
                self.emu[ell] = self.training['FULL'].GPy_model(ell)

            for dt in self.emu.keys():
                self.emu[dt].optimize(max_f_eval=max_f_eval)
                self.emu[dt].optimize_restarts(num_restarts=num_restarts)
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12','sv']:
                    self.emu[dt] = self.training['SHAPE'].GPy_model(dt)
                else:
                    self.emu[dt] = self.training['FULL'].GPy_model(dt)
                self.emu[dt].optimize(max_f_eval=max_f_eval)
                self.emu[dt].optimize_restarts(num_restarts=num_restarts)


    def save_emulator(self, fname_base, data_type=None):
        if data_type is None:
            ell_train = [0,2,4] if 'f' in self.params_list else [0]
            for dt in ['PL','s12']:
                with open('{}_{}.pickle'.format(fname_base, dt), "wb") as f:
                    pickle.dump(self.emu[dt], f)
            if self.RSD_model == 'VIR':
                with open('{}_sv.pickle'.format(fname_base), "wb") as f:
                    pickle.dump(self.emu['sv'], f)
            for ell in ell_train:
                with open('{}_ratios_ell{}.pickle'.format(fname_base, ell), "wb") as f:
                    pickle.dump(self.emu[ell], f)
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12','sv']:
                    with open('{}_{}.pickle'.format(fname_base, dt), "wb") as f:
                        pickle.dump(self.emu[dt], f)
                else:
                    with open('{}_ratios_ell{}.pickle'.format(fname_base, dt), "wb") as f:
                        pickle.dump(self.emu[dt], f)


    def load_emulator(self, fname_base, data_type=None):
        if data_type is None:
            ell_train = [0,2,4] if 'f' in self.params_list else [0]
            for dt in ['PL','s12']:
                self.emu[dt] = pickle.load(open('{}_{}.pickle'.format(fname_base, dt), "rb"))
            if self.RSD_model == 'VIR':
                self.emu['sv'] = pickle.load(open('{}_{}.pickle'.format(fname_base, dt), "rb"))
            for ell in ell_train:
                self.emu[ell] = pickle.load(open('{}_ratios_ell{}.pickle'.format(fname_base, ell), "rb"))
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12','sv']:
                    self.emu[dt] = pickle.load(open('{}_{}.pickle'.format(fname_base, dt), "rb"))
                else:
                    self.emu[dt] = pickle.load(open('{}_ratios_ell{}.pickle'.format(fname_base, dt), "rb"))


    def define_units(self, use_Mpc):
        if use_Mpc != self.use_Mpc:
            self.use_Mpc = use_Mpc
            self.nbar = 1. # in units of Mpc^3 or (Mpc/h)^3 depending on use_Mpc
            try:
                self.k_data
            except:
                self.k_data = None
            try:
                self.P_data
            except:
                self.P_data = None
            try:
                self.Cov_data
            except:
                self.Cov_data = None
            if self.kmax_is_set:
                self.kmax_is_set = False
            self.splines_up_to_date = False
            nbar_unit = '(1/Mpc)^3' if self.use_Mpc else '(h/Mpc)^3'
            print('Number density resetted to nbar = 1 {}. Data set (if defined) cleared.'.format(nbar_unit))


    def define_nbar(self, nbar):
        self.nbar     = np.copy(nbar)
        self.splines_up_to_date = False


    def define_data_set(self, k_data=None, P_data=None, Cov_data=None, theory_cov=True, Nrealizations=None):
        if k_data is not None:
            self.k_data = np.copy(k_data)
        if P_data is not None:
            self.P_data = np.copy(P_data) if P_data.ndim > 1 else np.copy(P_data)[:,None]
            self.n_ell = self.P_data.shape[1]
        if Cov_data is not None:
            self.Cov_data = np.copy(Cov_data)

        self.theory_cov = theory_cov
        if not self.theory_cov:
            if Nrealizations is not None:
                self.Nrealizations = Nrealizations
            else:
                raise ValueError('For non-analytical covariance matrix, Nrealizations \
                                  needs to be specified.')

        # udpate kmax-truncated data containers
        if self.kmax_is_set:
            self.set_kmax(self.kmax)


    def define_fiducial_cosmology(self, HDm_fid=None, params_fid=None, de_model='lambda'):
        if HDm_fid is not None:
            self.H_fid = HDm_fid[0]
            self.Dm_fid = HDm_fid[1]
        else:
            Om0 = (params_fid['wc']+params_fid['wb'])/params_fid['h']**2
            H0 = params_fid['h']*100
            Ok0 = 0 if not 'Ok' in params_fid else params_fid['Ok']
            if de_model == 'lambda':
                w0 = -1
                wa = 0
            elif de_model == 'w0':
                w0 = params_fid['w0']
                wa = 0
            elif de_model == 'wa':
                w0 = params_fid['w0']
                wa = params_fid['wa']
            self.cosmo.update_cosmology(Om0, H0, Ok0=Ok0, de_model=de_model, w0=w0, wa=wa)
            self.H_fid = self.cosmo.Hz(params_fid['z'])
            self.Dm_fid = self.cosmo.comoving_transverse_distance(params_fid['z'])


    def set_kmax(self, kmax):
        if not isinstance(kmax, list):
            self.kmax = [kmax for i in range(self.n_ell)]
        else:
            self.kmax = kmax

        nbin_total = self.k_data.shape[0]

        self.nbin = [0 for i in range(self.n_ell)]
        for l in range(self.n_ell):
            for i in range(nbin_total):
                if self.k_data[i] < self.kmax[l]:
                    self.nbin[l] += 1
                else:
                    break

        self.k_bins = []
        self.P_data_kmax = np.array([])
        for l in range(self.n_ell):
            self.k_bins.append(self.k_data[:self.nbin[l]])
            self.P_data_kmax = np.concatenate((self.P_data_kmax,self.P_data[:self.nbin[l],l])) \
                if self.P_data_kmax.size else self.P_data[:self.nbin[l],l]

        self.Cov_data_kmax = np.zeros([sum(self.nbin),sum(self.nbin)])
        for l1 in range(self.n_ell):
            for l2 in range(self.n_ell):
                self.Cov_data_kmax[sum(self.nbin[:l1]):sum(self.nbin[:l1+1]),
                                   sum(self.nbin[:l2]):sum(self.nbin[:l2+1])] = \
                    self.Cov_data[l1*nbin_total:l1*nbin_total+self.nbin[l1],
                                  l2*nbin_total:l2*nbin_total+self.nbin[l2]]
        self.InvCov_data_kmax = self.AHfactor(sum(self.nbin))*np.linalg.inv(self.Cov_data_kmax)

        self.kmax_is_set = True


    def AHfactor(self, nbin):
        return 1. if self.theory_cov else (self.Nrealizations - nbin -2)*1./(self.Nrealizations - 1)


    def get_Pell_data(self, ell, kmax=None):
        if kmax is None and not self.kmax_is_set:
            self.set_kmax(np.amax(self.k_data))
        elif kmax is not None and (not self.kmax_is_set or \
                (self.kmax != kmax and self.kmax != [kmax for i in range(self.n_ell)])):
            self.set_kmax(kmax)
        n = int(ell/2)
        return self.P_data_kmax[sum(self.nbin[:n]):sum(self.nbin[:n+1])]


    def get_std_data(self, ell, kmax=None):
        if kmax is None and not self.kmax_is_set:
            self.set_kmax(np.amax(self.k_data))
        elif kmax is not None and (not self.kmax_is_set or \
                (self.kmax != kmax and self.kmax != [kmax for i in range(self.n_ell)])):
            self.set_kmax(kmax)
        n = int(ell/2)
        return np.sqrt(np.diag(self.Cov_data_kmax)[sum(self.nbin[:n]):sum(self.nbin[:n+1])])


    def update_params(self, params, de_model=None):
        try:
            if de_model is None and self.use_Mpc:
                emu_params_updated = any([params[p] != self.params[p] for p in self.params_list])
                for p in self.params_list:
                    self.params[p] = params[p]
                self.params['As'] = 0.
                self.params['z'] = 0.
            elif de_model is None and not self.use_Mpc:
                emu_params_updated = any([params[p] != self.params[p] for p in self.params_list+['h']])
                for p in self.params_list+['h']:
                    self.params[p] = params[p]
                self.params['As'] = 0.
                self.params['z'] = 0.
            else:
                expected_params = self.params_shape_list + self.de_model_params_list[de_model]
                if 'Ok' not in params:
                    expected_params.remove('Ok')
                emu_params_updated = any([params[p] != self.params[p] for p in expected_params])
                for p in expected_params:
                    self.params[p] = params[p]
        except KeyError:
            print('Not all required parameter values have been defined.')

        if emu_params_updated:
            self.Pk_ratios = {0:None, 2:None, 4:None}

        for p in self.bias_params_list + self.RSD_params_list:
            if p in params.keys():
                self.params[p] = params[p]
            else:
                self.params[p] = 0.

        return emu_params_updated


    def get_bias_coeff(self, ell):
        b1   = self.params['b1']
        b2   = self.params['b2']
        g2   = self.params['g2']
        g21  = self.params['g21']
        cell = self.params['c{}'.format(ell)] if self.use_Mpc \
             else self.params['c{}'.format(ell)]/self.params['h']**2
        cnlo = self.params['cnlo'] if self.use_Mpc \
             else self.params['cnlo']/self.params['h']**4
        b1sq     = b1**2

        return np.array([b1sq, b1, 1., cell, b1sq*cnlo, b1*cnlo, cnlo, b1sq,
                         b1*b2, b1*g2, b1*g21, b2**2, b2*g2, g2**2, b2, g2, g21])


    def get_bias_coeff_for_table(self):
        b1   = self.params['b1']
        b2   = self.params['b2']
        g2   = self.params['g2']
        g21  = self.params['g21']
        c0   = self.params['c0'] if self.use_Mpc \
            else self.params['c0']/self.params['h']**2
        c2   = self.params['c2'] if self.use_Mpc \
            else self.params['c2']/self.params['h']**2
        c4   = self.params['c4'] if self.use_Mpc \
            else self.params['c4']/self.params['h']**2
        cnlo = self.params['cnlo'] if self.use_Mpc \
            else self.params['cnlo']/self.params['h']**4
        b1sq     = b1**2

        return np.array([b1sq, b1, 1., c0, c2, c4, b1sq*cnlo, b1*cnlo, cnlo,
                         b1sq, b1*b2, b1*g2, b1*g21, b2**2, b2*g2, g2**2, b2,
                         g2, g21])


    def eval_emulator(self, params, ell, de_model=None):
        emu_params_updated = self.update_params(params, de_model=de_model)
        params_shape = np.array([self.params[p] for p in self.params_shape_list])

        if de_model is None:
            params_all = np.array([self.params[p] for p in self.params_list])

            if self.Pk_lin is None or emu_params_updated:
                sigma12 = self.training['SHAPE'].transform_inv(
                    self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')
                self.Pk_lin = self.training['SHAPE'].transform_inv(
                    self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')
                self.Pk_lin *= (self.params['s12']/sigma12)**2

                if self.RSD_model == 'VIR':
                    self.params['sv'] = self.training['SHAPE'].transform_inv(
                        self.emu['sv'].predict(params_shape[None,:])[0][0], 'sv')[0]
                    self.params['sv'] *= self.params['s12']/sigma12
                    if not self.use_Mpc:
                        self.params['sv'] *= self.params['h']

            for l in ell:
                if self.Pk_ratios[l] is None or emu_params_updated:
                    self.Pk_ratios[l] = self.training['FULL'].transform_inv(
                        self.emu[l].predict(params_all[None,:])[0][0], l)
        else:
            if self.Pk_lin is None or emu_params_updated:
                sigma12 = self.training['SHAPE'].transform_inv(
                    self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')
                self.Pk_lin = self.training['SHAPE'].transform_inv(
                    self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')

                # compute growth factors corresponding to fiducial and target parameters + growth rate
                Om0_fid = (self.params['wc']+self.params['wb'])/self.emu_LCDM_params['h']**2
                H0_fid = 100*self.emu_LCDM_params['h']
                self.cosmo.update_cosmology(Om0=Om0_fid, H0=H0_fid)
                Dfid = self.cosmo.growth_factor(self.emu_LCDM_params['z'])

                Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
                H0 = 100*self.params['h']
                self.cosmo.update_cosmology(
                    Om0=Om0, H0=H0, Ok0=self.params['Ok'],
                    de_model=de_model, w0=self.params['w0'], wa=self.params['wa'])
                D, f = self.cosmo.growth_factor(self.params['z'], get_growth_rate=True)

                # rescale linear power spectrum and sigma12
                amplitude_scaling = np.sqrt(self.params['As']/self.emu_LCDM_params['As'])*D/Dfid
                self.Pk_lin *= amplitude_scaling**2
                self.params['s12'] = sigma12[0]*amplitude_scaling
                self.params['f'] = f

                if self.RSD_model == 'VIR':
                    self.params['sv'] = self.training['SHAPE'].transform_inv(
                        self.emu['sv'].predict(params_shape[None,:])[0][0], 'sv')[0]
                    self.params['sv'] *= amplitude_scaling
                    if not self.use_Mpc:
                        self.params['sv'] *= self.params['h']

            params_all = np.array([self.params[p] for p in self.params_list],dtype=object)

            for l in ell:
                if self.Pk_ratios[l] is None or emu_params_updated:
                    self.Pk_ratios[l] = self.training['FULL'].transform_inv(
                        self.emu[l].predict(params_all[None,:])[0][0], l)


    def Pell_fid_ktable(self, params, ell, de_model=None):
        ell = [ell] if not isinstance(ell, list) else ell
        self.eval_emulator(params, ell, de_model=de_model)

        Pell = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)

            Pk_bij = np.zeros([self.nk,self.n_diagrams-2])
            Pk_bij[:,:7]                        = np.multiply(
                self.Pk_ratios[l][:7*self.nk].reshape(
                (7,self.nk)), self.Pk_lin).T
            Pk_bij[(self.nk-self.nkloop):,7:17] = np.multiply(
                self.Pk_ratios[l][7*self.nk:].reshape(
                    (10,self.nkloop)), self.Pk_lin[(self.nk-self.nkloop):]).T

            Pell[:,i] = np.dot(bij,Pk_bij.T)

            # add shot noise
            if l == 0:
                N0   = self.params['N0'] if self.use_Mpc \
                    else self.params['N0']/self.params['h']**3
                N20  = self.params['N20'] if self.use_Mpc \
                    else self.params['N20']/self.params['h']**5
                Pell[:,i] += np.ones_like(self.k_table)*N0/self.nbar \
                           + self.k_table**2*N20/self.nbar
            elif l == 2:
                N22  = self.params['N22'] if self.use_Mpc \
                    else self.params['N22']/self.params['h']**5
                Pell[:,i] += self.k_table**2*N22/self.nbar

        return Pell


    def W_kurt(self, k, mu):
        t1 = (self.params['f']*k*mu)**2
        t2 = 1. + t1*self.params['avir']**2
        return 1./np.sqrt(t2)*np.exp(-t1*self.params['sv']**2/t2)


    def Pell(self, k, params, ell, de_model=None, alpha_tr_lo=None, W_damping=None):
        ell_for_recon = [0,2,4] if 'f' in self.params_list else [0]

        def P2d(q, mu):
            t = 0.
            for l in ell_for_recon:
                t += eval_legendre(l, mu)*self.Pell_spline[l](q)
            return t

        if self.RSD_model == 'EFT':
            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['alpha_lo']**2
                                + (1. - mu2)/self.params['alpha_tr']**2)
                kp = k*APfac
                mup = mu/self.params['alpha_lo']/APfac
                return np.outer(P2d(kp, mup), eval_legendre(ell, mu))
        elif self.RSD_model == 'VIR':
            if W_damping is None:
                W_damping = self.W_kurt
            def integrand(mu):
                mu2 = mu**2
                APfac = np.sqrt(mu2/self.params['alpha_lo']**2
                                + (1. - mu2)/self.params['alpha_tr']**2)
                kp = k*APfac
                mup = mu/self.params['alpha_lo']/APfac
                P2d_damped = P2d(kp, mup) * W_damping(kp, mup)
                return np.outer(P2d_damped, eval_legendre(ell, mu))
        else:
            raise ValueError('Unsupported RSD model.')

        ell = [ell] if not isinstance(ell, list) else ell

        if isinstance(k, list):
            if len(k) != len(ell):
                raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
            else:
                k_list = k
                k = np.unique(np.hstack(k_list))
        else:
            k_list = [k]*len(ell)

        params_updated = [params[p] != self.params[p] for p in params.keys()]
        params_nonzero = [x for x in self.bias_params_list \
                          + self.RSD_params_list if self.params[x] != 0]

        if (any(params_updated) or
                any(p not in params.keys() for p in params_nonzero) or
                not self.splines_up_to_date):
            Pell = self.Pell_fid_ktable(params, ell=ell_for_recon,
                                        de_model=de_model)
            for i,l in enumerate(ell_for_recon):
                if self.use_Mpc:
                    self.Pell_spline[l] = interp1d(self.k_table, Pell[:,i],
                                                   kind='cubic')
                else:
                    self.Pell_spline[l] = interp1d(self.k_table/self.params['h'],
                                                   Pell[:,i]*self.params['h']**3,
                                                   kind='cubic')
            self.splines_up_to_date = True

        # update AP parameters
        if de_model is not None and alpha_tr_lo is None:
            self.params['alpha_lo'] = self.H_fid/self.cosmo.Hz(self.params['z'])
            self.params['alpha_tr'] = self.cosmo.comoving_transverse_distance(
                self.params['z'])/self.Dm_fid
        elif de_model is not None:
            self.params['alpha_lo'] = alpha_tr_lo[1]
            self.params['alpha_tr'] = alpha_tr_lo[0]
        elif de_model is None and 'alpha_lo' in params and 'alpha_tr' in params:
            self.params['alpha_lo'] = params['alpha_lo']
            self.params['alpha_tr'] = params['alpha_tr']
        alpha3 = self.params['alpha_tr']**2 * self.params['alpha_lo']

        Pell_model = quad_vec(integrand, 0, 1)[0]
        Pell_model *= (2*np.array(ell)+1) / alpha3

        Pell_dict = {}
        for i,l in enumerate(ell):
            ids = np.intersect1d(k, k_list[i], return_indices=True)[1]
            Pell_dict['ell{}'.format(l)] = Pell_model[ids,i]

        return Pell_dict


    def Pdw(self, k, mu, params, de_model=None):
        ell_for_recon = [0,2,4] if 'f' in self.params_list else [0]
        self.eval_emulator(params, ell=ell_for_recon, de_model=de_model)

        Pdw_ell = np.zeros([self.nk,3])
        for i,ell in enumerate([0,2,4]):
            Pdw_ell[:,i] = self.Pk_ratios[ell][:self.nk]
        Pdw_ell = (Pdw_ell.T*self.Pk_lin).T

        Pdw_spline = {}
        for i,ell in enumerate(ell_for_recon):
            if self.use_Mpc:
                Pdw_spline[ell] = interp1d(self.k_table, Pdw_ell[:,i],
                                           kind='cubic')
            else:
                Pdw_spline[ell] = interp1d(self.k_table/self.params['h'],
                                           Pdw_ell[:,i]*self.params['h']**3,
                                           kind='cubic')

        Pdw_2d = 0.
        for ell in ell_for_recon:
            Pdw_2d += np.outer(Pdw_spline[ell](k), eval_legendre(ell, mu))

        return Pdw_2d


    def PX(self, k, mu, params, X, de_model=None):
        ids = None
        for n,diagram in enumerate(
                ['P0L_b1b1', 'PNL_b1', 'PNL_id','Pctr_clo',
                'Pctr_b1b1cnlo','Pctr_b1cnlo','Pctr_cnlo','P1L_b1b1',
                'P1L_b1b2','P1L_b1g2','P1L_b1g21','P1L_b2b2',
                'P1L_b2g2','P1L_g2g2','P1L_b2','P1L_g2','P1L_g21']):
            if diagram == X:
                if n < 7:
                    ids = [n*self.nk, (n+1)*self.nk]
                else:
                    ids = [7*self.nk + (n-7)*self.nkloop,
                           7*self.nk + (n-6)*self.nkloop]

        if ids is not None:
            ell_for_recon = [0,2,4] if 'f' in self.params_list else [0]
            self.eval_emulator(params, ell=ell_for_recon, de_model=de_model)

            PX_ell = np.zeros([self.nk, 3])
            for i, ell in enumerate(ell_for_recon):
                PX_ell[self.nk - (ids[1]-ids[0]):,i] = self.Pk_ratios[ell][ids[0]:ids[1]]
            PX_ell = (PX_ell.T*self.Pk_lin[self.nk - (ids[1]-ids[0]):]).T

            PX_spline = {}
            for i, ell in enumerate(ell_for_recon):
                if self.use_Mpc:
                    PX_spline[ell] = interp1d(self.k_table, PX_ell[:,i],
                                              kind='cubic')
                else:
                    PX_spline[ell] = interp1d(self.k_table/self.params['h'],
                                              PX_ell[:,i]*self.params['h']**3,
                                              kind='cubic')

            PX_2d = 0.
            for ell in ell_for_recon:
                PX_2d += np.outer(PX_spline[ell](k), eval_legendre(ell, mu))
        else:
            raise ValueError('{}: invalid identifier.'.format(X))

        return PX_2d


    def PL(self, k, params, de_model=None):
        self.eval_emulator(params, ell=[], de_model=de_model)

        if self.use_Mpc:
            PL_spline = interp1d(self.k_table, self.Pk_lin, kind='cubic')
        else:
            PL_spline = interp1d(self.k_table/self.params['h'],
                                 self.Pk_lin*self.params['h']**3, kind='cubic')

        return PL_spline(k)


    def Pell_from_table_fid_ktable(self, table, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        self.params['h'] = params['h']
        for p in self.bias_params_list:
            if p in params.keys():
                self.params[p] = params[p]
            else:
                self.params[p] = 0.

        bij = self.get_bias_coeff_for_table()
        Pell = np.zeros([self.nk,len(ell)])

        for i,l in enumerate(ell):
            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            cnt = 0
            for n in np.array([0,1,2,13,14,15,16,17,18])+self.n_diagrams*int(l/2):
                Pk_bij[:,cnt] = table[:,1+n]
                cnt += 1
            for n in np.arange(3,13)+self.n_diagrams*int(l/2):
                Pk_bij[:,cnt] = table[:,1+n]
                cnt += 1

            Pell[:,i] = np.dot(bij,Pk_bij.T)

        # this is simply to guarantee that upon the next call of Pell the splines will be updated
        self.params['wc'] = -1

        return Pell


    def Pell_from_table(self, table, k, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell

        Pell_list = self.Pell_from_table_fid_ktable(table, params, ell)
        for i,l in enumerate(ell):
            if self.use_Mpc:
                self.Pell_spline[l] = interp1d(self.k_table*self.params['h']/self.emu_LCDM_params['h'], Pell_list[:,i], kind='cubic')
            else:
                self.Pell_spline[l] = interp1d(self.k_table/self.emu_LCDM_params['h'], Pell_list[:,i]*self.params['h']**3, kind='cubic')

        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        Pell_dict = {}
        for i,l in enumerate(ell):
            Pell_dict['ell{}'.format(l)] = self.Pell_spline[l](k[i])

        # this is simply to guarantee that upon the next call of Pell or Pell_LCDM the parameter values will be updated
        self.params['wc'] = -1
        self.splines_up_to_date = False

        return Pell_dict


    def Gaussian_covariance(self, l1, l2, k, dk, Pell, volume, Nmodes=None):
        if Nmodes is None:
            Nmodes = volume/3/(2*np.pi**2)*((k+dk/2)**3 - (k-dk/2)**3)

        if 'f' in self.params_list:
            P0 = Pell['ell0']
            P2 = Pell['ell2']
            P4 = Pell['ell4']

            if l1 == l2 == 0:
                cov = P0**2 + 1./5.*P2**2 + 1./9.*P4**2
            elif l1 == 0 and l2 == 2:
                cov = 2*P0*P2 + 2/7.*P2**2 + 4/7.*P2*P4+ 100/693.*P4**2
            elif l1 == l2 == 2:
                cov = 5*P0**2 + 20/7*P0*P2 + 20/7*P0*P4 + 15/7.*P2**2 \
                    + 120/77.*P2*P4 + 8945/9009.*P4**2
            elif l1 == 0 and l2 == 4:
                cov = 2*P0*P4 + 18/35*P2**2 + 40/77*P2*P4 + 162/1001.*P4**2
            elif l1 == 2 and l2 == 4:
                cov = 36/7*P0*P2 + 200/77*P0*P4 + 108/77.*P2**2 \
                    + 3578/1001*P2*P4 + 900/1001*P4**2
            elif l1 == l2 == 4:
                cov = 9*P0**2 + 360/77*P0*P2 + 2916/1001*P0*P4 \
                    + 16101/5005*P2**2 + 3240/1001*P2*P4 + 42849/17017*P4**2
        else:
            cov = Pell['ell0']**2

        cov *= 2./Nmodes
        return cov


    def Pell_covariance(self, k, params, ell, dk, de_model=None,
                        alpha_tr_lo=None, W_damping=None,
                        volume=None, zmin=None, zmax=None,
                        fsky=15000./(360**2/np.pi), volfac=1):
        ell = [ell] if not isinstance(ell, list) else ell
        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        nbin = [x.shape[0] for x in k]
        cov = np.zeros([sum(nbin),sum(nbin)])

        k_all = np.unique(np.hstack(k))
        ell_for_cov = [0,2,4] if 'f' in self.params_list else 0
        Pell = self.Pell(k_all, params, ell=ell_for_cov, de_model=de_model,
                         alpha_tr_lo=alpha_tr_lo, W_damping=W_damping)
        Pell['ell0'] += 1./self.nbar

        if de_model is not None and volume is None:
            Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
            H0 = 100*self.params['h']
            self.cosmo.update_cosmology(Om0=Om0, H0=H0, Ok0=self.params['Ok'],
                                        de_model=de_model, w0=self.params['w0'],
                                        wa=self.params['wa'])
            volume = volfac*self.cosmo.comoving_volume(zmin, zmax, fsky)
            if not self.use_Mpc:
                volume *= self.params['h']**3
        elif de_model is None and volume is None:
            raise ValueError("If no dark energy model is specified, a value for the volume must be provided.")

        for i,l1 in enumerate(ell):
            for j,l2 in enumerate(ell):
                if j >= i:
                    kij, id1, id2 = np.intersect1d(k[i],k[j],return_indices=True)
                    ids_ij = np.intersect1d(k_all, kij, return_indices=True)[1]
                    cov_l1l2 = self.Gaussian_covariance(
                        l1, l2, k_all, dk, Pell, volume)[ids_ij]
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),
                        sum(nbin[:j]):sum(nbin[:j+1])][id1,id2] = cov_l1l2
                else:
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),
                        sum(nbin[:j]):sum(nbin[:j+1])] = \
                        cov[sum(nbin[:j]):sum(nbin[:j+1]),
                            sum(nbin[:i]):sum(nbin[:i+1])].T

        return cov


    def Pell_covariance_from_table(self, table, k, params, ell, dk, volume=None, zmin=None, zmax=None, fsky=15000./(360**2/np.pi), volfac=1):
        ell = [ell] if not isinstance(ell, list) else ell
        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        nbin = [x.shape[0] for x in k]
        cov = np.zeros([sum(nbin),sum(nbin)])

        k_all = np.unique(np.hstack(k))
        ell_for_cov = [0,2,4] if 'f' in self.params_list else 0
        Pell = self.Pell_from_table(table, k_all, params, ell=ell_for_cov)
        Pell['ell0'] += 1./self.nbar

        if volume is None:
            Om0 = (params['wc']+params['wb'])/params['h']**2
            H0 = 100*params['h']
            self.cosmo.update_cosmology(Om0=Om0, H0=H0)
            volume = volfac*self.cosmo.comoving_volume(zmin, zmax, fsky)
            if not self.use_Mpc:
                volume *= self.params['h']**3

        for i,l1 in enumerate(ell):
            for j,l2 in enumerate(ell):
                if j >= i:
                    kij, id1, id2 = np.intersect1d(k[i],k[j],return_indices=True)
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),sum(nbin[:j]):sum(nbin[:j+1])][id1,id2] = self.Gaussian_covariance(l1, l2, k_all, dk, Pell, volume)[np.intersect1d(k_all, kij, return_indices=True)[1]]
                else:
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),sum(nbin[:j]):sum(nbin[:j+1])] = cov[sum(nbin[:j]):sum(nbin[:j+1]),sum(nbin[:i]):sum(nbin[:i+1])].T

        return cov


    def chi2(self, params, kmax, de_model=None, alpha_tr_lo=None, W_damping=None):
        if not self.kmax_is_set or (self.kmax != kmax and \
                self.kmax != [kmax for i in range(self.n_ell)]):
            self.set_kmax(kmax)

        ell = [2*l for l in range(self.n_ell) if self.nbin[l] > 0]
        Pell = self.Pell(self.k_bins, params, ell, de_model=de_model,
                         alpha_tr_lo=alpha_tr_lo, W_damping=W_damping)
        Pell_list = np.hstack([Pell['ell{}'.format(l)] for l in ell])

        diff = Pell_list - self.P_data_kmax

        return diff @ self.InvCov_data_kmax @ diff.T


    def chi2_from_table(self, table, params, kmax):
        if not self.kmax_is_set or (self.kmax != kmax and \
                self.kmax != [kmax for i in range(self.n_ell)]):
            self.set_kmax(kmax)

        Pell_model = np.zeros(sum(self.nbin))
        ell = [2*l for l in range(self.n_ell) if self.nbin[l] > 0]
        Pell = self.Pell_from_table_fid_ktable(table, params, ell)

        for i,l in enumerate(ell):
            n = int(l/2)

            if self.use_Mpc:
                spline = interp1d(self.k_table*self.params['h']/self.emu_LCDM_params['h'], Pell[:,i], kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])
            else:
                spline = interp1d(self.k_table/self.emu_LCDM_params['h'], Pell[:,i]*self.params['h']**3, kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])

        diff = Pell_model - self.P_data_kmax

        return diff @ self.InvCov_data_kmax @ diff.T


    # def convert_ranges_LCDM(self, ranges, z):
    #     def s12_params(self, params):
    #         params_shape = np.array([params[p] for p in self.params_shape_list])
    #         sigma12 = self.training['SHAPE'].transform_inv(self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')
    #
    #         Om0_fid = (params['wc']+params['wb'])/self.emu_LCDM_params['h']**2
    #         H0_fid = 100*self.emu_LCDM_params['h']
    #         self.cosmo.update_cosmology(Om0=Om0_fid, H0=H0_fid)
    #         Dfid = self.cosmo.growth_factor(self.emu_LCDM_params['z'])
    #
    #         Om0 = (params['wc']+params['wb'])/params['h']**2
    #         H0 = 100*params['h']
    #         self.cosmo.update_cosmology(Om0=Om0, H0=H0)
    #         D = self.cosmo.growth_factor(params['z'])
    #
    #         alpha_lo = self.H_fid/self.cosmo.Hz(params['z'])
    #         alpha_tr = self.cosmo.comoving_transverse_distance(params['z'])/self.Dm_fid
    #         f = self.cosmo.growth_rate(params['z'])
    #
    #         s12 = sigma12[0]*np.sqrt(params['As']/self.emu_LCDM_params['As'])*(D/Dfid)
    #         return np.array([s12,alpha_tr,alpha_lo,f])
    #
    #     params = {}
    #     params['z'] = z
    #
    #     limit_min = {}
    #     limit_min['s12'] = [0,1,0,1,0]
    #     limit_min['alpha_tr'] = [0,1,0,1,0] # doesn't depend on ns, As
    #     limit_min['alpha_lo'] = [0,1,0,1,0] # doesn't depend on ns, As
    #     limit_min['f'] = [0,0,0,1,0] # doesn't depend on ns, As
    #
    #     ranges_s12 = {}
    #     for i,p in enumerate(['s12','alpha_tr','alpha_lo','f']):
    #         ranges_s12[p] = np.zeros(2)
    #         for pLCDM in self.params_shape_list+['h','As']:
    #             params[pLCDM] = ranges[pLCDM][limits[p]]
    #         ranges[p][0] = s12_params(params)[i]
    #         for pLCDM in self.params_shape_list+['h','As']:
    #             params[pLCDM] = ranges[pLCDM][1-limits[p]]
    #         ranges[p][1] = s12_params(params)[i]
    #
    #     # check if it leaves emulator ranges and give out warning
    #
    #     return ranges_s12
