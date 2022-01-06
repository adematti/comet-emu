import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.integrate import quad
from scipy.special import beta, betainc
from astropy.io import fits
from astropy import cosmology
import GPy
import pickle
from pyDOE import *


class Cosmo:
    def __init__(self, Om0, H0):
        self.Om0 = Om0
        self.H0 = H0
        self.Ode0 = 1. - self.Om0
        self.cosmo_astropy = None


    def update_cosmology(self, Om0, H0):
        self.Om0 = Om0
        self.H0 = H0
        self.Ode0 = 1. - self.Om0
        if self.cosmo_astropy is not None:
            self.cosmo_astropy = cosmology.LambdaCDM(H0=self.H0,Om0=self.Om0,Ode0=self.Ode0)


    def Ez(self, z):
        return np.sqrt(self.Om0*(1+z)**3+1-self.Om0)


    def one_over_Ez(self, z):
        return 1./self.Ez(z)


    def Hz(self, z):
        return self.H0*self.Ez(z)


    def angular_diameter_distance(self, z):
        return 2.998E3*quad(self.one_over_Ez, 0, z)[0]


    def growth_factor(self, z):
        zp1 = 1./(1+z)**3
        return 5./6*betainc(5./6,2./3,self.Ode0*zp1/(self.Om0+self.Ode0*zp1))*(self.Om0/self.Ode0)**(1./3)*np.sqrt(1 + self.Om0/(self.Ode0*zp1))*beta(5./6,2./3)


    def growth_rate(self, z):
        return 5./2*self.Om0*((1+z)/self.Ez(z))**2*(1./self.growth_factor(z)-3./5*(1+z))


    def comoving_volume(self, zmin, zmax, fsky):
        if self.cosmo_astropy is None:
            self.cosmo_astropy = cosmology.LambdaCDM(H0=self.H0,Om0=self.Om0,Ode0=self.Ode0)
        return fsky*4*np.pi*quad(lambda z: self.cosmo_astropy.differential_comoving_volume(z).value, zmin, zmax)[0]



class Tables:
    def __init__(self, params, validation=False):
        self.params = params
        self.n_params = len(params)
        self.param_ranges = None
        self.validation = validation
        self.model = None
        self.model_transformed = None

        self.n_diagrams = 20
        self.names_diagrams = ['P0L_b1b1', 'PNL_b1', 'PNL_id', 'P1L_b1b1', 'P1L_b1b2','P1L_b1g2','P1L_b1g21','P1L_b2b2','P1L_b2g2','P1L_g2g2','P1L_b2','P1L_g2','P1L_g21',
                               'Pctr_cell', 'Pctr_b1b1cnlo', 'Pctr_b1cnlo', 'Pctr_cnlo', 'Pnoise_N0', 'Pnoise_N20', 'Pnoise_N22']


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
        self.samples = np.zeros([self.n_samples,self.n_params])
        for i,p in enumerate(self.params):
            self.samples[:,i] = samples_hdu.data[p]


    def load_table(self, fname, nk, nkloop, data_type=None):
        self.nk = nk
        self.nkloop = nkloop

        temp = np.loadtxt(fname)
        if data_type is not None:
            if self.model is None:
                self.model = {}
            if data_type == 'PL':
                self.model['PL'] = np.zeros([self.n_samples, self.nk])
                for j in range(self.n_samples):
                    self.model['PL'][j] = temp[j * self.nk:(j + 1) * self.nk]
            elif data_type == 's12':
                self.model['s12'] = np.zeros([self.n_samples,1])
                self.model['s12'][:,0] = temp
        else:
            self.model = np.zeros([self.n_samples, self.nk, 1+self.n_diagrams*3])
            for j in range(self.n_samples):
                self.model[j, :, 0] = temp[j * self.nk:(j + 1) * self.nk, 0]
            for l in range(3):
                for j in range(self.n_samples):
                    cnt = 0
                    for i in range(25):
                        if i not in [15,16,17] and i not in [10,14]:
                            self.model[j, :, 1+self.n_diagrams*l+cnt] = temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                            cnt += 1
                        elif i == 10: # prop. b1
                            self.model[j, :, 1+self.n_diagrams*l+1] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 14: # prop. 1 (no bias coefficient)
                            self.model[j, :, 1+self.n_diagrams*l+2] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 15: # k2-correction, prop. to b1^2 (note that this is included in the b1^2 loop correction, without the linear term -> table ID=3 opposed to 0)
                            self.model[j, :, 1+self.n_diagrams*l+3] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 16: # k2-correction, prop. to b1
                            self.model[j, :, 1+self.n_diagrams*l+1] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 17: # k2-correction, prop. to 1 (no bias coefficient)
                            self.model[j, :, 1+self.n_diagrams*l+2] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]

        if not self.validation:
            self.transform_training_data(data_type=data_type)


    def assign_table(self, table_hdu, nk, nkloop):
        self.nk     = nk
        self.nkloop = nkloop

        if 'MODEL_SHAPE' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            self.model['s12'] = table_hdu.data['s12'][:,None]
            self.model['PL']  = table_hdu.data['PL']
            if not self.validation:
                self.transform_emulator_data(data_type=['s12','PL'])
        elif 'MODEL_FULL' == table_hdu.header['EXTNAME']:
            if self.model is None:
                self.model = {}
            self.model['PL'] = table_hdu.data['PL']
            for ell in [0,2,4]:
                for diagram in self.names_diagrams:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    if diagram == 'PNL_b1':
                        # combine tree-level, one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P0L_b1_ell{}'.format(ell)] + table_hdu.data['P1L_b1_ell{}'.format(ell)] + table_hdu.data['Pk2corr_b1_ell{}'.format(ell)]
                    elif diagram == 'PNL_id':
                        # combine tree-level, one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P0L_id_ell{}'.format(ell)] + table_hdu.data['P1L_id_ell{}'.format(ell)] + table_hdu.data['Pk2corr_id_ell{}'.format(ell)]
                    elif diagram == 'P1L_b1b1':
                        # combine one-loop and IR k^2 correction terms
                        self.model[diagram_full] = table_hdu.data['P1L_b1b1_ell{}'.format(ell)] + table_hdu.data['Pk2corr_b1b1_ell{}'.format(ell)]
                    else:
                        self.model[diagram_full] = table_hdu.data[diagram_full]
            if not self.validation:
                self.transform_emulator_data()
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
        if data_type not in ['PL','s12']:
            self.flip[data_type], self.offset[data_type] = self.get_flip_and_offset(table.T)
        else:
            self.flip[data_type]    = np.ones(table.shape[1])
            self.offset[data_type]  = 0.
        temp = np.log10(self.flip[data_type]*table+self.offset[data_type])
        self.mean[data_type] = np.mean(temp, axis=0)
        self.std[data_type]  = np.std(temp, axis=0)
        return (temp-self.mean[data_type])/self.std[data_type]


    def transform_inv(self, table, data_type):
        return (10**(table*self.std[data_type] + self.mean[data_type]) - self.offset[data_type])*self.flip[data_type]


    def transform_training_data(self, data_type=None):
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
                temp = np.zeros([self.n_samples, 10*self.nk + 10*self.nkloop])
                cnt = 0
                for i in np.array([0,1,2,13,14,15,16])+10*ell:
                    for j in range(self.n_samples):
                        temp[j, cnt*self.nk:(cnt+1)*self.nk] = self.model[j,:,1+i]/self.model[j,:,0]
                    cnt += 1
                cnt = 0
                for i in np.array([17,18,19])+10*ell:
                    for j in range(self.n_samples):
                        temp[j, 7*self.nk+cnt*self.nk:7*self.nk+(cnt+1)*self.nk] = self.model[j,:,1+i]
                    cnt += 1
                cnt = 0
                for i in np.arange(3,13)+10*ell:
                    for j in range(self.n_samples):
                        temp[j, 10*self.nk+cnt*self.nkloop:10*self.nk+(cnt+1)*self.nkloop] = self.model[j,(self.nk-self.nkloop):,1+i]/self.model[j,(self.nk-self.nkloop):,0]
                    cnt += 1

                self.model_transformed[ell] = self.transform(temp, data_type=ell)


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
                temp = np.zeros([self.n_samples, 10*self.nk + 10*self.nkloop])
                cnt = 0
                for diagram in ['P0L_b1b1', 'PNL_b1', 'PNL_id','Pctr_cell','Pctr_b1b1cnlo','Pctr_b1cnlo','Pctr_cnlo']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, cnt*self.nk:(cnt+1)*self.nk] = self.model[diagram_full]/self.model['PL']
                    cnt += 1
                cnt = 0
                for diagram in ['Pnoise_N0', 'Pnoise_N20', 'Pnoise_N22']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, 7*self.nk+cnt*self.nk:7*self.nk+(cnt+1)*self.nk] = self.model[diagram_full]
                    cnt += 1
                cnt = 0
                for diagram in ['P1L_b1b1', 'P1L_b1b2','P1L_b1g2','P1L_b1g21','P1L_b2b2','P1L_b2g2','P1L_g2g2','P1L_b2','P1L_g2','P1L_g21']:
                    diagram_full = '{}_ell{}'.format(diagram, ell)
                    temp[:, 10*self.nk+cnt*self.nkloop:10*self.nk+(cnt+1)*self.nkloop] = self.model[diagram_full][:,(self.nk-self.nkloop):]/self.model['PL'][:,(self.nk-self.nkloop):]
                    cnt += 1

                self.model_transformed[ell] = self.transform(temp, data_type=ell)


    def GPy_model(self, data_type):
        kernel = GPy.kern.RBF(input_dim=self.n_params, variance=np.var(self.model_transformed[data_type]), lengthscale=np.ones(self.n_params), ARD=True)
        return GPy.models.GPRegression(self.samples, self.model_transformed[data_type], kernel)



class PTEmu:
    def __init__(self, params, fid_LCDM_params={'wc':0.11544,'wb':0.0222191,'ns':0.9632,'h':0.695, 'As':2.2078559, 'z':1.0}):

        self.params_shape_list = [key for key,val in params.items() if 'SHAPE' in val]
        self.params_list       = [p for p in params.keys()]
        self.bias_params_list  = ['b1','b2','g2','g21','c0','c2','c4','cnlo','N0','N20','N22']

        self.params          = {p:0. for p in self.params_list+self.bias_params_list+['h','As','z']}
        self.fid_LCDM_params = fid_LCDM_params

        self.n_diagrams = 20
        # self.kHD = 0.4  # this is in units of hfid_emu/Mpc
        self.kmax_is_set = False

        self.training   = {}
        self.validation = {}

        self.training['SHAPE']    = Tables(self.params_shape_list)
        self.training['FULL']     = Tables(self.params_list)
        self.validation['SHAPE']  = Tables(self.params_shape_list, validation=True)
        self.validation['FULL']   = Tables(self.params_list, validation=True)

        self.emu   = {}
        self.cosmo = None

        self.Pk_lin    = None
        self.Pk_ratios = {0:None, 2:None, 4:None}

        self.Pell_spline = {}

        self.emu_params_updated = False


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


    def load_samples(self, type, fname, validation=False):
        if validation:
            self.validation[type].load_samples(fname)
        else:
            self.training[type].load_samples(fname)


    def load_table(self, type, fname, fname_kvector, data_type=None, validation=False):
        self.k_table = np.loadtxt(fname_kvector)
        self.nk = self.k_table.shape[0]
        self.nkloop = sum(self.k_table > 0.007)
        if validation:
            self.validation[type].load_table(fname, self.nk, self.nkloop, data_type=data_type)
        else:
            self.training[type].load_table(fname, self.nk, self.nkloop, data_type=data_type)


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

        self.k_table = hdul['K_TABLE'].data['bins']
        self.nk = self.k_table.shape[0]
        self.nkloop = sum(self.k_table > hdul['K_TABLE'].header['k1loop'])

        if validation:
            self.validation['FULL'].assign_samples(hdul['PARAMS_FULL'])
            self.validation['FULL'].assign_table(hdul['MODEL_FULL'], self.nk, self.nkloop)
        else:
            self.training['SHAPE'].assign_samples(hdul['PARAMS_SHAPE'])
            self.training['SHAPE'].assign_table(hdul['MODEL_SHAPE'], self.nk, self.nkloop)
            self.training['FULL'].assign_samples(hdul['PARAMS_FULL'])
            self.training['FULL'].assign_table(hdul['MODEL_FULL'], self.nk, self.nkloop)


    def train_emulator(self, max_f_eval=1000, num_restarts=5, data_type=None):
        if data_type is None:
            self.emu['PL'] = self.training['SHAPE'].GPy_model('PL')
            self.emu['s12'] = self.training['SHAPE'].GPy_model('s12')
            for ell in [0,2,4]:
                self.emu[ell] = self.training['FULL'].GPy_model(ell)

            for dt in self.emu.keys():
                self.emu[dt].optimize(max_f_eval=max_f_eval)
                self.emu[dt].optimize_restarts(num_restarts=num_restarts)
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12']:
                    self.emu[dt] = self.training['SHAPE'].GPy_model(dt)
                else:
                    self.emu[dt] = self.training['FULL'].GPy_model(dt)
                self.emu[dt].optimize(max_f_eval=max_f_eval)
                self.emu[dt].optimize_restarts(num_restarts=num_restarts)


    def save_emulator(self, fname_base, data_type=None):
        if data_type is None:
            for dt in ['PL','s12']:
                with open('{}_{}.pickle'.format(fname_base, dt), "wb") as f:
                    pickle.dump(self.emu[dt], f)
            for ell in [0,2,4]:
                with open('{}_ratios_ell{}.pickle'.format(fname_base, ell), "wb") as f:
                    pickle.dump(self.emu[ell], f)
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12']:
                    with open('{}_{}.pickle'.format(fname_base, dt), "wb") as f:
                        pickle.dump(self.emu[dt], f)
                else:
                    with open('{}_ratios_ell{}.pickle'.format(fname_base, dt), "wb") as f:
                        pickle.dump(self.emu[dt], f)


    def load_emulator(self, fname_base, data_type=None):
        if data_type is None:
            for dt in ['PL','s12']:
                self.emu[dt] = pickle.load(open('{}_{}.pickle'.format(fname_base, dt), "rb"))
            for ell in [0,2,4]:
                self.emu[ell] = pickle.load(open('{}_ratios_ell{}.pickle'.format(fname_base, ell), "rb"))
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12']:
                    self.emu[dt] = pickle.load(open('{}_{}.pickle'.format(fname_base, dt), "rb"))
                else:
                    self.emu[dt] = pickle.load(open('{}_ratios_ell{}.pickle'.format(fname_base, dt), "rb"))


    def define_data_set(self, use_Mpc=True, k_data=None, P_data=None, Cov_data=None, nbar=None, kHD=None, theory_cov=True, Nrealizations=None):
        self.use_Mpc = use_Mpc

        if k_data is not None:
            self.k_data = np.copy(k_data)
        if P_data is not None:
            self.P_data = np.copy(P_data) if P_data.ndim > 1 else np.copy(P_data)[:,None]
            self.n_ell = self.P_data.shape[1]
        if Cov_data is not None:
            self.Cov_data = np.copy(Cov_data)
        if nbar is not None:
            self.nbar = np.copy(nbar)

        # in units of hfid_emu/Mpc
        if kHD is None:
            self.kHD = 0.278 if self.use_Mpc else 0.4
        else:
            self.kHD = np.copy(kHD)

        # if not self.use_Mpc:
        #     if hfid is not None:
        #         self.hfid_data = hfid
        #         self.k_data *= self.hfid_data
        #         self.P_data *= (1./self.hfid_data)**3
        #         self.Cov_data *= (1./self.hfid_data)**6
        #         self.nbar *= (self.hfid_data/self.fid_LCDM_params['h'])**3 # convert to units of (Mpc/hfid_emu)^-3
        #     else:
        #         raise ValueError('If data is not given in Mpc units, need to provide hfid.')
        #     self.nb
        # else:
        #    self.nbar *= (1./self.fid_LCDM_params['h'])**3 # convert to units of (Mpc/hfid_emu)^-3, necessary because the table is in units of 1/hfid_emu^3
        #    self.kHd  *= 1./self.fid_LCDM_params['h']      # convert to units of hfid_emu/Mpc

        self.nbar *= (1./self.fid_LCDM_params['h'])**3 # convert to units of (Mpc/hfid_emu)^-3 or (Mpc h/hfid_emu)^-3, necessary because the table is in units of 1/hfid_emu^3
        self.kHD  *= 1./self.fid_LCDM_params['h']      # convert to units of hfid_emu 1/Mpc or hfid_emu/h 1/Mpc

        self.splines_up_to_date = [False]*3

        self.theory_cov = theory_cov
        if not self.theory_cov:
            if Nrealizations is not None:
                self.Nrealizations = Nrealizations
            else:
                raise ValueError('For non-analytical covariance matrix, Nrealizations needs to be specified.')

        # udpate kmax truncated data containers
        if self.kmax_is_set:
            self.set_kmax(self.kmax)


    def AHfactor(self, nbin):
        return 1. if self.theory_cov else (self.Nrealizations - nbin -2)*1./(self.Nrealizations - 1)


    def set_kmax(self, kmax):
        if not isinstance(kmax, list):
            self.kmax = [kmax for i in range(self.n_ell)]
        else:
            self.kmax = kmax

        nbin_total = self.k_data.shape[0]

        # unit_factor = 1. if self.use_Mpc else self.hfid_data

        self.nbin = [0 for i in range(self.n_ell)]
        for l in range(self.n_ell):
            for i in range(nbin_total):
                if self.k_data[i] < self.kmax[l]: #*unit_factor:
                    self.nbin[l] += 1
                else:
                    break

        self.k_bins = []
        self.P_data_kmax = np.array([])
        for l in range(self.n_ell):
            self.k_bins.append(self.k_data[:self.nbin[l]])
            self.P_data_kmax = np.concatenate((self.P_data_kmax,self.P_data[:self.nbin[l],l])) if self.P_data_kmax.size else self.P_data[:self.nbin[l],l]

        self.Cov_data_kmax = np.zeros([sum(self.nbin),sum(self.nbin)])
        for l1 in range(self.n_ell):
            for l2 in range(self.n_ell):
                self.Cov_data_kmax[sum(self.nbin[:l1]):sum(self.nbin[:l1+1]),sum(self.nbin[:l2]):sum(self.nbin[:l2+1])] = self.Cov_data[l1*nbin_total:l1*nbin_total+self.nbin[l1],l2*nbin_total:l2*nbin_total+self.nbin[l2]]
        self.InvCov_data_kmax = self.AHfactor(sum(self.nbin))*np.linalg.inv(self.Cov_data_kmax)

        self.kmax_is_set = True


    def define_fiducial_cosmology(self, params_fid=None, HDm_fid=None):
        if HDm_fid is not None:
            self.cosmo = Cosmo(0.3, 67) # initialising with arbitrary parameters
            self.H_fid = HDm_fid[0]
            self.Dm_fid = HDm_fid[1]
        else:
            Om0 = (params_fid['wc']+params_fid['wb'])/params_fid['h']**2
            H0 = params_fid['h']*100
            self.cosmo = Cosmo(Om0, H0)
            self.H_fid = self.cosmo.Hz(params_fid['z'])
            self.Dm_fid = self.cosmo.angular_diameter_distance(params_fid['z'])/params_fid['h']


    def update_params(self, params, flag):
        if flag == 'TEMPLATE' and self.use_Mpc:
            emu_params_updated = any([params[p] != self.params[p] for p in self.params_list])
        elif flag == 'TEMPLATE':
            emu_params_updated = any([params[p] != self.params[p] for p in self.params_list+['h']])
        elif flag == 'LCDM':
            emu_params_updated = any([params[p] != self.params[p] for p in self.params_shape_list+['alpha_tr','alpha_lo','h','As','z']])

        try:
            if flag == 'TEMPLATE' and self.use_Mpc:
                for p in self.params_list:
                    self.params[p] = params[p]
                self.params['As'] = 0.
                self.params['z'] = 0.
            elif flag == 'TEMPLATE':
                for p in self.params_list+['h']:
                    self.params[p] = params[p]
                self.params['As'] = 0.
                self.params['z'] = 0.
            elif flag == 'LCDM':
                for p in self.params_shape_list+['h','As','z']:
                    self.params[p] = params[p]
        except KeyError:
            print('Not all required parameter values have been defined.')

        for p in self.bias_params_list:
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
        cell = self.params['c{}'.format(ell)] if self.use_Mpc else self.params['c{}'.format(ell)]/self.params['h']**2
        cnlo = self.params['cnlo'] if self.use_Mpc else self.params['cnlo']/self.params['h']**4
        N0   = self.params['N0'] if self.use_Mpc else self.params['N0']/self.params['h']**3
        N20  = self.params['N20'] if self.use_Mpc else self.params['N20']/self.params['h']**5
        N22  = self.params['N22'] if self.use_Mpc else self.params['N22']/self.params['h']**5
        return np.array([b1**2, b1, 1., cell/self.kHD**2, b1**2*cnlo/self.kHD**4, b1*cnlo/self.kHD**4,
                         cnlo/self.kHD**4, N0/self.nbar, N20/self.nbar/self.kHD**2, N22/self.nbar/self.kHD**2,
                         b1**2, b1*b2, b1*g2, b1*g21, b2**2, b2*g2, g2**2, b2, g2, g21])


    def Pell_fid_ktable(self, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        emu_params_updated = self.update_params(params, 'TEMPLATE')
        params_shape = np.array([self.params[p] for p in self.params_shape_list])
        params_all   = np.array([self.params[p] for p in self.params_list])

        if self.Pk_lin is None or emu_params_updated:
            sigma12 = self.training['SHAPE'].transform_inv(self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')
            self.Pk_lin = self.training['SHAPE'].transform_inv(self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')
            self.Pk_lin *= (self.params['s12']/sigma12)**2

        Pell_list = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)
            if self.Pk_ratios[l] is None or emu_params_updated:
                self.Pk_ratios[l] = self.training['FULL'].transform_inv(self.emu[l].predict(params_all[None,:])[0][0], l)

            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            for n in range(7):
                Pk_bij[:,n] = self.Pk_ratios[l][n*self.nk:(n+1)*self.nk]*self.Pk_lin
            for n in range(7,10):
                Pk_bij[:,n] = self.Pk_ratios[l][n*self.nk:(n+1)*self.nk]
            for n in range(10):
                Pk_bij[(self.nk-self.nkloop):,10+n] = self.Pk_ratios[l][10*self.nk+n*self.nkloop:10*self.nk+(n+1)*self.nkloop]*self.Pk_lin[(self.nk-self.nkloop):]

            Pell_list[:,i] = np.dot(bij,Pk_bij.T)

        return Pell_list


    def Pell(self, k, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        if any([params[p] != self.params[p] for p in params.keys()]):
            self.splines_up_to_date = [False]*3
            Pell_list = self.Pell_fid_ktable(params, ell)
            for i,l in enumerate(ell):
                if self.use_Mpc:
                    self.Pell_spline[l] = interp1d(self.k_table, Pell_list[:,i], kind='cubic')
                else:
                    self.Pell_spline[l] = interp1d(self.k_table/self.params['h'], Pell_list[:,i]*self.params['h']**3, kind='cubic')
                self.splines_up_to_date[int(l/2)] = True

        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        Pell_model = []
        for i,l in enumerate(ell):
            if self.splines_up_to_date[int(l/2)]:
                Pell_model.append(self.Pell_spline[l](k[i]))
            else:
                Pell = self.Pell_fid_ktable(params, l)
                if self.use_Mpc:
                    self.Pell_spline[l] = interp1d(self.k_table, Pell[:,0], kind='cubic')
                else:
                    self.Pell_spline[l] = interp1d(self.k_table/self.params['h'], Pell[:,0]*self.params['h']**3, kind='cubic')
                self.splines_up_to_date[int(l/2)] = True
                Pell_model.append(self.Pell_spline[l](k[i]))

        return Pell_model if len(ell) > 1 else Pell_model[0]


    def Pell_LCDM_fid_ktable(self, params, ell, alpha_tr_lo=None):
        ell = [ell] if not isinstance(ell, list) else ell
        if alpha_tr_lo is not None:
            self.params['alpha_tr'] = alpha_tr_lo[0]
            self.params['alpha_lo'] = alpha_tr_lo[1]
        emu_params_updated = self.update_params(params, 'LCDM')
        params_shape = np.array([self.params[p] for p in self.params_shape_list])

        if self.Pk_lin is None or emu_params_updated:
            sigma12 = self.training['SHAPE'].transform_inv(self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')
            self.Pk_lin = self.training['SHAPE'].transform_inv(self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')

            # compute growth factors corresponding to fiducial and target parameters
            Om0_fid = (self.params['wc']+self.params['wb'])/self.fid_LCDM_params['h']**2
            H0_fid = 100*self.fid_LCDM_params['h']
            self.cosmo.update_cosmology(Om0=Om0_fid, H0=H0_fid)
            Dfid = self.cosmo.growth_factor(self.fid_LCDM_params['z'])

            Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
            H0 = 100*self.params['h']
            self.cosmo.update_cosmology(Om0=Om0, H0=H0)
            D = self.cosmo.growth_factor(self.params['z'])

            # compute AP parameters and growth rate
            if alpha_tr_lo is None:
                self.params['alpha_lo'] = self.H_fid/self.cosmo.Hz(self.params['z'])
                self.params['alpha_tr'] = self.cosmo.angular_diameter_distance(self.params['z'])/self.params['h']/self.Dm_fid
            self.params['f'] = self.cosmo.growth_rate(self.params['z'])

            # rescale linear power spectrum and sigma12
            self.Pk_lin *= self.params['As']/self.fid_LCDM_params['As']*(D/Dfid)**2
            self.params['s12'] = sigma12[0]*np.sqrt(params['As']/self.fid_LCDM_params['As'])*(D/Dfid)

        params_all = np.array([self.params[p] for p in self.params_list],dtype=object)

        Pell_list = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)

            # rescale nbar and kHD for change in h
            # bij[3] *= (self.fid_LCDM_params['h']/self.params['h'])**2
            # bij[4:7] *= (self.fid_LCDM_params['h']/self.params['h'])**4
            # bij[7] *= (self.fid_LCDM_params['h']/self.params['h'])**3
            # bij[8:10] *= (self.fid_LCDM_params['h']/self.params['h'])**5

            if self.Pk_ratios[l] is None or emu_params_updated:
                self.Pk_ratios[l] = self.training['FULL'].transform_inv(self.emu[l].predict(params_all[None,:])[0][0], l)

            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            for n in range(7):
                Pk_bij[:,n] = self.Pk_ratios[l][n*self.nk:(n+1)*self.nk]*self.Pk_lin
            for n in range(7,10):
                Pk_bij[:,n] = self.Pk_ratios[l][n*self.nk:(n+1)*self.nk]
            for n in range(10):
                Pk_bij[(self.nk-self.nkloop):,10+n] = self.Pk_ratios[l][10*self.nk+n*self.nkloop:10*self.nk+(n+1)*self.nkloop]*self.Pk_lin[(self.nk-self.nkloop):]

            Pell_list[:,i] = np.dot(bij,Pk_bij.T)

        return Pell_list


    def Pell_LCDM(self, k, params, ell, alpha_tr_lo=None):
        ell = [ell] if not isinstance(ell, list) else ell

        if any([params[p] != self.params[p] for p in params.keys()]) \
        or (alpha_tr_lo is not None and any([alpha_tr_lo[i] != self.params[p] for i,p in enumerate(['alpha_tr','alpha_lo'])])):
            self.splines_up_to_date = [False]*3
            Pell_list = self.Pell_LCDM_fid_ktable(params, ell, alpha_tr_lo=alpha_tr_lo)
            for i,l in enumerate(ell):
                if self.use_Mpc:
                    self.Pell_spline[l] = interp1d(self.k_table, Pell_list[:,i], kind='cubic')
                else:
                    self.Pell_spline[l] = interp1d(self.k_table/self.params['h'], Pell_list[:,i]*self.params['h']**3, kind='cubic')
                self.splines_up_to_date[int(l/2)] = True

        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        Pell_model = []
        for i,l in enumerate(ell):
            if self.splines_up_to_date[int(l/2)]:
                Pell_model.append(self.Pell_spline[l](k[i]))
            else:
                Pell = self.Pell_LCDM_fid_ktable(params, l, alpha_tr_lo=alpha_tr_lo)
                if self.use_Mpc:
                    self.Pell_spline[l] = interp1d(self.k_table, Pell[:,0], kind='cubic')
                else:
                    self.Pell_spline[l] = interp1d(self.k_table/self.params['h'], Pell[:,0]*self.params['h']**3, kind='cubic')
                self.splines_up_to_date[int(l/2)] = True
                Pell_model.append(self.Pell_spline[l](k[i]))

        return Pell_model if len(ell) > 1 else Pell_model[0]


    def Pell_from_table_fid_ktable(self, table, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        self.params['h'] = params['h']
        for p in self.bias_params_list:
            if p in params.keys():
                self.params[p] = params[p]
            else:
                self.params[p] = 0.

        Pell_list = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)
            bij[3] *= (self.params['h']/self.fid_LCDM_params['h'])**2
            bij[4:7] *= (self.params['h']/self.fid_LCDM_params['h'])**4
            bij[7] *= (self.params['h']/self.fid_LCDM_params['h'])**3
            bij[8:10] *= (self.params['h']/self.fid_LCDM_params['h'])**5

            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            cnt = 0
            for n in np.array([0,1,2,13,14,15,16,17,18,19])+10*l:
                Pk_bij[:,cnt] = table[:,1+n]
                cnt += 1
            for n in np.arange(3,13)+10*l:
                Pk_bij[:,cnt] = table[:,1+n]
                cnt += 1

            Pell_list[:,i] = np.dot(bij,Pk_bij.T)

        # this is simply to guarantee that upon the next call of Pell or Pell_LCDM the parameter values will be updated
        self.params['wc'] = -1

        return Pell_list


    def Pell_from_table(self, table, k, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell

        Pell_list = self.Pell_from_table_fid_ktable(table, params, ell)
        for i,l in enumerate(ell):
            if self.use_Mpc:
                self.Pell_spline[l] = interp1d(self.k_table*self.params['h']/self.fid_LCDM_params['h'], Pell_list[:,i], kind='cubic')
            else:
                self.Pell_spline[l] = interp1d(self.k_table/self.fid_LCDM_params['h'], Pell_list[:,i]*self.params['h']**3, kind='cubic')

        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        Pell_model = []
        for i,l in enumerate(ell):
            Pell_model.append(self.Pell_spline[l](k[i]))

        # this is simply to guarantee that upon the next call of Pell or Pell_LCDM the parameter values will be updated
        self.params['wc'] = -1
        self.splines_up_to_date = [False]*3

        return Pell_model if len(ell) > 1 else Pell_model[0]


    def Gaussian_covariance(self, l1, l2, k, dk, Pell, volume, Nmodes=None):
        if Nmodes is None:
            Nmodes = volume/3/(2*np.pi**2)*((k+dk/2)**3 - (k-dk/2)**3)

        if 'f' in self.params_list:
            P0, P2, P4 = np.copy(Pell)

            if l1==l2==0:
                cov = P0**2 + 1./5.*P2**2 + 1./9.*P4**2
            elif l1==0 and l2==2:
                cov = 2*P0*P2 + 2/7.*P2**2 + 4/7.*P2*P4+ 100/693.*P4**2
            elif l1==l2==2:
                cov = 5*P0**2 + 20/7*P0*P2 + 20/7*P0*P4 + 15/7.*P2**2 + 120/77.*P2*P4 + 8945/9009.*P4**2
            elif l1==0 and l2==4:
                cov = 2*P0*P4 + 18/35*P2**2 + 40/77*P2*P4 + 162/1001.*P4**2
            elif l1==2 and l2==4:
                cov = 36/7*P0*P2 + 200/77*P0*P4 + 108/77.*P2**2 + 3578/1001*P2*P4 + 900/1001*P4**2
            elif l1==l2==4:
                cov = 9*P0**2 + 360/77*P0*P2 + 2916/1001*P0*P4 + 16101/5005*P2**2 + 3240/1001*P2*P4 + 42849/17017*P4**2
        else:
            cov = Pell**2

        cov *= 2./Nmodes
        return cov


    def Pell_covariance(self, k, params, ell, dk, volume):
        ell = [ell] if not isinstance(ell, list) else ell
        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        nbin = [x.shape[0] for x in k]
        cov = np.zeros([sum(nbin),sum(nbin)])

        for i in range(len(ell)):
            k_all = k[i] if i==0 else np.hstack((k_all,k[i]))
        k_all = np.unique(k_all)
        if 'f' in self.params_list:
            Pell = self.Pell(k_all, params, ell=[0,2,4])
            Pell[0] += 1./self.nbar/(self.fid_LCDM_params['h'])**3
        else:
            Pell = self.Pell(k_all, params, ell=0) + 1./self.nbar/(self.fid_LCDM_params['h'])**3

        for i,l1 in enumerate(ell):
            for j,l2 in enumerate(ell):
                if j >= i:
                    kij, id1, id2 = np.intersect1d(k[i],k[j],return_indices=True)
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),sum(nbin[:j]):sum(nbin[:j+1])][id1,id2] = self.Gaussian_covariance(l1, l2, k_all, dk, Pell, volume)[np.intersect1d(k_all, kij, return_indices=True)[1]]
                else:
                    cov[sum(nbin[:i]):sum(nbin[:i+1]),sum(nbin[:j]):sum(nbin[:j+1])] = cov[sum(nbin[:j]):sum(nbin[:j+1]),sum(nbin[:i]):sum(nbin[:i+1])].T

        return cov


    def Pell_covariance_LCDM(self, k, params, ell, dk, alpha_tr_lo=None, volume=None, zmin=None, zmax=None, fsky=15000./(360**2/np.pi), volfac=1):
        ell = [ell] if not isinstance(ell, list) else ell
        if not isinstance(k, list):
            k = [np.array(k)]*len(ell)
        elif isinstance(k, list) and len(k) != len(ell):
            raise ValueError("If 'k' is given as a list, it must match the length of 'ell'.")
        else:
            k = [np.array(x) for x in k]

        nbin = [x.shape[0] for x in k]
        cov = np.zeros([sum(nbin),sum(nbin)])

        for i in range(len(ell)):
            k_all = k[i] if i==0 else np.hstack((k_all,k[i]))
        k_all = np.unique(k_all)
        if 'f' in self.params_list:
            Pell = self.Pell_LCDM(k_all, params, ell=[0,2,4], alpha_tr_lo=alpha_tr_lo)
            Pell[0] += 1./self.nbar/(self.fid_LCDM_params['h'])**3
        else:
            Pell = self.Pell_LCDM(k_all, params, ell=0, alpha_tr_lo=alpha_tr_lo) + 1./self.nbar/(self.fid_LCDM_params['h'])**3

        if volume is None:
            Om0 = (self.params['wc']+self.params['wb'])/self.params['h']**2
            H0 = 100*self.params['h']
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

        for i in range(len(ell)):
            k_all = k[i] if i==0 else np.hstack((k_all,k[i]))
        k_all = np.unique(k_all)
        if 'f' in self.params_list:
            Pell = self.Pell_from_table(table, k_all, params, ell=[0,2,4])
            Pell[0] += 1./self.nbar/(self.fid_LCDM_params['h'])**3
        else:
            Pell = self.Pell_from_table(table, k_all, params, ell=0) + 1./self.nbar/(self.fid_LCDM_params['h'])**3

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


    def chi2(self, params, kmax, mode='TEMPLATE', alpha_tr_lo=None):
        if not self.kmax_is_set or (self.kmax != kmax and self.kmax != [kmax for i in range(self.n_ell)]):
            self.set_kmax(kmax)

        Pell_model = np.zeros(sum(self.nbin))
        ell = [2*l for l in range(self.n_ell) if self.nbin[l] > 0]
        if mode == 'TEMPLATE':
            Pell = self.Pell_fid_ktable(params, ell)
        elif mode == 'LCDM':
            Pell = self.Pell_LCDM_fid_ktable(params, ell, alpha_tr_lo=alpha_tr_lo)

        for i,l in enumerate(ell):
            n = int(l/2)

            if self.use_Mpc:
                spline = interp1d(self.k_table, Pell[:,i], kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])
            else:
                spline = interp1d(self.k_table/self.params['h'], Pell[:,i]*self.params['h']**3, kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])

        diff = Pell_model - self.P_data_kmax

        return diff @ self.InvCov_data_kmax @ diff.T


    def chi2_from_table(self, table, params, kmax):
        if not self.kmax_is_set or (self.kmax != kmax and self.kmax != [kmax for i in range(self.n_ell)]):
            self.set_kmax(kmax)

        Pell_model = np.zeros(sum(self.nbin))
        ell = [2*l for l in range(self.n_ell) if self.nbin[l] > 0]
        Pell = self.Pell_from_table_fid_ktable(table, params, ell)

        for i,l in enumerate(ell):
            n = int(l/2)

            if self.use_Mpc:
                spline = interp1d(self.k_table*self.params['h']/self.fid_LCDM_params['h'], Pell[:,i], kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])
            else:
                spline = interp1d(self.k_table/self.fid_LCDM_params['h'], Pell[:,i]*self.params['h']**3, kind='cubic')
                Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])

        diff = Pell_model - self.P_data_kmax

        return diff @ self.InvCov_data_kmax @ diff.T
