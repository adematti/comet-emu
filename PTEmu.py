import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.integrate import quad
from scipy.special import beta, betainc
import GPy
import pickle
from pyDOE import *
from colossus.cosmology import cosmology


class PTEmu_tables:
    def __init__(self, params, validation=False):
        self.params = params
        self.n_params = len(params)
        self.param_ranges = None
        self.validation = validation
        self.model = None
        self.model_transformed = None

        self.nk = 76
        self.nkloop = 56
        self.n_diagrams = 20

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


    def load_table(self, fname, data_type=None):
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
                        elif i == 10:
                            self.model[j, :, 1+self.n_diagrams*l+1] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 14:
                            self.model[j, :, 1+self.n_diagrams*l+2] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 15:
                            self.model[j, :, 1+self.n_diagrams*l+3] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 16:
                            self.model[j, :, 1+self.n_diagrams*l+1] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]
                        elif i == 17:
                            self.model[j, :, 1+self.n_diagrams*l+2] += temp[j * self.nk:(j + 1) * self.nk, 1+25*l+i]

        if not self.validation:
            self.transform_training_data(data_type=data_type)


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
            if self.model_transformed is None:
                self.model_transformed = {}
                self.mean   = {}
                self.std    = {}
                self.flip   = {}
                self.offset = {}
            self.model_transformed[data_type] = self.transform(self.model[data_type], data_type=data_type)
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


    def GPy_model(self, data_type):
        kernel = GPy.kern.RBF(input_dim=self.n_params, variance=np.var(self.model_transformed[data_type]), lengthscale=np.ones(self.n_params), ARD=True)
        return GPy.models.GPRegression(self.samples, self.model_transformed[data_type], kernel)



class PTEmu:
    def __init__(self, params, fid_LCDM_params={'wc':0.11544,'wb':0.0222191,'ns':0.9632,'h':0.695, 'As':2.2078559, 'z':1.0}):

        self.params_shape_list = [key for key,val in params.items() if 'shape' in val]
        self.params_list = [p for p in params.keys()]
        self.bias_params_list = ['b1','b2','g2','g21','c0','c2','c4','cnlo','N0','N20','N22']

        self.params = {p:0. for p in self.params_list+self.bias_params_list}
        self.fid_LCDM_params = fid_LCDM_params

        self.nk = 76
        self.nkloop = 56
        self.n_diagrams = 20
        self.kHD = 0.4
        self.kmax_is_set = False

        self.training   = {}
        self.validation = {}

        self.training['shape']   = PTEmu_tables(self.params_shape_list)
        self.training['all']     = PTEmu_tables(self.params_list)
        self.validation['shape'] = PTEmu_tables(self.params_shape_list, validation=True)
        self.validation['all']   = PTEmu_tables(self.params_list, validation=True)

        self.emu = {}

        # self.LCDM     = cosmology.setCosmology('planck18', {'print_warnings': False, 'persistence':''})
        # self.LCDM.H0  = 100*self.fid_LCDM_params['h']
        # self.LCDM.Om0 = (self.fid_LCDM_params['wc']+self.fid_LCDM_params['wb'])/self.fid_LCDM_params['h']**2
        # self.LCDM.Ob0 = self.fid_LCDM_params['wb']/self.fid_LCDM_params['h']**2
        # self.LCDM.ns  = self.fid_LCDM_params['ns']
        # self.LCDM.checkForChangedCosmology()


    def generate_samples(self, type, ranges, n_samples, n_trials=0, validation=False):
        if validation:
            self.validation[type].geneate_samples(ranges, n_samples, n_trials)
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


    def load_table(self, type, fname, data_type=None, validation=False):
        self.k_table = np.loadtxt('../tables/k_vector.dat')
        if validation:
            self.validation[type].load_table(fname, data_type=data_type)
        else:
            self.training[type].load_table(fname, data_type=data_type)


    def train_emulator(self, max_f_eval=1000, num_restarts=5, data_type=None):
        if data_type is None:
            self.emu['PL'] = self.training['shape'].GPy_model('PL')
            self.emu['s12'] = self.training['shape'].GPy_model('s12')
            for ell in [0,2,4]:
                self.emu[ell] = self.training['all'].GPy_model(ell)

            for dt in self.emu.keys():
                self.emu[dt].optimize(max_f_eval=max_f_eval)
                self.emu[dt].optimize_restarts(num_restarts=num_restarts)
        else:
            data_type = [data_type] if not isinstance(data_type, list) else data_type
            for dt in data_type:
                if dt in ['PL','s12']:
                    self.emu[dt] = self.training['shape'].GPy_model(dt)
                else:
                    self.emu[dt] = self.training['all'].GPy_model(dt)
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


    def define_data_set(self, k, obs, cov, nbar, theory_cov=False, Nrealizations=300):
        self.k_data = k
        self.P_data = obs if obs.ndim > 1 else obs[:,None]
        self.n_ell = self.P_data.shape[1]
        self.Cov_data = cov
        self.nbar = nbar
        self.theory_cov = theory_cov
        self.Nrealizations = Nrealizations


    def AHfactor(self, nbin):
        return 1. if self.theory_cov else (self.Nrealizations - nbin -2)*1./(self.Nrealizations - 1)


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
            self.P_data_kmax = np.concatenate((self.P_data_kmax,self.P_data[:self.nbin[l],l])) if self.P_data_kmax.size else self.P_data[:self.nbin[l],l]

        self.Cov_data_kmax = np.zeros([sum(self.nbin),sum(self.nbin)])
        for l1 in range(self.n_ell):
            for l2 in range(self.n_ell):
                self.Cov_data_kmax[sum(self.nbin[:l1]):sum(self.nbin[:l1+1]),sum(self.nbin[:l2]):sum(self.nbin[:l2+1])] = self.Cov_data[l1*nbin_total:l1*nbin_total+self.nbin[l1],l2*nbin_total:l2*nbin_total+self.nbin[l2]]
        self.InvCov_data_kmax = self.AHfactor(sum(self.nbin))*np.linalg.inv(self.Cov_data_kmax)

        self.kmax_is_set = True


    def Ez(self, z, Om0):
        return np.sqrt(Om0*(1+z)**3+1-Om0)


    def oneOverEz(self, z, Om0):
        return 1./self.Ez(z, Om0)


    def Hz(self, z, Om0, H0):
        return H0*self.Ez(z, Om0)


    def angularDiameterDistance(self, z, Om0):
        return 2.998E3*quad(self.oneOverEz, 0, z, args=Om0)[0]


    def growthFactor(self, z, Om0):
        Ode0 = 1.-Om0
        zp1 = 1./(1+z)**3
        return 5./6*betainc(5./6,2./3,Ode0*zp1/(Om0+Ode0*zp1))*(Om0/Ode0)**(1./3)*np.sqrt(1 + Om0/(Ode0*zp1))*beta(5./6,2./3)


    def growthRate(self, z, Om0):
        return 5./2*Om0*((1+z)/self.Ez(z,Om0))**2*(1./self.growthFactor(z, Om0)-3./5*(1+z))


    def define_fiducial_cosmology(self, params_fid=None, HDm_fid=None):
        if HDm_fid is not None:
            self.H_fid = HDm_fid[0]
            self.Dm_fid = HDm_fid[1]
        else:
            # self.update_LCDM_h_wc_wb(params_fid['h'], params_fid['wc'], params_fid['wb'])
            # self.H_fid = self.LCDM.Hz(params_fid['z'])
            # self.Dm_fid = self.LCDM.angularDiameterDistance(params_fid['z'])*(1 + params_fid['z'])/params_fid['h']
            Om0 = (params_fid['wc']+params_fid['wb'])/params_fid['h']**2
            H0 = params_fid['h']*100
            self.H_fid = self.Hz(params_fid['z'], Om0, H0)
            self.Dm_fid = self.angularDiameterDistance(params_fid['z'], Om0)/params_fid['h']


    def update_LCDM_h_wc_wb(self, h, wc, wb):
        self.LCDM.H0 = 100*h
        self.LCDM.Om0 = (wc + wb)/h**2
        self.LCDM.Ob0 = wb/h**2
        self.LCDM.checkForChangedCosmology()


    def update_params(self, params, flag):
        try:
            if flag == 'generic':
                for p in self.params_list:
                    self.params[p] = params[p]
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


    def get_bias_coeff(self, ell):
        b1   = self.params['b1']
        b2   = self.params['b2']
        g2   = self.params['g2']
        g21  = self.params['g21']
        cell = self.params['c{}'.format(ell)]
        cnlo = self.params['cnlo']
        N0   = self.params['N0']
        N20  = self.params['N20']
        N22  = self.params['N22']
        return np.array([b1**2, b1, 1., cell/self.kHD**2, b1**2*cnlo/self.kHD**4, b1*cnlo/self.kHD**4,
                         cnlo/self.kHD**4, N0/self.nbar, N20/self.nbar/self.kHD**2, N22/self.nbar/self.kHD**2,
                         b1**2, b1*b2, b1*g2, b1*g21, b2**2, b2*g2, g2**2, b2, g2, g21])


    def Pell(self, params, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        self.update_params(params, 'generic')
        params_shape = np.array([self.params[p] for p in self.params_shape_list])
        params_all   = np.array([self.params[p] for p in self.params_list])

        Pk_lin = self.training['shape'].transform_inv(self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')
        sigma12 = self.training['shape'].transform_inv(self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')

        Pk_lin *= (self.params['s12']/sigma12)**2

        Pell_list = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)
            Pk_ratios = self.training['all'].transform_inv(self.emu[l].predict(params_all[None,:])[0][0], l)

            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            for n in range(7):
                Pk_bij[:,n] = Pk_ratios[n*self.nk:(n+1)*self.nk]*Pk_lin
            for n in range(7,10):
                Pk_bij[:,n] = Pk_ratios[n*self.nk:(n+1)*self.nk]
            for n in range(10):
                Pk_bij[(self.nk-self.nkloop):,10+n] = Pk_ratios[10*self.nk+n*self.nkloop:10*self.nk+(n+1)*self.nkloop]*Pk_lin[(self.nk-self.nkloop):]

            Pell_list[:,i] = np.dot(bij,Pk_bij.T)

        return Pell_list if len(ell) > 1 else Pell_list[:,0]


    def Pell_LCDM(self, params, ell, alpha_tr_lo=None):
        ell = [ell] if not isinstance(ell, list) else ell
        self.update_params(params, 'LCDM')
        params_shape = np.array([self.params[p] for p in self.params_shape_list])

        Pk_lin = self.training['shape'].transform_inv(self.emu['PL'].predict(params_shape[None,:])[0][0], 'PL')
        sigma12 = self.training['shape'].transform_inv(self.emu['s12'].predict(params_shape[None,:])[0][0], 's12')

        # compute growth factors corresponding to fiducial and target parameters
        Om0  = (params['wc']+params['wb'])/params['h']**2
        Dfid = self.growthFactor(self.fid_LCDM_params['z'], (params['wc']+params['wb'])/self.fid_LCDM_params['h']**2)
        D    = self.growthFactor(params['z'], Om0)
        # self.update_LCDM_h_wc_wb(self.fid_LCDM_params['h'], params['wc'], params['wb'])
        # Dfid = self.LCDM.growthFactorUnnormalized(self.fid_LCDM_params['z'])
        # self.update_LCDM_h_wc_wb(params['h'], params['wc'], params['wb'])
        # D = self.LCDM.growthFactorUnnormalized(params['z'])

        # compute AP parameters and growth rate
        if alpha_tr_lo is None:
            alpha_lo = self.H_fid/self.Hz(params['z'], Om0, 100*params['h'])
            alpha_tr = self.angularDiameterDistance(params['z'], Om0)/params['h']/self.Dm_fid
        else:
            alpha_lo = alpha_tr_lo[1]
            alpha_tr = alpha_tr_lo[0]
        f = self.growthRate(params['z'], Om0)
        # alpha_lo = self.H_fid/self.LCDM.Hz(params['z'])
        # alpha_tr = self.LCDM.angularDiameterDistance(params['z'])*(1+params['z'])/params['h']/self.Dm_fid
        # f = - (1.+params['z'])*self.LCDM.growthFactor(params['z'], derivative=1)/self.LCDM.growthFactor(params['z'])

        # rescale linear power spectrum and sigma12
        Pk_lin *= params['As']/self.fid_LCDM_params['As']*(D/Dfid)**2
        sigma12 *= np.sqrt(params['As']/self.fid_LCDM_params['As'])*(D/Dfid)

        params_all = np.concatenate((params_shape, np.array([sigma12, alpha_tr, alpha_lo, f])))
        # print(params_all)

        Pell_list = np.zeros([self.nk,len(ell)])
        for i,l in enumerate(ell):
            bij = self.get_bias_coeff(l)

            # rescale nbar and kHD for change in h
            bij[3] *= (self.fid_LCDM_params['h']/params['h'])**2
            bij[4:7] *= (self.fid_LCDM_params['h']/params['h'])**4
            bij[7] *= (self.fid_LCDM_params['h']/params['h'])**3
            bij[8:10] *= (self.fid_LCDM_params['h']/params['h'])**5

            Pk_ratios = self.training['all'].transform_inv(self.emu[l].predict(params_all[None,:])[0][0], l)

            Pk_bij = np.zeros([self.nk,self.n_diagrams])
            for n in range(7):
                Pk_bij[:,n] = Pk_ratios[n*self.nk:(n+1)*self.nk]*Pk_lin
            for n in range(7,10):
                Pk_bij[:,n] = Pk_ratios[n*self.nk:(n+1)*self.nk]
            for n in range(10):
                Pk_bij[(self.nk-self.nkloop):,10+n] = Pk_ratios[10*self.nk+n*self.nkloop:10*self.nk+(n+1)*self.nkloop]*Pk_lin[(self.nk-self.nkloop):]

            Pell_list[:,i] = np.dot(bij,Pk_bij.T)

        return Pell_list if len(ell) > 1 else Pell_list[:,0]


    def Pell_from_table(self, table, params, ell):
        for p in params.keys():
            self.params[p] = params[p]
        bij = self.get_bias_coeff(ell)

        Pk_bij = np.zeros([self.nk,self.n_diagrams])
        cnt = 0
        for n in np.array([0,1,2,13,14,15,16,17,18,19])+10*ell:
            Pk_bij[:,cnt] = table[:,1+n]
            cnt += 1
        for n in np.arange(3,13)+10*ell:
            Pk_bij[:,cnt] = table[:,1+n]
            cnt += 1
        return np.dot(bij,Pk_bij.T)


    def chi2(self, params, kmax, mode='generic', alpha_tr_lo=None):
        if not self.kmax_is_set or (self.kmax != kmax and self.kmax != [kmax for i in range(self.n_ell)]):
            self.set_kmax(kmax)

        Pell_model = np.zeros(sum(self.nbin))
        ell = [2*l for l in range(self.n_ell) if self.nbin[l] > 0]
        if mode == 'generic':
            Pell = self.Pell(params, ell)
        elif mode == 'LCDM':
            Pell = self.Pell_LCDM(params, ell, alpha_tr_lo=alpha_tr_lo)

        if Pell.ndim == 1:
            Pell = Pell[:,None]

        for i,l in enumerate(ell):
            n = int(l/2)
            spline = interp1d(self.k_table*self.fid_LCDM_params['h'], Pell[:,i], kind='cubic')
            Pell_model[sum(self.nbin[:n]):sum(self.nbin[:n+1])] = spline(self.k_bins[n])
        diff = Pell_model - self.P_data_kmax

        return diff @ self.InvCov_data_kmax @ diff.T
