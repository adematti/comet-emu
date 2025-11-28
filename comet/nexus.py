import yaml
import numpy as np
import re
from scipy.stats import norm
from astropy.io import fits
from comet import comet

class Nexus:

    def __init__(self):
        self.cosmo_params = ['h','wc','wb','ns','As','w0','wa','Ok','Mnu']
        self.fiducial_values = {}
        self.relations = {}
        self.priors = {}
        self.species = set()
        self.emu = None

    def read_yaml_configuration(self, config):
        self.model = config.get("model")
        self.bias_basis = config.get("bias_basis", 'EggScoSmi')
        if self.bias_basis == 'EggScoSmi':
            bias_params = ['b1','b2','g2','g21']
        elif self.bias_basis == 'AssBauGre':
            bias_params = ['b1','b2','bG2','bGam3']
        else:
            raise ValueError('Bias parametrisation must be either "EggScoSmi" or '
                             '"AssBauGre".')

        self.ctr_noise_basis = config.get("ctr_noise_basis", "Comet")
        if self.ctr_noise_basis == 'Comet':
            ctr_noise_params = ['c0', 'c2', 'c4', 'cnlo', 'NP0', 'NP20', 'NP22']
        elif self.ctr_noise_basis ==  'ClassPT':
            ctr_noise_params = ['c0*', 'c2*', 'c4*', 'cnlo*', 'NP0', 'NP20*',
                                'NP22*']
        elif self.ctr_noise_basis == 'PBJ':
            ctr_noise_params = ['c0t', 'c2t', 'c4t', 'cnlot', 'NP0', 'eps0',
                                'eps2']
        else:
            raise ValueError('Counterterm/noise parametrisation must be either '
                             '"Comet", "ClassPT", or "PBJ".')

        self.nuisance_params = bias_params + ctr_noise_params
        if 'VDG_infty' in self.model:
            self.nuisance_params += ['avir']
        self.nuisance_params += ['sigma','gamma']

        self.use_Mpc = config.get("use_Mpc", True)
        self.use_Planck = config.get("use_Planck", False)
        self.use_BAO = config.get("use_BAO", False)
        self.use_SN = config.get("use_SN", False)
        self.use_Jeffreys = config.get("use_Jeffreys", False)

    def read_yaml_fiducial_cosmology(self, config):
        self.fiducial_cosmology = {}
        for p in config:
            self.fiducial_cosmology[p] = config.get(p)

    def _identify_prior(self, x, p):
        patterns = {}
        #patterns['flat'] = r'U\(([^,]+)\s*,\s*([^)]+)\)'
        patterns['flat'] = r'U\(\s*([^|]+?)\s*\|\s*([^)]+?)\s*\)'
        #patterns['Gaussian'] = r'N\(([^,]+)\s*,\s*([^)]+)\)'
        patterns['Gaussian'] = r'N\(\s*([^|]+?)\s*\|\s*([^)]+?)\s*\)'
        #patterns['AM'] = r'A\(([^,]+)\s*,\s*([^)]+)\)'
        patterns['AM'] = r'A\(\s*([^|]+?)\s*\|\s*([^)]+?)\s*\)'

        if isinstance(x, (int, float)):
            return x, 'fixed'
        elif x in ['LL','coevolution','excursion_set']:
            return None, x
        else:
            for ptype, pattern in patterns.items():
                match = re.match(pattern, x)
                if match:
                    prior = [float(match.group(1)), float(match.group(2))]
                    return prior, ptype
            else:
                raise ValueError(f'Prior for {p} not correctly specified.')

    def read_yaml_parameters(self, config):
        for p in config:
            val, ptype = self._identify_prior(config[p], p)
            if ptype == 'fixed':
                self.fiducial_values[p] = val
            elif ptype in ['LL','coevolution','excursion_set']:
                self.relations[p] = ptype
            else:
                self.priors[p] = {'value':val, 'type':ptype}

    def read_yaml_sampling(self, config):
        self.output_dir = config.get("output_dir", '')
        self.output_fname  = config.get("output_filename")
        self.sampler = config.get("sampler")
        self.n_live = config.get("n_live", 500)
        self.sampling_efficiency = config.get("sampling_efficiency",0.5)
        self.evidence_tolerance  = config.get("evidence_tolerance", 0.4)

    def read_yaml_data(self, config):
        def read_yaml_sample_info(config, container):
            container['fiducials'] = {}
            container['priors'] = {}
            container['relations'] = {}
            container['zerror'] = {}
            val, ptype = self._identify_prior(
                config.get("fraction", 1.0), 'fraction')
            if ptype == 'fixed':
                container['fiducials']['fraction'] = val
            else:
                container['priors']['fraction'] = {'value':val, 'type':ptype}
            container['lambda'] = config.get("lambda", None)
            container['zerror'] = config.get("zerror", None)
            for p in config.get("parameters", {}):
                val, ptype = self._identify_prior(config['parameters'][p], p)
                if ptype == 'fixed':
                    container['fiducials'][p] = val
                elif ptype in ['LL','coevolution','excursion_set']:
                    container['relations'][p] = ptype
                else:
                    container['priors'][p] = {'value':val, 'type':ptype}

        def get_redshift(correct, interloper):
            return correct['lambda']/interloper['lambda']*(1+correct['zeff']) - 1

        self.data_model = config.get("data_model", 'LE3')
        self.mixing_matrix_kp_max_mult = config.get(
            "mixing_matrix_kp_max_multiplier", 1.75)
        self.input_dir = config.get("input_dir", '')

        self.observables = list(config['observables'].keys())
        fractions = ['fraction' in config['observables'][oi]
                     for oi in self.observables]
        self.has_composition = True if any(fractions) else False
        if self.has_composition:
            self.composition = {}
            self.species.update({'correct'})
            if 'fraction' not in self.nuisance_params:
                self.nuisance_params += ['fraction']
        else:
            self.sample = {}

        self.fname_data, self.fname_cov, self.fname_mixing_matrix = {}, {}, {}
        self.stat, self.kmax, self.kmin, self.zeff = {}, {}, {}, {}
        self.nbar, self.fiducial_cosmology_obs = {}, {}

        for oi in self.observables:
            self.fname_data[oi] = config['observables'][oi].get("fname_data")
            self.fname_cov[oi] = config['observables'][oi].get("fname_cov")
            self.fname_mixing_matrix[oi] = config['observables'][oi].get(
                "fname_mixing_matrix", None)
            self.stat[oi] = config['observables'][oi].get("stat")
            self.kmax[oi] = config['observables'][oi].get("kmax")
            self.kmin[oi] = config['observables'][oi].get("kmin", [0,0,0])
            self.zeff[oi] = config['observables'][oi].get("zeff")
            self.nbar[oi] = config['observables'][oi].get("nbar", 1.0)
            self.fiducial_cosmology_obs[oi] = self.fiducial_cosmology.copy()
            self.fiducial_cosmology_obs[oi]['z'] = self.zeff[oi]

            if self.has_composition:
                self.composition[oi] = {'correct':{}}
                read_yaml_sample_info(config['observables'][oi],
                                      self.composition[oi]['correct'])
                self.composition[oi]['correct']['zeff'] = self.zeff[oi]
                for species in config['observables'][oi].get(
                        "interlopers", {}):
                    self.species.update({species})
                    self.composition[oi][species] = {}
                    read_yaml_sample_info(
                        config['observables'][oi]['interlopers'][species],
                        self.composition[oi][species])
                    self.composition[oi][species]['zeff'] = get_redshift(
                        self.composition[oi]['correct'],
                        self.composition[oi][species]
                    )
            else:
                self.sample[oi] = {}
                read_yaml_sample_info(config['observables'][oi],
                                      self.sample[oi])

        if self.has_composition:
            self.n_obs = {s:0 for s in self.species}
            self.obs_id = {s:{} for s in self.species}
            for s in self.species:
                for oi in self.observables:
                    if s in self.composition[oi]:
                        self.obs_id[s][oi] = np.copy(self.n_obs[s])
                        self.n_obs[s] += 1
        else:
            self.composition = {oi:None for oi in self.observables}
            self.n_obs = len(self.observables)
            self.obs_id = {}
            for i,oi in enumerate(self.observables):
                self.obs_id[oi] = i

    def _create_params_setup(self):
        # create dictionaries of all sampled and fixed parameters
        self.sampled_cosmo_params = {}
        self.fixed_cosmo_params = {}
        self.sampled_nuisance_params = {}
        self.fixed_nuisance_params = {}
        self.relation_nuisance_params = {}
        self.AM_priors = {oi:{} for oi in self.observables}

        # cosmological parameters with prior information -> sampled (unless AM)
        for p in self.priors:
            if p in self.cosmo_params:
                if self.priors[p]['type'] != 'AM':
                    self.sampled_cosmo_params[p] = {
                        'prior':self.priors[p]['value'],
                        'type':self.priors[p]['type']
                    }
        # by default assign fiducial cosmology as fixed parameters
        for p in self.fiducial_cosmology:
            if p in self.cosmo_params and p not in self.priors:
                self.fixed_cosmo_params[p] = self.fiducial_cosmology[p]
        # overwrite fiducial cosmology if fixed cosmological parameters have been
        # given in "Parameters" section (note: this does not affect fiducial
        # cosmology for computation of AP distortions)
        for p in self.fiducial_values:
            if p in self.cosmo_params and p not in self.priors:
                self.fixed_cosmo_params[p] = self.fiducial_values[p]
        self.n_cosmo_params = len(self.sampled_cosmo_params)

        if 'wa' in self.sampled_cosmo_params or 'wa' in self.fixed_cosmo_params:
            self.de_model = 'w0wa'
        elif 'w0' in self.sampled_cosmo_params or 'w0' in self.fixed_cosmo_params:
            self.de_model = 'w0'
        else:
            self.de_model = 'lambda'

        nonu = False
        if 'Mnu' not in self.sampled_cosmo_params:
            if 'Mnu' not in self.fixed_cosmo_params:
                nonu = True
            elif self.fixed_cosmo_params['Mnu'] == 0.0:
                nonu = True
        if nonu and 'nonu' not in self.model:
            self.model += '_nonu'

        if self.has_composition:
            for oi in self.observables:
                for s in self.composition[oi]:
                    self.AM_priors[oi][s] = {}
            for p in self.nuisance_params:
                for oi in self.observables:
                    for s in self.composition[oi]:
                        pos = f'{p}.{oi}.{s}'
                        priors_os = self.composition[oi][s]['priors']
                        fixed_os = self.composition[oi][s]['fiducials']
                        relation_os = self.composition[oi][s]['relations']
                         # first check if sample specific information was given,
                         # otherwise resort to global information
                        if p in priors_os:
                            if priors_os[p]['type'] != 'AM':
                                self.sampled_nuisance_params[pos] = {
                                    'prior':priors_os[p]['value'],
                                    'type':priors_os[p]['type']
                                }
                            else:
                                self.AM_priors[oi][s][p] = priors_os[p]['value']
                        elif p in fixed_os:
                            self.fixed_nuisance_params[pos] = fixed_os[p]
                        elif p in relation_os:
                            self.relation_nuisance_params[pos] = relation_os[p]
                        elif p in self.priors:
                            if self.priors[p]['type'] != 'AM':
                                self.sampled_nuisance_params[pos] = {
                                    'prior':self.priors[p]['value'],
                                    'type':self.priors[p]['type']
                                }
                            else:
                                self.AM_priors[oi][s][p] = self.priors[p]['value']
                        elif p in self.fiducial_values:
                            self.fixed_nuisance_params[pos] = \
                                self.fiducial_values[p]
                        elif p in self.relations:
                            self.relation_nuisance_params[pos] = self.relations[p]
        else:
            for p in self.nuisance_params:
                for oi in self.observables:
                    po = f'{p}.{oi}'
                    priors_o = self.sample[oi]['priors']
                    fixed_o = self.sample[oi]['fiducials']
                    relation_o = self.sample[oi]['relations']
                    if p in priors_o:
                        if priors_o[p]['type'] != 'AM':
                            self.sampled_nuisance_params[po] = {
                                'prior':priors_o[p]['value'],
                                'type':priors_o[p]['type']
                            }
                        else:
                            self.AM_priors[oi][p] = priors_o[p]['value']
                    elif p in fixed_o:
                        self.fixed_nuisance_params[po] = fixed_o[p]
                    elif p in relation_o:
                        self.relation_nuisance_params[po] = relation_o[p]
                    elif p in self.priors:
                        if self.priors[p]['type'] != 'AM':
                            self.sampled_nuisance_params[po] = {
                                'prior':self.priors[p]['value'],
                                'type':self.priors[p]['type']
                            }
                        else:
                            self.AM_priors[oi][p] = self.priors[p]['value']
                    elif p in self.fiducial_values:
                        self.fixed_nuisance_params[po] = \
                            self.fiducial_values[p]
                    elif p in self.relations:
                        self.relation_nuisance_params[po] = self.relations[p]
        self.n_nuisance_params = len(self.sampled_nuisance_params)
        self.n_params_total = self.n_cosmo_params + self.n_nuisance_params

        self.n_AM_params = 0
        for oi in self.AM_priors:
            if self.has_composition:
                for s in self.AM_priors[oi]:
                    for p in self.AM_priors[oi][s]:
                        self.n_AM_params += 1
            else:
                for p in self.AM_priors[oi]:
                    self.n_AM_params += 1

    def read_yaml(self, obj, type='file'):
        if type in ['f', 'file']:
            with open(obj, 'r') as f:
                config = yaml.safe_load(f)
        elif type in ['t','text']:
            config = yaml.safe_load(obj)

        if 'Configuration' in config:
            self.read_yaml_configuration(config['Configuration'])
        if 'FiducialCosmology' in config:
            self.read_yaml_fiducial_cosmology(config['FiducialCosmology'])
        if 'Parameters' in config:
            self.read_yaml_parameters(config['Parameters'])
        if 'Sampling' in config:
            self.read_yaml_sampling(config['Sampling'])
        if 'Data' in config:
            self.read_yaml_data(config['Data'])

    def _data_LE3_to_Comet(self, datafile, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        k = datafile['spectrum'].data['k']
        k_eff = datafile['spectrum'].data['k_eff']
        data = np.stack(
            [datafile['spectrum'].data['PK{}'.format(l)] for l in ell],
            axis=-1
        )
        return k, k_eff, data

    def _cov_LE3_to_Comet(self, covfile, ell):
        ell = [ell] if not isinstance(ell, list) else ell
        n_bin = covfile['covariance'].header['n_bin']
        n_ell = len(ell)
        n_all = covfile['covariance'].data['KI'].shape[0]
        ids_all = np.arange(n_all)
        i = ids_all[np.isin(covfile['covariance'].data['multipole-i'], ell)]
        j = ids_all[np.isin(covfile['covariance'].data['multipole-j'], ell)]
        ij = np.intersect1d(i,j)
        cov = covfile['covariance'].data['covariance'][ij].reshape(
            (n_bin*n_ell, n_bin*n_ell))
        return cov

    def _mixing_matrix_LE3_to_Comet(self, wfile, ell, k_max, kp_max):
        k = wfile['bins_output'].data['k']
        kp = wfile['bins_input'].data['kp0']
        i_max = sum(k < k_max)
        ip_max = sum(kp < kp_max)
        n_ell = len(ell)
        w_all = np.empty((n_ell*i_max,n_ell*ip_max))
        for i,l in enumerate(ell):
            for j,lp in enumerate(ell):
                w_all[i*i_max:(i+1)*i_max, j*ip_max:(j+1)*ip_max] = \
                    wfile['mixing_matrix'].data['W{}{}'.format(l,lp)][:i_max,:ip_max]
        return k[:i_max], kp[:ip_max], w_all

    def init_comet(self):
        self._create_params_setup()
        if self.emu is None or self.emu.model != self.model \
                or self.emu.use_Mpc != self.use_Mpc \
                or self.emu.bias_basis != self.bias_basis \
                or self.emu.counterterm_basis != self.ctr_noise_basis:
            self.emu = comet(model=self.model, use_Mpc=self.use_Mpc,
                            bias_basis=self.bias_basis,
                            counterterm_basis=self.ctr_noise_basis)

        for oi in self.observables:
            if self.data_model == 'LE3':
                data = fits.open(f'{self.input_dir}/{self.fname_data[oi]}')
                cov = fits.open(f'{self.input_dir}/{self.fname_cov[oi]}')
                k, k_eff, data_comet = self._data_LE3_to_Comet(data, [0,2,4])
                cov_comet = self._cov_LE3_to_Comet(cov, [0,2,4])
                if self.fname_mixing_matrix[oi] is not None:
                    mixing_matrix = fits.open(
                        f'{self.input_dir}/{self.fname_mixing_matrix[oi]}')
                    mm_k, mm_kp, mm_comet = self._mixing_matrix_LE3_to_Comet(
                        mixing_matrix, [0,2,4], np.amax(self.kmax[oi]),
                        np.amax(self.kmax[oi])*self.mixing_matrix_kp_max_mult
                    )
                    self.emu.define_data_set(
                        obs_id=oi, stat=self.stat[oi], zeff=self.zeff[oi],
                        bins=k_eff, signal=data_comet, cov=cov_comet,
                        bins_mixing_matrix=[mm_k,mm_kp],
                        W_mixing_matrix=mm_comet,
                        fiducial_cosmology=self.fiducial_cosmology_obs[oi],
                        composition=self.composition[oi], nbar=self.nbar[oi])
                else:
                    self.emu.define_data_set(
                        obs_id=oi, stat=self.stat[oi], zeff=self.zeff[oi],
                        bins=k_eff, signal=data_comet, cov=cov_comet,
                        fiducial_cosmology=self.fiducial_cosmology[oi],
                        composition=self.composition[oi], nbar=self.nbar[oi])
            self.emu.data[oi].set_kmax(self.kmax[oi], self.kmin[oi])

    def _g2bG2_relation(self, relation, b1):
        if relation == 'LL' or relation == 'coevolution':
            return - 2./7 * (b1 - 1)
        elif relation == 'excursion_set':
            return 0.524 - 0.547*b1 + 0.046*b1**2

    def _g21bGam3_relation(self, relation, b1, g2bG2):
        if relation == 'coevolution':
            if self.bias_basis == 'EggScoSmi':
                return 2.0/21.0 * (b1 - 1) + 6.0/7.0 * g2bG2
            elif self.bias_basis == 'AssBauGre':
                return -1./6. * (b1 - 1) - 5./2. * g2bG2

    def _assign_params(self, cube):
        n = 0
        if self.has_composition:
            params = {s:{} for s in self.species}
            for p in self.sampled_cosmo_params:
                for s in self.species:
                    params[s][p] = np.repeat(cube[n], self.n_obs[s])
                n += 1
            for p in self.fixed_cosmo_params:
                for s in self.species:
                    params[s][p] = np.repeat(self.fixed_cosmo_params[p],
                                             self.n_obs[s])
            for pos in self.sampled_nuisance_params:
                p, oi, s = pos.split('.')
                noi = self.obs_id[s][oi]
                if p not in params[s]:
                    params[s][p] = np.zeros(self.n_obs[s])
                params[s][p][noi] = cube[n]
                n += 1
            for pos in self.fixed_nuisance_params:
                p, oi, s = pos.split('.')
                noi = self.obs_id[s][oi]
                if p not in params[s]:
                    params[s][p] = np.zeros(self.n_obs[s])
                params[s][p][noi] = self.fixed_nuisance_params[pos]
            for pos in self.relation_nuisance_params:
                p, oi, s = pos.split('.')
                noi = self.obs_id[s][oi]
                if p not in params[s]:
                    params[s][p] = np.zeros(self.n_obs[s])
                rel = self.relation_nuisance_params[pos]
                if p in ['g2','bG2']:
                    params[s][p][noi] = self._g2bG2_relation(
                        rel, params[s]['b1'][noi])
                elif p in ['g21','bGam3']:
                    if self.bias_basis == 'EggScoSmi':
                        params[s][p][noi] = self._g21bGam3_relation(
                            rel, params[s]['b1'][noi], params[s]['g2'][noi])
                    elif self.bias_basis == 'AssBauGre':
                        params[s][p][noi] = self._g21bGam3_relation(
                            rel, params[s]['b1'][noi], params[s]['bG2'][noi])
            for s in self.species:
                params[s]['z'] = np.array([self.composition[oi][s]['zeff']
                                           for oi in self.observables
                                           if s in self.composition[oi]])
        else:
            params = {}
            for p in self.sampled_cosmo_params:
                params[p] = np.repeat(cube[n], self.n_obs)
                n += 1
            for p in self.fixed_cosmo_params:
                params[p] = np.repeat(self.fixed_cosmo_params[p], self.n_obs)
            for po in self.sampled_nuisance_params:
                p, oi = po.split('.')
                noi = self.obs_id[oi]
                if p not in params:
                    params[p] = np.zeros(self.n_obs)
                params[p][noi] = cube[n]
                n += 1
            for po in self.fixed_nuisance_params:
                p, oi = po.split('.')
                noi = self.obs_id[oi]
                if p not in params:
                    params[p] = np.zeros(self.n_obs)
                params[p][noi] = self.fixed_nuisance_params[po]
            for po in self.relation_nuisance_params:
                p, oi = po.split('.')
                noi = self.obs_id[oi]
                if p not in params:
                    params[p] = np.zeros(self.n_obs)
                rel = self.relation_nuisance_params[po]
                if p in ['g2','bG2']:
                    params[p][noi] = self._g2bG2_relation(
                        rel, params['b1'][noi])
                elif p in ['g21','bGam3']:
                    if self.bias_basis == 'EggScoSmi':
                        params[p][noi] = self._g21bGam3_relation(
                            rel, params['b1'][noi], params['g2'][noi])
                    elif self.bias_basis == 'AssBauGre':
                        params[p][noi] = self._g21bGam3_relation(
                            rel, params['b1'][noi], params['bG2'][noi])
            params['z'] = np.array([self.zeff[oi] for oi in self.observables])
        return params

    def _generate_prior_loglik(self):
        if self.sampler == 'multinest':
            from scipy.special import erfinv
            def prior(cube):
                n = 0
                for p in self.sampled_cosmo_params:
                    if self.sampled_cosmo_params[p]['type'] == 'flat':
                        a_min, a_max = self.sampled_cosmo_params[p]['prior']
                        cube[n] = a_min + (a_max - a_min) * cube[n]
                    else:
                        mean, std = self.sampled_cosmo_params[p]['prior']
                        cube[n] = mean + std * np.sqrt(2) * erfinv(2*cube[n]-1.0)
                    n += 1
                for p in self.sampled_nuisance_params:
                    if self.sampled_nuisance_params[p]['type'] == 'flat':
                        a_min, a_max = self.sampled_nuisance_params[p]['prior']
                        cube[n] = a_min + (a_max - a_min) * cube[n]
                    else:
                        mean, std = self.sampled_nuisance_params[p]['prior']
                        cube[n] = mean + std * np.sqrt(2) * erfinv(2*cube[n]-1.0)
                    n += 1
                return cube

            def loglik(cube):
                params = self._assign_params(cube)
                chi2 = self.emu.chi2(self.observables, params, self.kmax,
                                     self.de_model, AM_priors=self.AM_priors)
                return -0.5 * chi2

            return prior, loglik

    def run_chain(self, resume=False, verbose=True):
        if self.sampler == 'multinest':
            import pymultinest
            prior, loglik = self._generate_prior_loglik()
            pymultinest.solve(
                loglik, prior, self.n_params_total,
                outputfiles_basename=f'{self.output_dir}/{self.output_fname}',
                resume=resume, verbose=verbose, n_live_points=self.n_live,
                sampling_efficiency=self.sampling_efficiency,
                evidence_tolerance=self.evidence_tolerance
            )
