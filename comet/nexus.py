import yaml
import numpy as np
import re, os
from scipy.stats import norm
from astropy.io import fits
from comet import comet
from pyfiglet import Figlet

def boldtext(text):
    return f"\033[1m{text}\033[0m"

def is_flat_dict(d):
    return not any(isinstance(v, dict) for v in d.values())

def pretty_nested(d, indent=2, level=1):
    pad = " " * (indent * level)
    lines = []

    for k, v in d.items():
        if isinstance(v, dict) and not is_flat_dict(v):
            lines.append(f"{pad}{boldtext(k)}:")
            lines.append(pretty_nested(v, indent, level+1))
        else:
            lines.append(f"{pad}{boldtext(k)}: {v}")

    return "\n".join(lines)


class Nexus:

    def __init__(self):

        f = Figlet(font='standard')
        print(f.renderText('Welcome to NEXUS'))

        self.cosmo_params = ['wc', 'wb', 'ns', 'Mnu',
                             'h', 'As', 'w0', 'wa', 'Ok']
        self.fiducial_values = {}
        self.relations = {}
        self.priors = {}
        self.species = set()
        self.emu = None

    def read_yaml_configuration(self, config):
        r"""Reader/handler of main configurations

        Parameters
        ----------
        config: dict
            Configuration dictionary
        """
        print (boldtext("Reading configurations...\n"))

        expected_keys = {
            "model": None,
            "bias_basis": "EggScoSmi",
            "ctr_noise_basis": "Comet",
            "use_Mpc": True,
            "use_Planck": False,
            "use_BAO": False,
            "use_SN": False,
            "use_Jeffreys": False
        }

        maxlen = max(len(k) for k in expected_keys.keys())

        for key, default in expected_keys.items():
            value = config.get(key, default)
            setattr(self, key, value)
            print(f"{boldtext(key.ljust(maxlen))} = {value}")

        if self.bias_basis == 'EggScoSmi':
            bias_params = ['b1', 'b2', 'g2', 'g21']
        elif self.bias_basis == 'AssBauGre':
            bias_params = ['b1', 'b2', 'bG2', 'bGam3']
        else:
            raise ValueError('Bias parametrisation must be either "EggScoSmi" '
                             'or "AssBauGre".')

        if self.ctr_noise_basis == 'Comet':
            ctr_noise_params = ['c0', 'c2', 'c4', 'NP0', 'NP20', 'NP22']
        elif self.ctr_noise_basis ==  'ClassPT':
            ctr_noise_params = ['c0*', 'c2*', 'c4*', 'NP0', 'NP20*', 'NP22*']
        elif self.ctr_noise_basis == 'PBJ':
            ctr_noise_params = ['c0t', 'c2t', 'c4t', 'NP0', 'eps0', 'eps2']
        else:
            raise ValueError('Counterterm and noise parametrisation must be '
                             'either "Comet", "ClassPT", or "PBJ".')

        self.nuisance_params = bias_params + ctr_noise_params
        if self.model == 'EFT':
            if self.ctr_noise_basis == 'Comet':
                self.nuisance_params += ['cnlo']
            elif self.ctr_noise_basis == 'ClassPT':
                self.nuisance_params += ['cnlo*']
            elif self.ctr_noise_basis == 'PBJ':
                self.nuisance_params += ['cnlot']
        if self.model == 'VDG_infty':
            self.nuisance_params += ['avir']
        self.nuisance_params += ['sigma', 'gamma']
        print (boldtext("\n↳ nuisance parameters ="), self.nuisance_params)

        print ()

    def read_yaml_fiducial_cosmology(self, config):
        r"""Reader/handler of fiducial cosmology

        Parameters
        ----------
        config: dict
            Fiducial cosmology dictionary
        """
        print (boldtext("Reading fiducial cosmology...\n"))

        maxlen = max(len(p) for p in self.cosmo_params)

        self.fiducial_cosmology = {}
        for key in self.cosmo_params:
            value = config.get(key)
            self.fiducial_cosmology[key] = value
            print(f"{boldtext(key.ljust(maxlen))} = {value}")

        print ()

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
        r"""Reader/handler of priors

        Parameters
        ----------
        config: dict
            Priors dictionary
        """
        print (boldtext("Reading priors...\n"))

        for p, cfg_val in config.items():
            val, ptype = self._identify_prior(cfg_val, p)

            if ptype == 'fixed':
                self.fiducial_values[p] = val
                msg = f"fixed to {val}"
            elif ptype in ['LL', 'coevolution', 'excursion_set']:
                self.relations[p] = ptype
                msg = f"fixed to {ptype.replace('_', ' ')} relation"
            else:
                self.priors[p] = {'value': val, 'type': ptype}
                msg = f"{ptype} {val}"

            print(f"{boldtext(p.ljust(5) + '→')} {msg}")

        print ()

    def read_yaml_sampling(self, config):
        r"""Reader/handler of sampler setup

        Parameters
        ----------
        config: dict
            Sampler dictionary
        """
        print(boldtext("Reading sampler setup...\n"))

        expected_keys = {
            "common": {
                "output_dir": '',
                "output_filename": None,
                "sampler": None,
                "n_live": 500,
            },
            "multinest": {
                "sampling_efficiency": 0.5,
                "evidence_tolerance": 0.4,
            },
            "nautilus": {
                "f_live": 0.01,
                "n_eff": 10000,
                "pool": None
            }
        }

        sampler = config.get("sampler")

        cdict = expected_keys["common"].copy()
        cdict.update(expected_keys.get(sampler, {}))
        maxlen = max(len(k) for k in cdict.keys())

        for key, default in cdict.items():
            value = config.get(key, default)
            setattr(self, key, value)
            print(f"{boldtext(key.ljust(maxlen))} = {value}")

        print()

    def read_yaml_data(self, config):
        r"""Reader/handler of data structure

        Parameters
        ----------
        config: dict
            Data dictionary
        """
        print(boldtext("Reading data...\n"))

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
                container['priors']['fraction'] = {'value': val, 'type': ptype}
            container['lambda'] = config.get("lambda", None)
            container['zerror'] = config.get("zerror", None)
            for p in config.get("parameters", {}):
                val, ptype = self._identify_prior(config['parameters'][p], p)
                if ptype == 'fixed':
                    container['fiducials'][p] = val
                elif ptype in ['LL','coevolution','excursion_set']:
                    container['relations'][p] = ptype
                else:
                    container['priors'][p] = {'value': val, 'type': ptype}

        def get_redshift(correct, interloper):
            return (correct['lambda'] / interloper['lambda']
                    * (1.0 + correct['zeff']) - 1.0)

        expected_keys = {
            "data_model": "LE3",
            "mixing_matrix_kp_max_multiplier": 1.75,
            "input_dir": ""
        }
        maxlen = max(len(k) for k in expected_keys.keys())

        for key, default in expected_keys.items():
            value = config.get(key, default)
            setattr(self, key, value)
            print(f"{boldtext(key.ljust(maxlen))} = {value}")
        print ()

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
        self.stat, self.kmax, self.zeff = {}, {}, {}
        self.nbar, self.fiducial_cosmology_obs = {}, {}

        for oi in self.observables:
            self.fname_data[oi] = config['observables'][oi].get("fname_data")
            self.fname_cov[oi] = config['observables'][oi].get("fname_cov")
            self.fname_mixing_matrix[oi] = config['observables'][oi].get(
                "fname_mixing_matrix", None)
            self.stat[oi] = config['observables'][oi].get("stat")
            self.kmax[oi] = config['observables'][oi].get("kmax")
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
            self.n_obs = {s: 0 for s in self.species}
            self.obs_id = {s: {} for s in self.species}
            for s in self.species:
                for oi in self.observables:
                    if s in self.composition[oi]:
                        self.obs_id[s][oi] = np.copy(self.n_obs[s])
                        self.n_obs[s] += 1
        else:
            self.composition = {oi: None for oi in self.observables}
            self.n_obs = len(self.observables)
            self.obs_id = {}
            for i,oi in enumerate(self.observables):
                self.obs_id[oi] = i
        print (boldtext("n_obs ="), self.n_obs)
        print (boldtext("obs_id ="), self.obs_id)
        print ()

        expected_keys = ["fname_data", "fname_cov", "fname_mixing_matrix",
                         "stat", "kmax", "zeff", "nbar", "fiducial_cosmology_obs"]
        expected_keys += ["composition" if self.has_composition else "sample"]
        maxlen = max(len(k) for k in expected_keys)

        for oi in self.observables:
            print("▮"*30 + boldtext(" Observable ") + f"{boldtext(oi)} " + "▮"*30 + "\n")

            for key in expected_keys:
                value = getattr(self, key)[oi]

                if isinstance(value, dict) and not is_flat_dict(value):
                    print(f"{boldtext(key.ljust(maxlen))} =")
                    print(pretty_nested(value, indent=4, level=1))
                else:
                    print(f"{boldtext(key.ljust(maxlen))} = {value}")

            print()

    def _create_params_setup(self):
        print(boldtext("Creating parameter structure...\n"))
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
        # AP: if a parameter is in self.fiducial_values it cannot be also in
        # self.priors
        for p in self.fiducial_values:
            if p in self.cosmo_params:
                self.fixed_cosmo_params[p] = self.fiducial_values[p]
        self.n_cosmo_params = len(self.sampled_cosmo_params)

        print (boldtext("↳ Sampled cosmo params ="), self.sampled_cosmo_params)
        print (boldtext("↳ Fixed cosmo params ="), self.fixed_cosmo_params)

        if 'wa' in self.sampled_cosmo_params or (
                'wa' in self.fixed_cosmo_params and
                self.fixed_cosmo_params['wa'] != 0.0):
            self.de_model = 'w0wa'
        elif 'w0' in self.sampled_cosmo_params or (
                'w0' in self.fixed_cosmo_params and
                self.fixed_cosmo_params['w0'] != -1.0):
            self.de_model = 'w0'
        else:
            self.de_model = 'lambda'
        print (boldtext("↳ Dark energy model ="), self.de_model)

        nonu = False
        if not ('Mnu' in self.sampled_cosmo_params or (
                'Mnu' in self.fixed_cosmo_params and
                self.fixed_cosmo_params['Mnu'] != 0.0)):
            nonu = True
        if nonu:
            self.model += '_nonu'
            self.fixed_cosmo_params.pop('Mnu')

        print (boldtext("↳ RSD model ="), self.model)

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
                                    'prior': priors_os[p]['value'],
                                    'type': priors_os[p]['type']
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

        expected_keys = ["sampled_nuisance_params", "fixed_nuisance_params",
                         "relation_nuisance_params", "AM_priors"]
        maxlen = max(len(k) for k in expected_keys)

        for key in expected_keys:
            value = getattr(self, key)

            if isinstance(value, dict) and not is_flat_dict(value):
                print("↳ " + f"{boldtext(key.ljust(maxlen))} =")
                print("↳ " + pretty_nested(value, indent=4, level=1))
            else:
                print("↳ " + f"{boldtext(key.ljust(maxlen))} = {value}")

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
                        np.amax(self.kmax[oi])*self.mixing_matrix_kp_max_multiplier
                    )
                    bins_mixing_matrix = [mm_k, mm_kp]
                    W_mixing_matrix = mm_comet
                else:
                    bins_mixing_matrix = None
                    W_mixing_matrix = None
                self.emu.define_data_set(
                    obs_id=oi, stat=self.stat[oi], zeff=self.zeff[oi],
                    bins=k_eff, signal=data_comet, cov=cov_comet,
                    bins_mixing_matrix=[mm_k,mm_kp],
                    W_mixing_matrix=mm_comet,
                    fiducial_cosmology=self.fiducial_cosmology_obs[oi],
                    composition=self.composition[oi], nbar=self.nbar[oi])
            self.emu.data[oi].set_kmax(self.kmax[oi])

    def _g2bG2_relation(self, relation, b1):
        if relation == 'LL' or relation == 'coevolution':
            return - 2.0 / 7.0 * (b1 - 1.0)
        elif relation == 'excursion_set':
            return 0.524 - 0.547 * b1 + 0.046 * b1**2

    def _g21bGam3_relation(self, relation, b1, g2bG2):
        if relation == 'coevolution':
            if self.bias_basis == 'EggScoSmi':
                return 2.0 / 21.0 * (b1 - 1.0) + 6.0 / 7.0 * g2bG2
            elif self.bias_basis == 'AssBauGre':
                return -1.0 / 6.0 * (b1 - 1.0) - 5.0 / 2.0 * g2bG2

    def _assign_params(self, get_value):

        if self.has_composition:
            params = {s: {} for s in self.species}

            for p in self.sampled_cosmo_params:
                val = get_value(p)
                for s in self.species:
                    params[s][p] = np.repeat(val, self.n_obs[s])

            for p in self.fixed_cosmo_params:
                for s in self.species:
                    params[s][p] = np.repeat(
                        self.fixed_cosmo_params[p], self.n_obs[s])

            for pos in self.sampled_nuisance_params:
                p, oi, s = pos.split('.')
                noi = self.obs_id[s][oi]
                if p not in params[s]:
                    params[s][p] = np.zeros(self.n_obs[s])
                params[s][p][noi] = get_value(pos)

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
                if p in ['g2', 'bG2']:
                    params[s][p][noi] = self._g2bG2_relation(
                        rel, params[s]['b1'][noi])
                elif p in ['g21', 'bGam3']:
                    if self.bias_basis == 'EggScoSmi':
                        params[s][p][noi] = self._g21bGam3_relation(
                            rel, params[s]['b1'][noi], params[s]['g2'][noi])
                    elif self.bias_basis == 'AssBauGre':
                        params[s][p][noi] = self._g21bGam3_relation(
                            rel, params[s]['b1'][noi], params[s]['bG2'][noi])

            for s in self.species:
                params[s]['z'] = np.array([
                    self.composition[oi][s]['zeff']
                    for oi in self.observables if s in self.composition[oi]])

        else:
            params = {}

            for p in self.sampled_cosmo_params:
                params[p] = np.repeat(get_value(p), self.n_obs)

            for p in self.fixed_cosmo_params:
                params[p] = np.repeat(self.fixed_cosmo_params[p], self.n_obs)

            for po in self.sampled_nuisance_params:
                p, oi = po.split('.')
                noi = self.obs_id[oi]
                if p not in params:
                    params[p] = np.zeros(self.n_obs)
                params[p][noi] = get_value(po)

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

    def _assign_params_multinest(self, cube):
        n = 0

        def get_value(key):
            nonlocal n
            val = cube[n]
            n += 1
            return val

        return self._assign_params(get_value)

    def _assign_params_nautilus(self, params_dict):

        def get_value(key):
            return params_dict[key]

        return self._assign_params(get_value)


    def _generate_prior_loglike(self):
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

            def loglike(cube):
                params = self._assign_params_multinest(cube)
                chi2 = self.emu.chi2(self.observables, params, self.kmax,
                                     self.de_model, AM_priors=self.AM_priors)
                return -0.5 * chi2

        elif self.sampler == "nautilus":

            import nautilus
            prior = nautilus.Prior()
            for p in self.sampled_cosmo_params:
                if self.sampled_cosmo_params[p]['type'] == 'flat':
                    p_min, p_max = self.sampled_cosmo_params[p]['prior']
                    prior.add_parameter(p, dist=(p_min, p_max))
                else:
                    mean, std = self.sampled_cosmo_params[p]['prior']
                    prior.add_parameter(p, dist=norm(loc=mean, scale=std))
            for p in self.sampled_nuisance_params:
                if self.sampled_nuisance_params[p]['type'] == 'flat':
                    p_min, p_max = self.sampled_nuisance_params[p]['prior']
                    prior.add_parameter(p, dist=(p_min, p_max))
                else:
                    mean, std = self.sampled_nuisance_params[p]['prior']
                    prior.add_parameter(p, dist=norm(loc=mean, scale=std))

            def loglike(params_dict):
                params = self._assign_params_nautilus(params_dict)
                chi2 = self.emu.chi2(self.observables, params, self.kmax,
                                     self.de_model, AM_priors=self.AM_priors)
                return -0.5 * chi2.squeeze()

        return prior, loglike

    def run_chain(self, resume=False, verbose=True):
        if self.sampler == 'multinest':
            import pymultinest
            prior, loglike = self._generate_prior_loglike()
            pymultinest.solve(
                loglike, prior, self.n_params_total,
                outputfiles_basename=f'{self.output_dir}/{self.output_filename}',
                resume=resume, verbose=verbose, n_live_points=self.n_live,
                sampling_efficiency=self.sampling_efficiency,
                evidence_tolerance=self.evidence_tolerance
            )
        elif self.sampler == 'nautilus':
            import nautilus
            prior, loglike = self._generate_prior_loglike()
            base, _ = os.path.splitext(f'{self.output_dir}/{self.output_filename}')
            #checkpoint = base + ".hdf5"
            sampler = nautilus.Sampler(prior, loglike, n_live=self.n_live,
                                       pool=self.pool)#, filepath=checkpoint,
                                       #resume=resume)
            sampler.run(f_live=self.f_live, n_eff=self.n_eff,
                        verbose=verbose, discard_exploration=True)
            log_z = sampler.evidence()
            points, log_w, log_l = sampler.posterior()
            np.save(f'{self.output_dir}/{self.output_filename}',
                    np.c_[log_w, log_l, points])
