import numpy as np
import configparser
import json
import pymultinest
import sys
from PTEmu import PTEmu

parameters = ['z', 'h', 'wc', 'wb', 'ns', 'As',
              'b1', 'b2', 'g2', 'g21',
              'c0', 'c2', 'c4', 'cnlo',
              'N0', 'N20', 'N22']
fiducial_values = {'z': 0.57, 'h': 0.695, 'wc': 0.11544, 'wb': 0.0222191,
                   'ns': 0.9632, 'As': 2.2078559,
                   'b1': 2.02, 'b2': 0.0, 'g2': 0.0, 'g21': 0.0,
                   'c0': 0.0, 'c2': 0.0, 'c4': 0.0, 'cnlo': 0.0,
                   'N0': 0.0, 'N20': 0.0, 'N22': 0.0}

# Get settings
config = configparser.ConfigParser()
config.read(sys.argv[1])

emu_params_shape = config.get("Emulator",
                              "emu_params_shape").replace(',','').split()
emu_params_RSD   = config.get("Emulator",
                              "emu_params_RSD").replace(',','').split()
fname_tables     = config.get("Emulator", "fname_tables")
fname_emulator   = config.get("Emulator", "fname_emulator")

VIR = True if 'avir' in emu_params_RSD else False
if VIR:
    parameters.append('avir')
    fiducial_values['avir'] = 0.0
RSD = config.get("Emulator", "RSD")
RSD = True if RSD == 'T' else False

nbar       = float(config.get("Data", "nbar"))
volfac     = float(config.get("Data", "volfac"))
fname_data = config.get("Data", "fname_data")
fname_cov  = config.get("Data", "fname_cov")
kmax       = json.loads(config.get("Data", "kmax"))
use_Mpc    = config.get("Data", "use_Mpc")
use_Mpc    = True if use_Mpc == 'T' else False

params_sampling = config.get("Parameters",
                             "params_sampling").replace(',','').split()
n_params = len(params_sampling)

priors = {}
for p in params_sampling:
    priors[p] = json.loads(config.get("Priors", p))

g2coevo  = config.get("Coevolution", "g2")
g21coevo = config.get("Coevolution", "g21")
g2coevo  = True if g2coevo == 'T' else False
g21coevo = True if g21coevo == 'T' else False

fiducial_values_update = {}
for p in parameters:
    fiducial_values_update[p] = float(config.get("FiducialValues", p,
                                                 fallback=fiducial_values[p]))
fiducial_values = fiducial_values_update

fname_chain  = config.get("Output","fname_chain")

n_live = int(config.get("MultiNest", "n_live", fallback=400))
sampling_efficiency = float(config.get("MultiNest", "sampling_efficiency",
                                       fallback=0.8))
evidence_tolerance  = float(config.get("MultiNest", "evidence_tolerance",
                                       fallback=0.5))

# Save parameter ranges to file
with open('{}.ranges'.format(fname_chain), "w") as franges:
    for p in params_sampling:
        franges.write('{}\t{}\t{}\n'.format(p,
                                            str(priors[p][0]),
                                            str(priors[p][1])))

# Load emulator
params_definition = {}
for p in emu_params_shape:
    params_definition[p] = 'SHAPE'
for p in emu_params_RSD:
    params_definition[p] = 'RSD'
print (params_definition)
emu = PTEmu(params=params_definition)
emu.load_emulator_data(fname=fname_tables)

# Define data set
data = np.loadtxt(fname_data)
cov = np.loadtxt(fname_cov)

if RSD:
    emu.load_emulator(fname_base=fname_emulator, data_type=['PL', 's12'])
    emu.load_emulator(fname_base=fname_emulator, data_type=[0, 2, 4])
    emu.define_units(use_Mpc=use_Mpc)
    emu.define_nbar(nbar=nbar)
    emu.define_data_set(
        obs_id='Pk', bins=data[:, 0], signal=data[:, [1,3,5]],
        cov=cov, theory_cov=True)
else:
    emu.load_emulator(fname_base=fname_emulator, data_type=['PL', 's12'])
    emu.load_emulator(fname_base=fname_emulator, data_type=0)
    emu.define_units(use_Mpc=use_Mpc)
    emu.define_nbar(nbar=nbar)
    emu.define_data_set(
        obs_id='Pk', bins=data[:, 0], signal=data[:, [1]],
        cov=cov, theory_cov=True)

emu.define_fiducial_cosmology(params_fid=fiducial_values)

# Co-evolution relations
def g2f(b1):
    return 0.524 - 0.547*b1 + 0.046*b1**2

def g21f(b1, g2):
    return 2./21*(b1-1)+6./7*g2


# Set up likelihood and prior functions for multinest
def assign_params(cube):
    params = {}
    n = 0
    for p in parameters:
        if p in params_sampling:
            params[p] = cube[n]
            n += 1
        else:
            params[p] = fiducial_values[p]
    return params

def prior(cube):
    for n in range(n_params):
        p = params_sampling[n]
        cube[n] = priors[p][0] + (priors[p][1] - priors[p][0])*cube[n]
    return cube

def loglike(cube):
    params = assign_params(cube)

    if g2coevo:
        params['g2'] = g2f(params['b1'])
    if g21coevo:
        params['g21'] = g21f(params['b1'], params['g2'])

    if RSD:
        chi2 = volfac*emu.chi2(obs_id='Pk', params=params, kmax=kmax,
                               de_model='lambda')
    else:
        chi2 = volfac*emu.chi2(obs_id='Pk', params=params, kmax=kmax,
                               de_model='lambda', alpha_tr_lo=[1., 1.])

    return -0.5*chi2


# Run multinest
pymultinest.solve(loglike, prior, n_params, outputfiles_basename=fname_chain,
                  resume=False, verbose=True, n_live_points=n_live,
                  sampling_efficiency=sampling_efficiency,
                  evidence_tolerance=evidence_tolerance)
