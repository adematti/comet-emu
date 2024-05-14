# Loading modules
import numpy as np
import glob, os, sys, time
from scipy.interpolate import interp1d
from scipy.special import ndtri
from scipy.stats import norm
from comet import comet
import pymultinest 
import configparser
import json
import cProfile
import re





fiducial_values = {'z':1.0,'h':0.67, 'wc':0.121203, 'wb':0.0219961,'w0':-1.0, 'wa':0.0, 'ns':0.96, 'As':2.1, 'b1':2.02, 'b2':0., 'g2':0., 'g21':0.,\
                  'bG2':0,'bGam3':0,'c0':0., 'c2':0., 'c4':0., 'cnlo':0., 'avir':0., 'NP0':0., 'NP20':0., 'NP22':0.}


#Define the numvber of redshift bin indicating by Zeff


zeff=4


cosmo_fid = ['z','h','wc','wb','ns','As']
# Get settings
config = configparser.ConfigParser()
config.read('THEORYDV.ini')

## setttings ###################################################################

#model = 'EFT'
model = config.get("Emulator", "RSD")
de_model = config.get("Emulator", "de_model", fallback="lambda")
discretized  = config.get("Emulator", "discretized")
CosmoChain = config.get("Emulator", "CosmoChain")
use_Mpc    = config.get("Emulator", "use_Mpc") == 'T'
chi2_decomposition = True if CosmoChain == 'F' else False

AMM=config.get("Analytical_Marginalisation", "marginalisation")
ANM= True if AMM == 'T' else False
#Data
l_box = float(config.get("Data", "lbox"))
n_fft= float(config.get("Data", "n_fft"))
fname_data = config.get("Data","fname_data")
fname_cov  = config.get("Data","fname_cov")

try:
    window = config.get("Data","window")
    convolve_window = True if window == 'T' else False
    fname_window = config.get("Data","fname_window")
    fname_kprime  = config.get("Data","fname_kprime")

except:
    print("No window applied c:")
    convolve_window = False


#nbar = float(config.get("Data","nbar"))
nbar = json.loads(config.get('Data', 'nbar'))
print ('nbar [(h/Mpc)**3] =', nbar)
kmax= json.loads(config.get("Data","kmax"))
kmax_chain = kmax

folder_chain = config.get("Output", "folder_chain")


#Bias relations
discretized= True if discretized == 'T' else False
CosmoChain= True if CosmoChain == 'T' else False

g2Fix  = config.get("Coevolution", "g2Fix")
g2ExSet  = config.get("Coevolution", "g2ExSet")
g21CoEvol = config.get("Coevolution", "g21CoEvol")

g2Fix  = True if g2Fix == 'T' else False
g2ExSet  = True if g2ExSet == 'T' else False
g21CoEvol = True if g21CoEvol == 'T' else False

#Bias Labels to be fixed
if g21CoEvol:
    bias_labes = "_g21CoEvol"
    
elif g2ExSet:
    bias_labes = "_g2ExSet"

elif g2Fix:
    bias_labes = "_g2LL"

else:
    bias_labes = ""
    
if g21CoEvol and g2Fix:
    bias_labes = "_g21CoEvolg2LL"

if g21CoEvol and g2ExSet:
    bias_labes = "_g21CoEvolg2ExSet"

def b2f(b1, g2):
    return 0.412 - 2.143*b1 + 0.929*b1**2 + 0.008*b1**3 + 4./3*g2
def g2LL(b1):
    return -2./7*(b1-1)
def g2f(b1):
    return 0.524 - 0.547*b1 + 0.046*b1**2
def g21f(b1, g2):
    return 2./21*(b1-1)+ 6./7*g2

#MultiNest config
n_live = int(config.get("MultiNest", "n_live", fallback=600))
sampling_efficiency = float(config.get("MultiNest", "sampling_efficiency",
                                       fallback=0.8))
evidence_tolerance  = float(config.get("MultiNest", "evidence_tolerance",
                                       fallback=0.5))

resume=config.get("MultiNest", "resume", fallback="F")
resume= True if discretized == 'T' else False

if discretized:
    binning = {'kfun':2*np.pi/l_box, 'dk':2*np.pi/l_box, 'effective':True}
else: binning = None

## defintions ###################################################################
if model=="EFT":
    parameters = ['z','h','As','wc','wb', 'w0', 'wa','ns','b1','b2','bG2','bGam3','c0','c2','c4','cnlo','NP0','NP20','NP22']
    #parameters = ['z','h','wc','wb','ns','As','b1','b2','g2','g21','c0','c2','c4','cnlo','NP0','NP20','NP22'] 

else: parameters = ['z','h','wc','wb','ns','As','b1','b2','g2','g21','c0','c2','c4','avir','NP0','NP20','NP22']

params_sampling={}
if ANM==True :
   params_sampling=config.get("Parameters","params_sampling_AM").replace(',','').split()
else:
   params_sampling= config.get("Parameters","params_sampling").replace(',','').split()

#Concatenate the sampling
listcosmo=['wc','wb','ns','h','As']
cosmoparam=[]#'h','As','ns','wc','wb']
for p in params_sampling:
    if p in listcosmo:
        cosmoparam=np.append(cosmoparam,p)

params_sampling_new=params_sampling
if zeff != 1:
    #params_sampling_new=np.hstack((zeff*params_sampling))
    params_sampling=params_sampling[len(cosmoparam):]
    params22=np.hstack(np.column_stack((params_sampling,params_sampling,params_sampling,params_sampling)))
    params22=np.hstack((cosmoparam,params22))#To do for generic redshift bin now working only for 2
else:
    params22=params_sampling


print(params22)

fiducial_values_update = {}
for p in cosmo_fid:
    fiducial_values_update[p] = float(config.get("FiducialValues",p,fallback=fiducial_values[p]))
cosmo_fid = fiducial_values_update

fiducial_values_update = {}
for p in parameters:
    fiducial_values_update[p] = float(config.get("FiducialValues",p,fallback=fiducial_values[p]))
fiducial_values = fiducial_values_update

priors = {}
for p in params_sampling:
    #print(config.get("Priors",p))
    priors[p] = json.loads(config.get("Priors",p))
    #print(priors[p])
for p in cosmoparam:
    priors[p]=json.loads(config.get("Priors",p))
print(cosmoparam)
'''
priors['h']=json.loads(config.get("Priors",'h'))
priors['As']=json.loads(config.get("Priors",'As'))
priors['wc']=json.loads(config.get("Priors",'wc'))
priors['wb']=json.loads(config.get("Priors",'wb'))
priors['ns']=json.loads(config.get("Priors",'ns'))
'''
#print(priors)
#fname_chain  = config.get("Output","fname_chain")
#fname_base = fname_data.split("/")[-1].split(".")[0] + '_model{}_zs_{}_{}_{}_{}{}_nlive{}-'.format(
fname_base = fname_data.split("/")[-1].split(".")[0] + '_'.format(
    model,
    str(kmax_chain[0]).replace('.','p'),
    str(kmax_chain[1]).replace('.','p'),
    #str(kmax_chain[2]).replace('.','p'),
    "".join(params_sampling),
    bias_labes,
    n_live
)
fname_chain=folder_chain+fname_base
print("We are going to run the case = ", fname_base)

# Save parameter ranges to file
'''
with open('{}.ranges'.format(fname_chain), "w") as franges:
    for p in params_sampling:
        franges.write('{}\t{}\t{}\n'.format(p, str(priors[p][0]), str(priors[p][1])))
'''
params_fid_comet = {par: np.repeat(cosmo_fid[par], 4) for par in ['h', 'wc', 'wb', 'ns', 'As']}
params_fid_comet['z']=[1.0, 1.2, 1.4, 1.65]

print('oleeee',params_fid_comet)
def init_emu():

    emu = comet(model=model, use_Mpc=False)

    emu.define_fiducial_cosmology(params_fid=params_fid_comet)
    emu.define_nbar(nbar=nbar)

    data = np.loadtxt(fname_data)

    if fname_data==fname_cov:
        cov = np.diag(np.concatenate((data[:,1][data[:,0]<0.5]**2, data[:,2][data[:,0]<0.5]**2, data[:,3][data[:,0]<0.5]**2)))
    else:
        cov = np.loadtxt(fname_cov)

    if discretized:
        kvec = np.arange(2*np.pi/l_box, n_fft*2*np.pi/l_box+2*np.pi/l_box, 2*np.pi/l_box)

    else:
        kvec = data[:,0]
    
    monovec=data[:,1]
    quadvec=data[:,3]
    hexavec=data[:,5]

    cov2=np.loadtxt('data/comet_datavec/comet_datavec/cov_comet_EFT_z1.2_nosyst_DR3.txt')
    data2=np.loadtxt('data/comet_datavec/comet_datavec/power_comet_EFT_z1.2_nosyst_DR3.txt')
    monovec2=data2[:,1]
    quadvec2=data2[:,3]
    hexavec2=data2[:,5]
    signal2 = np.asarray([monovec2, quadvec2, hexavec2]).T
  
    
    
    signal = np.asarray([monovec, quadvec, hexavec]).T
    
    signal2 = np.asarray([monovec2, quadvec2, hexavec2]).T
    data3=np.loadtxt('data/comet_datavec/comet_datavec/power_comet_EFT_z1.4_nosyst_DR3.txt')
    cov3=np.loadtxt('data/comet_datavec/comet_datavec/cov_comet_EFT_z1.4_nosyst_DR3.txt')
    monovec3=data3[:,1]
    quadvec3=data3[:,3]
    hexavec3=data3[:,5]
    
    signal3 = np.asarray([monovec3, quadvec3, hexavec3]).T
    cov4=np.loadtxt('data/comet_datavec/comet_datavec/cov_comet_EFT_z1.65_nosyst_DR3.txt')
    data4=np.loadtxt('data/comet_datavec/comet_datavec/power_comet_EFT_z1.65_nosyst_DR3.txt')
    monovec4=data4[:,1]
    quadvec4=data4[:,3]
    hexavec4=data4[:,5]
    
    signal4 = np.asarray([monovec4, quadvec4, hexavec4]).T
    emu.change_bias_basis('AssBauGre')
    emu.define_data_set('Pk',zeff=1.0, bins=kvec, signal=signal,cov=cov)

    emu.define_data_set('Pk2',zeff=1.2, bins=kvec, signal=signal2,cov=cov2)
    emu.define_data_set('Pk3',zeff=1.4, bins=kvec, signal=signal3,cov=cov3)
    emu.define_data_set('Pk4',zeff=1.65, bins=kvec, signal=signal4,cov=cov4)#/scalecov[z])

    if convolve_window:
        W = np.loadtxt(fname_window)
        kprime = np.loadtxt(fname_kprime)
        emu.define_data_set(obs_id='Pk', zeff=0.57,bins_mixing_matrix=[kvec, kprime], W_mixing_matrix=W_mixing_matrix)

    return emu

def run_chain(emu, fname_base, n_live=400, sampling_efficiency=0.8,
              evidence_tolerance=0.5):
    zbin=zeff
    def assign_params(cube):
        params={}
        n = 0
        if zbin !=1 :
            for p in parameters:
                #print(p,n)
                params[p]=np.zeros(zbin)
                if p in cosmoparam:
                    #params[p][:]=cube[n]
                    for j in range(0,zbin):
                        params[p][j]=cube[n]
                    n+=1
                else:
                    for j in range(0,zbin):
                        if p in params_sampling: 
                            params[p][j]=cube[n]
                            n += 1
                        else:
                            params[p][j] = fiducial_values[p]
            if g2Fix:
                params['g2'] = g2LL(params['b1'])
            if g2ExSet:
                params['g2'] = g2f(params['b1'])
            if g21CoEvol:
                params['g21'] = g21f(params['b1'], params['g2'])
                
        else:
            for p in parameters:
                if p in params_sampling:
                    params[p] = cube[n]
                    n += 1
                else:
                    params[p] = fiducial_values[p]

        params['z']=[1.0,1.2,1.4,1.65]

        return params
    

    n_params = len(params22)


    def prior(cube):
        for n in range(n_params):
            p = params22[n]#params_sampling[n]
            #print(p)
            try:
                priorType=priors[p][2]
            except:
                #print('As no prior type is defined, we are assuming Uniform prior on' + p)
                priorType='uniform'

            if priorType=='uniform':
                #print('Uniform prior on' + p)
                cube[n] = priors[p][0] + (priors[p][1] - priors[p][0])*cube[n]

            elif priorType=='gaussian':
                #print('Gaussian prior on' + p)
                cube[n] = priors[p][0] + priors[p][1]*ndtri(cube[n]) 

        return cube

        
    
    def loglike(cube):

        params=assign_params(cube)

        params_AM=['bGam3','cnlo','c0','c2','c4','NP0','NP20']
        Gpriors1={}
        Gpriors1['Pk']={'mu':[0,0,0,0,0,1,0],'sigma':[5,500,500,500,500,1,10]}
        Gpriors1['Pk2']={'mu':[0,0,0,0,0,1,0],'sigma':[5,500,500,500,500,1,10]}
        Gpriors1['Pk3']={'mu':[0,0,0,0,0,1,0],'sigma':[5,500,500,500,500,1,10]}
        Gpriors1['Pk4']={'mu':[0,0,0,0,0,1,0],'sigma':[5,500,500,500,500,1,10]}
        
        chi2=emu.chi2(['Pk','Pk2','Pk3','Pk4'], params, {'Pk':0.3, 'Pk2':0.3, 'Pk3':0.3,'Pk4':0.3}, de_model='lambda',chi2_decomposition=False,binning = binning,convolve_window=convolve_window,Analytical_Marg=True,params_tomarg=params_AM,Gpriors=Gpriors1)


        return chi2
    


    pymultinest.solve(
        loglike,prior,n_params,
        outputfiles_basename=fname_base,
        resume=False,verbose=True, n_live_points=n_live,
        sampling_efficiency=sampling_efficiency,
        evidence_tolerance=evidence_tolerance,const_efficiency_mode=False) #
    


    
emu = init_emu()



amsi='/testAMneww6'

run_chain(emu, '/home/hidra4/gambardella/Power_Comet/Test'+amsi,n_live=n_live, sampling_efficiency=sampling_efficiency,
                  evidence_tolerance=evidence_tolerance) 

