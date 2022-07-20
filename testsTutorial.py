from comet import comet
import numpy as np
import matplotlib.pyplot as plt

EFT=comet(model="EFT", use_Mpc=True)
EFT.define_nbar(nbar=1.32904e-4)

 # Let's create a parameter dictionary
params = {}

# We always need to specify the shape parameter values, e.g.
params['wc'] = 0.11544
params['wb'] = 0.0222191
params['ns'] = 0.9632

# For predictions using the RSD parameter space we also need to specify values for the following four parameters, e.g.
params['s12']      = 0.6
params['q_lo'] = 1.1
params['q_tr'] = 0.9
params['f']        = 0.7

# Finally, the bias parameters: any parameters from {b1, b2, g2, g21, c0, c2, c4, cnlo, N0, N20, N22} can be specified.
# Parameters, which are not explicitly specified are automatically set to zero. As an example, let's just set b1 and b2:
params['b1'] = 2.
params['b2'] = -0.5

EFT.Pell(0.1, params, ell=2)
EFT.Pell(np.array([0.1,0.2,0.3]), params, ell=[0,2,4])

EFT.Pell([np.array([0.1,0.2]),0.3], params, ell=[0,4])

# Define range of scales (remember: in 1/Mpc)
k_Mpc = np.logspace(-3,np.log10(0.3),100)

# get multipoles (in Mpc^3)
Pell_Mpc_1 = EFT.Pell(k_Mpc, params, ell=[0,2,4])

# Now, let's add/change some parameter values and obtain a second set of predictions
params['q_tr'] = 1.2
params['g2']       = -0.3
params['c0']       = -4.
params['cnlo']     = 6.
params['NP0']       = 0.6
Pell_Mpc_2 = EFT.Pell(k_Mpc, params, ell=[0,2,4])

# Plot the results!
f = plt.figure()
ax = f.add_subplot(111)

ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_1['ell0'],c='C0',ls='-',label='P0')
ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_2['ell0'],c='C0',ls='--')

ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_1['ell2'],c='C1',ls='-',label='P2')
ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_2['ell2'],c='C1',ls='--')

ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_1['ell4'],c='C2',ls='-',label='P4')
ax.semilogx(k_Mpc, k_Mpc**0.5*Pell_Mpc_2['ell4'],c='C2',ls='--')

ax.set_xlabel('$k$ [1/Mpc]',fontsize=12)
ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc})^{5/2}$]',fontsize=12)
ax.legend(fontsize=12)
#plt.savefig("docs/source/imgs/EFT_Multipoles.png")

EFT.define_units(use_Mpc=False)
EFT.define_nbar(nbar=3.95898e-4)
params['h'] = 0.695

k_hMpc = np.logspace(-3,np.log10(0.3),100)
Pell_hMpc_2 = EFT.Pell(k_hMpc,params,ell=[0,2,4])

f = plt.figure()
ax = f.add_subplot(111)

ax.semilogx(k_Mpc/params['h'], (k_Mpc/params['h'])**0.5*Pell_Mpc_2['ell0']*params['h']**3,c='C0',ls='-',label='P0')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_hMpc_2['ell0'],c='C0',ls='--')

ax.semilogx(k_Mpc/params['h'], (k_Mpc/params['h'])**0.5*Pell_Mpc_2['ell2']*params['h']**3,c='C1',ls='-',label='P2')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_hMpc_2['ell2'],c='C1',ls='--')

ax.semilogx(k_Mpc/params['h'], (k_Mpc/params['h'])**0.5*Pell_Mpc_2['ell4']*params['h']**3,c='C2',ls='-',label='P4')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_hMpc_2['ell4'],c='C2',ls='--')

ax.set_xlabel('$k$ [h/Mpc]',fontsize=12)
ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{5/2}$]',fontsize=12)
ax.legend(fontsize=12)

#plt.savefig("docs/source/imgs/EFT_Multipoles_hMpc.png")

params_fid_Minerva = {'h':0.695, 'wc':0.11544, 'wb':0.0222191, 'z':0.57}

EFT.define_fiducial_cosmology(params_fid=params_fid_Minerva, de_model='lambda')

params['h']  = 0.8
params['As'] = 2.3
params['z']  = 0.6

Pell_LCDM_hMpc_1 = EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model='lambda') # E.g., this is for a flat LCDM cosmology
# s12, q_tr, q_lo and f are different now!
print(EFT.params)

Pell_LCDM_hMpc_2 = EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model='lambda', q_tr_lo=[1,1])

 # The results differ accordingly!
f = plt.figure()
ax = f.add_subplot(111)

ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_1["ell0"],c='C0',ls='-',label='P0')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_2["ell0"],c='C0',ls='--')

ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_1["ell2"],c='C1',ls='-',label='P2')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_2["ell2"],c='C1',ls='--')

ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_1["ell4"],c='C2',ls='-',label='P4')
ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM_hMpc_2["ell4"],c='C2',ls='--')

ax.set_xlabel('$k$ [h/Mpc]',fontsize=12)
ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{5/2}$]',fontsize=12)
ax.legend(fontsize=12)

#plt.savefig("docs/source/imgs/EFT_Multipoles_LCDM.png")

print(EFT.params_ranges)

print(min(EFT.k_table), max(EFT.k_table))
