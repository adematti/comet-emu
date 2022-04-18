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
params['alpha_lo'] = 1.1
params['alpha_tr'] = 0.9
params['f']        = 0.7

# Finally, the bias parameters: any parameters from {b1, b2, g2, g21, c0, c2, c4, cnlo, N0, N20, N22} can be specified.
# Parameters, which are not explicitly specified are automatically set to zero. As an example, let's just set b1 and b2:
params['b1'] = 2.
params['b2'] = -0.5

print(EFT.Pell(0.1, params, ell=2))
print(EFT.Pell(np.array([0.1,0.2,0.3]), params, ell=[0,2,4]))

print(EFT.Pell([np.array([0.1,0.2]),0.3], params, ell=[0,4]))

# Define range of scales (remember: in 1/Mpc)
k_Mpc = np.logspace(-3,np.log10(0.3),100)

# get multipoles (in Mpc^3)
Pell_Mpc_1 = EFT.Pell(k_Mpc, params, ell=[0,2,4])

# Now, let's add/change some parameter values and obtain a second set of predictions
params['alpha_tr'] = 1.2
params['g2']       = -0.3
params['c0']       = -4.
params['cnlo']     = 6.
params['N0']       = 0.6
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
plt.savefig("docs/imgs/EFT_Multipoles.png")
