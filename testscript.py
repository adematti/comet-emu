from comet import comet
import numpy as np
import matplotlib.pyplot as plt


zsnap=0.9
zsim = {0.90:0.904588119125, 1.2:1.188754948786,
        1.5:1.530882839039, 1.8:1.7899589}
nbarsim = {0.90:2.042611E-03, 1.2:1.02876011E-3,
        1.5:0.58531983E-03, 1.8:0.313402E-03}

nbar = nbarsim[zsnap]

#we create an instantiation of the emulator with the name of the
#model we want to use

Pell=comet(model="EFT", use_Mpc=False)
#Pell.define_units(use_Mpc=False)
Pell.define_nbar(nbar=nbar)

params_fid_Euclid = {'h':0.67, 'wc':0.1212029, 'wb':0.0219961, 'ns':0.96,
              'z':zsim[zsnap], 'As':2.1108474277}

Pell.define_fiducial_cosmology(params_fid=params_fid_Euclid)

g2 = lambda b1: -2/7.*(b1-1)
fidutial_params = {}
fidutial_params["h"] = 0.67
fidutial_params["wc"] = 0.121203
fidutial_params["As"] = 2.110651747790177
fidutial_params["wb"] = 0.0219961
fidutial_params["ns"] = 0.96
fidutial_params["z"] = zsim[zsnap]

fidutial_params["b1"] = 1.3660692977077329#1.3679
fidutial_params["b2"] = -0.7341744950598681#-0.6209
fidutial_params["g2"] = g2(fidutial_params["b1"])#-0.2283
fidutial_params["g21"] = -0.010824734433118266#0.4114#-4/7.*(0.4114 + fidutial_params["b1"]) #0.4114
fidutial_params["c0"] = 0.2104184561082434#2.8042*0.4**2
fidutial_params["c2"] = 0*1.2104184561082434#2.8042*0.4**2
fidutial_params["c4"] = 0*1.2104184561082434#2.8042*0.4**2
fidutial_params["NP0"] = 0*0.7634175841445507#0.72792*0.4**3

k_table = np.logspace(-3,np.log10(0.3),100)
Pell_LCDM = Pell.Pell(k=k_table, params=fidutial_params, ell=[0,2,4], de_model="lambda")


# Real space plot
f = plt.figure(figsize=(10, 6))
ax = f.add_subplot(111)


ax.semilogx(k_table, k_table**0.5*Pell_LCDM["ell0"], c='C0',ls='-',label=r'$P_0$')
ax.semilogx(k_table, k_table**0.5*Pell_LCDM["ell2"], c='C1',ls='-',label=r'$P_2$')
ax.semilogx(k_table, k_table**0.5*Pell_LCDM["ell4"], c='C2',ls='-',label=r'$P_4$')

ax.set_xlabel('$k$ [h/Mpc]',fontsize=12)
ax.set_ylabel(r'$P_{\ell}(k)$ [$(\mathrm{Mpc/h})^{5/2}$]',fontsize=12)
ax.legend(fontsize=12)
plt.show()
