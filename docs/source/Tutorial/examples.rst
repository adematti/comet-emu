.. _examples:

Tutorials
---------

Quick-start
===========

.. container:: cell markdown

   .. rubric:: \
      :name: quick-start

   -  In this tutorial we will show

      -  How to initialise the emulator.
      -  How to obtain multipoles for the standard :math:`\Lambda CDM`
         cosmology.

.. container:: cell markdown

   Let’s first import ``comet`` as well as other requried libraries:

.. container:: cell code

   .. code:: python

      from comet import comet
      import numpy as np
      import matplotlib.pyplot as plt

.. container:: cell markdown

   At initialisation we only need to specify the perturbation theory
   model that we want to use (for an overview of the models implemented
   in COMET, see :ref:`here<models>`) and we can configure COMET either in :math:`Mpc`
   units (``use_Mpc = True``, which is the default option) or in
   :math:`h^{-1}Mpc` units (``use_Mpc = False``). All quantities that
   are not dimensionless are then returned or assumed to be given in the
   respective unit system. Let’s define an emulator object for the EFT
   model using the standard :math:`h^{-1}Mpc` units:

.. container:: cell code

   .. code:: python

      EFT=comet(model="EFT", use_Mpc=False)

.. container:: cell markdown

   In order to make predictions for a given cosmological model we first
   need to specify the fiducial background cosmology, from which the
   Alcock-Paczynski distortions will be computed. This is done by
   calling the function ``define_fiducial_cosmology`` with a dictionary
   specifying the cosmological parameters and the redshift:

.. container:: cell code

   .. code:: python

      params_fid = {'h':0.695, 'wc':0.11544, 'wb':0.0222191, 'z':0.57}

      # This assumes by default a "lambda" cosmology with w0 = -1, for other
      # options, see the in-depth examples below.
      EFT.define_fiducial_cosmology(params_fid=params_fid)

.. container:: cell markdown

   The function ``Pell``, which returns the power spectrum multipoles
   takes generally three parameters:

   #. The scales for which to compute the multipoles (in the
      corresponding units)
   #. A parameter dictionary, specifying cosmological, bias, and (if
      applicable) additional redshift-space distortions parameters
   #. The multipole number, i.e. ell = 0, 2, 4, or a list of multipole
      numbers

   The parameter dictionary must include all shape parameters: the
   physical cold dark matter and baryon densities (``wc`` and ``wb``)
   and the scalar spectral index (``ns``). In case of a flat
   :math:`\Lambda CDM` model we also need to specify values for
   :math:`h` (``h``), the amplitude of scalar fluctuations (``As``) and
   redshift (``z``). For other cosmologies, see :ref:`examples_in_depth`.

.. container:: cell code

   .. code:: python

      # Let's create a parameter dictionary
      params = {}

      # We always need to specify the shape parameter values, e.g.
      params['wc'] = 0.11544
      params['wb'] = 0.0222191
      params['ns'] = 0.9632

      # For a LCDM cosmology, we also need:
      params['h']  = 0.8
      params['As'] = 2.3
      params['z']  = 0.6

.. container:: cell markdown

   Finally, we define the values of the bias parameters. The complete
   list of parameters along with a brief explanation and their
   dioctionary keywords can be found here. In the following we only
   specify values for the linear and quadratic bias, all other
   parameters are automatically set to zero:

.. container:: cell code

   .. code:: python

      params['b1'] = 2.
      params['b2'] = -0.5

.. container:: cell markdown

   Now, let’s compute the monopole (``ell=0``), quadrupole (``ell=2``)
   and hexadecapole (``ell=4``) for a range of scales from
   :math:`0.001 hMpc^{−1}` to :math:`0.3hMpc^{−1}`:

.. container:: cell code

   .. code:: python

      k_hMpc = np.logspace(-3,np.log10(0.3),100)
      Pell_LCDM = EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model='lambda')

.. container:: cell markdown

   The output of ``Pell`` is given in a dictionary format:

.. container:: cell code

   .. code:: python

      print(Pell_LCDM.keys())

   .. container:: output stream stdout

      ::

         dict_keys(['ell0', 'ell2', 'ell4'])

.. container:: cell markdown

   So we can access our results and plot them as follow.

.. container:: cell code

   .. code:: python

      f = plt.figure(figsize=(10,5))
      ax = f.add_subplot(111)

      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell0"],c='C0',ls='-',label='P0')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell2"],c='C1',ls='-',label='P2')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell4"],c='C2',ls='-',label='P4')
      ax.set_xlabel('$k$ [h/Mpc]',fontsize=15)
      ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{5/2}$]',fontsize=15)
      ax.legend(fontsize=15)
      plt.show()

   .. container:: output display_data

      .. image:: vertopal_a899190433aa4f85bb1541798bd599a6/537407037d704ddedc6cd575f28290c731218f9f.png


.. _examples_in_depth:

In-depth options for obtaining multipoles
=========================================
.. container:: cell markdown

   .. rubric:: \
      :name: in-depth-options-for-obtaining-multipoles

.. container:: cell markdown

   -  Now, let's see some details:

      -  different cosmologies (:math:`\omega_0 + \omega_0\omega_a`)
      -  using the :math:`f-\sigma_{12}` parameter space
      -  the options for providing different :math:`k`-scales, float vs
         np.array vs list and the corresponding outputs
      -  describe the ``fixed_cosmo_boost`` function, i.e., speedup when
         just changing bias parameters


An alternative Dark energy model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: an-alternative-dark-energy-model

.. container:: cell markdown

   As mentioned before, we can make use of a different cosmological
   model. Let's try the ``w0wa`` model. We need to add such parameters
   to our params dictionary first.

.. container:: cell code

   .. code:: python

      # a non-flat
      # cosmology is assumed if `params_fid` includes the key `Ok`.
      # For other dark energy models one can set `de_model` to `w0` or `w0wa`, in
      # which case one needs to provide the values for w0, wa in `params_fid`.
      params['w0'] = -1.1
      params['wa'] = 0.1

.. container:: cell markdown

   Then let's recompute the model updating such parameters and compare
   with the :math:`\Lambda CDM` prediction

.. container:: cell code

   .. code:: python

      Pell_w0wa = EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model='w0wa')

.. container:: cell code

   .. code:: python

      f = plt.figure(figsize=(10,5))
      ax = f.add_subplot(111)

      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell0"],c='C0',ls='-',label='P0')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell2"],c='C1',ls='-',label='P2')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell4"],c='C2',ls='-',label='P4')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_w0wa["ell0"],c='C0',ls='--')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_w0wa["ell2"],c='C1',ls='--')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_w0wa["ell4"],c='C2',ls='--')
      ax.set_xlabel('$k$ [h/Mpc]',fontsize=15)
      ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{5/2}$]',fontsize=15)
      ax.legend(fontsize=15)
      plt.show()

   .. container:: output display_data

      .. image:: vertopal_a899190433aa4f85bb1541798bd599a6/9ebd8852cc810799a3f8285b02bb172d804750d5.png


The :math:`f-\sigma_{12}` parameter space.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: the-f-sigma_12-parameter-space

.. container:: cell markdown

   The function called before ignores the values of ``s12``,
   ``alpha_tr``, ``alpha_lo`` and ``f`` in the parameter dictionary and
   instead converts the :math:`\Lambda`\ CDM parameters to the
   :math:`\sigma_{12}` parameter space. The internal values of those
   parameters (which can be accessed via ``EFT.params``) have therefore
   been updated:

.. container:: cell code

   .. code:: python

      # s12, alpha_tr, alpha_lo and f are computed internally!
      EFT.params

   .. container:: output execute_result

      ::

         {'wc': 0.11544,
          'wb': 0.0222191,
          'ns': 0.9632,
          's12': array([0.5993593]),
          'f': 0.7252890877752092,
          'b1': 2.0,
          'b2': -0.5,
          'g2': 0.0,
          'g21': 0.0,
          'c0': 0.0,
          'c2': 0.0,
          'c4': 0.0,
          'cnlo': 0.0,
          'N0': 0.0,
          'N20': 0.0,
          'N22': 0.0,
          'h': 0.8,
          'As': 2.3,
          'Ok': 0.0,
          'w0': -1.1,
          'wa': 0.1,
          'z': 0.6,
          'alpha_tr': 1.0960392096062852,
          'alpha_lo': 1.0718295294749038}

.. container:: cell markdown

   First, we need to redefine our parameters.

.. container:: cell code

   .. code:: python

      # For predictions using the RSD parameter space we also need to specify values for the following four parameters, e.g.
      params['s12']      = 0.6
      params['alpha_lo'] = 1.1
      params['alpha_tr'] = 0.9
      params['f']        = 0.7

.. container:: cell markdown

   **Note**: When computing the multipoles using the :math:`\sigma_{12}`
   parameter space we need to specify a fiducial value for the Hubble
   rate. This is required to convert the native emulator output from Mpc
   to Mpc/h units.

.. container:: cell code

   .. code:: python

      Pell_s12 = EFT.Pell(k_hMpc, params, ell=[0,2,4])

.. container:: cell code

   .. code:: python

      f = plt.figure(figsize=(10,5))
      ax = f.add_subplot(111)

      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell0"],c='C0',ls='-',label='P0')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell2"],c='C1',ls='-',label='P2')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_LCDM["ell4"],c='C2',ls='-',label='P4')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_s12["ell0"],c='C0',ls='--')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_s12["ell2"],c='C1',ls='--')
      ax.semilogx(k_hMpc, k_hMpc**0.5*Pell_s12["ell4"],c='C2',ls='--')
      ax.set_xlabel('$k$ [h/Mpc]',fontsize=15)
      ax.set_ylabel(r'$k^{1/2}\,P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{5/2}$]',fontsize=15)
      ax.legend(fontsize=15)
      plt.show()

   .. container:: output display_data

      .. image:: vertopal_a899190433aa4f85bb1541798bd599a6/3facc8bde43a8184ef0a5a04de74d2e6de557b77.png


How to provide different k-scales.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: how-to-provide-different-k-scales

.. container:: cell markdown

   The scales for which to compute the multipoles: if given as a number
   or numpy array all specified multipoles will be computed for those
   scales, if given as a list, the length must match the number of
   specified multipoles (``ell``) and the first entry of the list is
   evaluated for the first multipole etc.

   We can output at a single scale and single multipole number, e.g. for
   the quadrupole at :math:`k = 0.1 1/Mpc`:

.. container:: cell code

   .. code:: python

      # The provided scales are also assumed to be in 1/Mpc units
      EFT.Pell(0.1, params, ell=2)

   .. container:: output execute_result

      ::

         {'ell2': array([12735.02642573])}

.. container:: cell markdown

   Or for various multipoles and multiple scales, in which case the
   output is a list with the first entry corresponding to the first
   multipole specified in ``ell``:

.. container:: cell code

   .. code:: python

      EFT.Pell(np.array([0.1,0.2,0.3]), params, ell=[0,2,4])

   .. container:: output execute_result

      ::

         {'ell0': array([21993.63231466,  8419.32061295,  5052.40597447]),
          'ell2': array([12735.02642573,  7163.41410577,  5357.67022233]),
          'ell4': array([3027.55903402, 2244.05437432, 1870.91419438])}

.. container:: cell markdown

   Or at different scales for different multipoles (providing a list of
   numbers or numpy arrays):

.. container:: cell code

   .. code:: python

      EFT.Pell([np.array([0.1,0.2]),0.3], params, ell=[0,4])

   .. container:: output execute_result

      ::

         {'ell0': array([21993.63231466,  8419.32061295]),
          'ell4': array([1870.91419438])}


Speedup when changing just bias parameters.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: speedup-when-changing-just-bias-parameters

.. container:: cell markdown

   It is a common task to test the models just by changing parameters
   that does not involve any cosmological computation. On that case we
   can call the function ``Pell_fixed_cosmo_boost``, which looks into
   the parameters specified and if any cosmological parameter has
   changed, it uses the computation from previous calls. In the
   following cells the diferences on time can be seen, which reflects a
   speed up of around 3 orders of magnitude.

.. container:: cell code

   .. code:: python

      %timeit EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model="lambda")

   .. container:: output stream stdout

      ::

         22.3 ms ± 676 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

.. container:: cell code

   .. code:: python

      %timeit EFT.Pell_fixed_cosmo_boost(k_hMpc, params, ell=[0,2,4], de_model="lambda")

   .. container:: output stream stdout

      ::

         22.8 µs ± 3.19 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)


Beyond :math:`P_{\ell}` predictions.
====================================
.. container:: cell markdown

   .. rubric:: \
      :name: beyond-p_ell-predictions

.. container:: cell markdown

   We have included some required tools needed for cosmological analysis
   in order to make easiert to integrate it into a complete pipeline


Computing covariance matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: computing-covariance-matrices

.. container:: cell markdown

   Apart from the multipoles we can also generate (Gaussian) covariance
   matrices, for which there are again different flags, that are either
   defined for the :math:`\sigma_{12}` or the :math:`\Lambda`\ CDM
   parameter spaces. The first three arguments, ``k``, ``params``, and
   ``ell``, are identical to those for ``Pell``. In addition, we need to
   specify a binwidth ``dk`` and volume (both of which need to be given
   in the respective units for which the emulator is configured in), for
   example:

.. container:: cell code

   .. code:: python

      dk_hMpc = 0.001
      k_hMpc_lin = np.arange(0.001, 0.3, dk_hMpc)

.. container:: cell code

   .. code:: python

      Cov_hMpc = EFT.Pell_covariance(k_hMpc, params, ell=[0,2,4], dk=dk_hMpc, volume=3e9)

.. container:: cell code

   .. code:: python

      plt.figure(figsize=(9,6))
      plt.title(r"")
      plt.title(r"Corr matrix", y=1.05)
      var_inv = np.diag(1./np.sqrt(np.diag(Cov_hMpc)))
      R_hMpc = var_inv @ Cov_hMpc @ var_inv
      plt.imshow(R_hMpc,cmap='magma_r', interpolation="blackman", filterrad=10)
      plt.show()

   .. container:: output display_data

      .. image:: vertopal_a899190433aa4f85bb1541798bd599a6/b2f4536ec9c2f0f8e7b1e57e5ceb347e60e832d7.png

.. container:: cell markdown

   For the :math:`\Lambda`\ CDM version, instead of providing a volume,
   we can provide minimum and maximum redshifts, ``zmin`` and ``zmax``,
   a sky fraction ``fsky``, and a volume scaling factor ``volfac`` (by
   default equal to 1), which computes then evalutes the corresponding
   volume, also assuming a flat :math:`\Lambda`\ CDM model. For example:

.. container:: cell code

   .. code:: python

      Cov_hMpc_LCDM = EFT.Pell_covariance(
                              k_hMpc,
                              params,
                              ell=[0,2,4],
                              dk=2*np.pi/3780,
                              zmin=params['z']-0.1,
                              zmax=params['z']+0.1,
                              fsky=15000./(360**2/np.pi),
                              volfac=1,
                              de_model="lambda",
                              volume=3780**3
                      )

.. container:: cell code

   .. code:: python

      plt.figure(figsize=(9,6))
      plt.title(r"")
      plt.title(r"Corr matrix", y=1.05)
      var_inv = np.diag(1./np.sqrt(np.diag(Cov_hMpc_LCDM)))
      R_hMpc = var_inv @ Cov_hMpc_LCDM @ var_inv
      plt.imshow(R_hMpc,cmap='magma_r', interpolation="blackman", filterrad=10)
      plt.show()

   .. container:: output display_data

      .. image:: vertopal_a899190433aa4f85bb1541798bd599a6/affd7df3aa9238b86a5216b22deea909da142f8e.png


Computing the :math:`\chi^2`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. container:: cell markdown

   .. rubric:: \
      :name: computing-the-chi2

.. container:: cell markdown

   Finally, the Comet class provides a function to directly compute the
   :math:`\chi^2`. In order to do so, we need to specify the data set,
   i.e., range of scales of the observations, the measured multipoles
   and their covariance matrix. As before, these must be given in the
   same unit system as that for which the emulator has been configured.

.. container:: cell code

   .. code:: python

      # Loading a sample data set:
      data = np.loadtxt('../../../../data/cmass/Minerva_HOD_zs_z0.57_pkmulti_mean.dat')
      cov = np.loadtxt('../../../../data/cmass/Minerva_HOD_zs_z0.57_pkmulti_covar.dat')

.. container:: cell markdown

   As the data vector would have an assosiated shot noise, we can
   specify it by indicating the number density of the sample.

.. container:: cell code

   .. code:: python

      EFT.define_nbar(nbar=3.95898e-4)

.. container:: cell markdown

   We have the option of specifying whether the covariance matrix is a
   "theory" covariance matrix or not. If ``theory_cov = False``, the
   Anderson-Hartlap factor is included in the inverse covariance matrix,
   which is why we need to also provide the number of n_realizations
   from which the covariance matrix was estimated (if
   ``theory_cov = True``, ``Nrealizations`` can be ignored).

.. container:: cell code

   .. code:: python

      EFT.define_data_set(obs_id='Pk', bins=data[:,0], signal=data[:,(1,3,5)], cov=cov, theory_cov=False, n_realizations=300)

.. container:: cell markdown

   Now we can call ``chi2``, which takes as arguments the parameter
   dictionary, a maximum k-mode value ``kmax``, a model argument
   ``de_model``. ``kmax`` can either be a number, in which case the same
   cutoff is applied for all multipoles, or a list of numbers for each
   individual multipole, as for the multipoles case. If the cutoff is
   zero (or smaller than the minimum scale of the observations) for a
   particular multipole, then it is excluded from the computation of the
   chi-square. ``kmax`` is also assumed to be in the units of the
   emulator. ``de_model`` can be one of the specified before.

.. container:: cell code

   .. code:: python

      EFT.chi2(obs_id='Pk',params=params, kmax=[0.3,0.30, 0.30], de_model='lambda', chi2_decomposition=False)

   .. container:: output execute_result

      ::

         9453.159316045188

.. container:: cell markdown

   Moreover, in order to speed up the computation of the :math:`\chi^2`,
   in the same way as ``Pell_fixed_cosmo_boost`` function, we can
   specify the flag ``chi2_decomposition`` in order to avoid recompute
   the cosmological dependent quantities. Let's see how it works

.. container:: cell code

   .. code:: python

      %timeit EFT.chi2(obs_id='Pk',params=params, kmax=[0.3,0.30, 0.30], de_model='lambda', chi2_decomposition=False)

   .. container:: output stream stdout

      ::

         23 ms ± 538 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

.. container:: cell code

   .. code:: python

      %timeit EFT.chi2(obs_id='Pk',params=params, kmax=[0.3,0.30, 0.30], de_model='lambda', chi2_decomposition=True)

   .. container:: output stream stdout

      ::

         20.7 µs ± 470 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

.. container:: cell code

   .. code:: python
