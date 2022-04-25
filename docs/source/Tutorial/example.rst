.. _examples:

Examples
--------

Quick-start
===========

.. container:: cell markdown

   .. rubric::
      :name: quick-start

   -  Here we will show

      -  How to initialise the emulator.
      -  How to obtain multipoles for the standard LCDM cosmology.

.. container:: cell markdown

   Lets first call the main function ``comet`` as well as the required
   libraries

.. container:: cell code

   .. code:: python

      from comet import comet
      import numpy as np
      import matplotlib.pyplot as plt

.. container:: cell markdown

   Before being able to use the emulator for predictions of the
   multipoles, we need to specify the perturbation theory model that we
   want to use, in this case we are going to use the ``EFT`` model, at
   the same time, we configure the emulator to output results either in
   units of 1/Mpc (``use_Mpc = True``) or h/Mpc (``use_Mpc = False``)
   units, we will use the standar h/Mpc. All quantities that are not
   dimensionless are then assumed to be given in the corresponding
   units. The last requirement is a number density (i.e., the inverse
   Poisson shot noise).

.. container:: cell code

   .. code:: python

      EFT=comet(model="EFT", use_Mpc=False)
      EFT.define_nbar(nbar=3.95898e-4)

.. container:: cell markdown

   We then can provide the argument ``de_model`` in order to obtain
   predictions directly in terms of the corresponding cosmological
   parameters. Currently, ``de_model`` can either be ``lambda``, ``w0``
   or ``w0wa``, in which cases one must include the Hubble rate ``h``,
   the scalar amplitude of fluctuations ``As``, the redshift ``z``, and
   potentially ``w0`` and ``wa`` in the parameter dictionary.
   Optionally, it is also possible to specify the curvature density
   parameter at present time, ``Ok``, in order to obtain predictions for
   non-flat cosmologies.

   A corresponding model is used to make the parameter conversions and
   since the computation of the Alcock-Paczynski parameters requires a
   fiducial cosmology we first need to specify the corresponding
   parameter values as follows:

.. container:: cell code

   .. code:: python

      params_fid_Minerva = {'h':0.695, 'wc':0.11544, 'wb':0.0222191, 'z':0.57}

      # This assumes by default a "lambda" cosmology with w0 = -1, a non-flat cosmology is assumed if `params_fid` includes the key `Ok`.
      # For other dark energy models one can set `de_model` to `w0` or `w0wa`, in which case one needs to provide the values for w0, wa in `params_fid`.
      EFT.define_fiducial_cosmology(params_fid=params_fid_Minerva, de_model='lambda')

.. container:: cell markdown

   The functions returning the multipoles take generally two arguments:

   #. The scales for which to compute the multipoles: if given as a
      number or numpy array all specified multipoles will be computed
      for those scales, if given as a list, the length must match the
      number of specified multipoles (``ell``) and the first entry of
      the list is evaluated for the first multipole etc.
   #. The multipole number, i.e. ell = 0, 2, 4, or a list of multipole
      numbers

.. container:: cell code

   .. code:: python

      # Let's create a parameter dictionary
      params = {}

      # We always need to specify the shape parameter values, e.g.
      params['wc'] = 0.11544
      params['wb'] = 0.0222191
      params['ns'] = 0.9632

.. container:: cell markdown

   Next, we specify the three additional :math:`\Lambda`\ CDM parameters
   in the dictionary (keeping the three shape parameters
   :math:`\omega_c`, :math:`\omega_b` and :math:`n_s` fixed from
   before):

.. container:: cell code

   .. code:: python

      params['h']  = 0.8
      params['As'] = 2.3
      params['z']  = 0.6

.. container:: cell code

   .. code:: python

      # Finally, the bias parameters: any parameters from {b1, b2, g2, g21, c0, c2, c4, cnlo, N0, N20, N22} can be specified.
      # Parameters, which are not explicitly specified are automatically set to zero. As an example, let's just set b1 and b2:
      params['b1'] = 2.
      params['b2'] = -0.5

.. container:: cell markdown

   Now, let's define a range of scales

.. container:: cell code

   .. code:: python

      k_hMpc = np.logspace(-3,np.log10(0.3),100)

.. container:: cell markdown

   Let's generate multipoles for this parameter set for the range of
   scales above:

.. container:: cell code

   .. code:: python

      Pell_LCDM = EFT.Pell(k_hMpc, params, ell=[0,2,4], de_model='lambda') # E.g., this is for a flat LCDM cosmology

.. container:: cell markdown

   The output is given in a dictionary format.

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

      .. image:: vertopal_63f8ea5046a542bd93141360260eb85c/537407037d704ddedc6cd575f28290c731218f9f.png

.. container:: cell markdown

In-depth options for obtaining multipoles
=========================================

   .. rubric::
      :name: in-depth-options-for-obtaining-multipoles

.. container:: cell markdown

   -  Now, let's see somo details:

      -  different cosmologies (w0 + w0wa)
      -  using the f-s12 parameter space
      -  the options for providing different k-scales, float vs np.array
         vs list and the corresponding outputs
      -  describe the 'fixed_cosmo_boost' function, i.e., speedup when
         just changing bias parameters

.. container:: cell markdown

An alternative Dark energy model.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

      :name: an-alternative-dark-energy-model

.. container:: cell markdown

   As mentioned before, we can make use of a different cosmological
   model. Let's try the ``w0wa`` model. We need to add such parameters
   to our params dictionary first.

.. container:: cell code

   .. code:: python

      params['w0'] = -1.1
      params['wa'] = 0.1

.. container:: cell markdown

   Then lets recompute the model updating such parameters and compare
   with the LCDM prediction

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

      .. image:: vertopal_63f8ea5046a542bd93141360260eb85c/1f2260c5530af2f777489539a44c2194a8f180d5.png


The :math:`f-\sigma_{12}` parameter space.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: cell markdown

   .. rubric::
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
          'f': array([0.72528909]),
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

   *Note: When computing the multipoles using the :math:`\sigma_{12}`
   parameter space we need to specify a fiducial value for the Hubble
   rate. This is required to convert the native emulator output from Mpc
   to Mpc/h units.*

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

      .. image:: vertopal_63f8ea5046a542bd93141360260eb85c/3facc8bde43a8184ef0a5a04de74d2e6de557b77.png


How to provide different k-scales.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. container:: cell markdown

   .. rubric::
      :name: how-to-provide-different-k-scales

.. container:: cell markdown

   We can output at a single scale and single multipole number, e.g. for
   the quadrupole at k = 0.1 1/Mpc:

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

   .. rubric:: Speedup when changing just bias parameters.
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

         23.8 ms ± 2.07 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

.. container:: cell code

   .. code:: python

      %timeit EFT.Pell_fixed_cosmo_boost(k_hMpc, params, ell=[0,2,4], de_model="lambda")

   .. container:: output stream stdout

      ::

         23.9 µs ± 1.02 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

.. container:: cell code

   .. code:: python
