.. _spaceparams:

Parameter space
---------------

On this page we describe the native emulator parameter space with the
supported range of values, as well as the various cosmological and bias
parameters that can be specified.

Ranges of emulated parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Since COMET employs the evolution mapping approach, its parameter space consists
of a set of shape parameters, in addition to :math:`\sigma_{12}` and the
growth rate :math:`f`, which capture the dependence on redshift and evolution
parameters. The following table lists the ranges for each emulated parameter.

+--------------------+-------------+-------------+
| Parameter          | Minimum     | Maximum     |
+====================+=============+=============+
| :math:`\omega_c`   | 0.085       | 0.155       |
+--------------------+-------------+-------------+
| :math:`\omega_b`   | 0.0205      | 0.02415     |
+--------------------+-------------+-------------+
| :math:`n_s`        | 0.92        | 1.01        |
+--------------------+-------------+-------------+
| :math:`\sigma_{12}`| 0.2         | 1.0         |
+--------------------+-------------+-------------+
| :math:`f`          | 0.5         | 1.05        |
+--------------------+-------------+-------------+

All predictions from COMET are limited to the range of scales
:math:`k \in [6.95 \times 10^{-4}, 0.35028]\,\mathrm{Mpc}^{-1}` (note that the range is
defined in units of :math:`\mathrm{Mpc}`, opposed to :math:`h^{-1}\,\mathrm{Mpc}`!).

.. note::
   Note that for values of :math:`k` beyond 0.35 :math:`\mathrm{Mpc}^{-1}`
   COMET returns a power-law extrapolation of the power spectrum multipoles.
   This is mainly intended for performing the convolution with the survey window
   function. Direct evaluation of the multipoles should always be kept within
   the supported range above.

Cosmological parameters
~~~~~~~~~~~~~~~~~~~~~~~

The table below lists all available cosmological parameters and the
corresponding keys by which they are identified in the parameter dictionary
that is given as an argument to the function returning the power spectrum
multipoles (see :ref:`examples`).

.. note::
   The three shape parameters :math:`(\omega_c,\, \omega_b,\, n_s)` always need
   to be specified in the dictionary, while the remaining required parameters
   are determined by the option ``de_model``. If ``de_model = None`` (default),
   the native emulator parameter space is assumed and we need to provide values
   for :math:`\sigma_{12}` and :math:`f`. For ``de_model = 'lambda'``, ``'w0'``, or
   ``'w0wa'`` different sets of parameters are required. The curvature density
   is always optional and if not explicitly included in the parameter dictionary
   a flat cosmology is assumed.

+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| Parameter           | Description                                  | Key            | Option                                        |
+=====================+==============================================+================+===============================================+
| :math:`\omega_c`    | Phys. cold dark matter density               | ``wc``         |                                               |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`\omega_b`    | Phys. baryon density                         | ``wb``         |                                               |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`n_s`         | Scalar spectral index                        | ``ns``         |                                               |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`\sigma_{12}` | RMS of fluctuations in spheres of 12 Mpc     | ``s12``        | ``de_model = None``                           |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`f`           | Growth rate                                  | ``f``          | ``de_model = None``                           |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`A_s`         | Amplitude of scalar fluctuations             | ``As``         | ``de_model = 'lambda'``, ``'w0'``, ``'w0wa'`` |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`h`           | Hubble rate                                  | ``h``          | ``de_model = 'lambda'``, ``'w0'``, ``'w0wa'`` |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`w_0`         | Const. DE equation of state parameter        | ``w0``         | ``de_model = 'w0'``, ``'w0wa'``               |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`w_a`         | Time evolving DE equation of state parameter | ``wa``         | ``de_model = 'w0wa'``                         |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+
| :math:`\Omega_K`    | Curvature density                            | ``Ok``         | ``de_model = 'lambda'``, ``'w0'``, ``'w0wa'`` |
+---------------------+----------------------------------------------+----------------+-----------------------------------------------+

Bias and RSD parameters
~~~~~~~~~~~~~~~~~~~~~~~

Finally we list all bias and RSD parameters that can be specified in the models that COMET implements. All of them are optional and assumed to be zero if they are not explicitly included in the parameter dictionary.

+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| Parameter                  | Description                                                                                                             | Key      |
+============================+=========================================================================================================================+==========+
| :math:`b_1`                | Linear bias                                                                                                             | ``b1``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`b_2`                | Quadratic non-linear bias                                                                                               | ``b2``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`\gamma_2`           | Second-order tidal bias                                                                                                 | ``g2``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`\gamma_{21}`        | Third-order tidal bias                                                                                                  | ``g21``  |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c_0`                | Counterterm parameter, in units of :math:`L^2`                                                                          | ``c0``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c_2`                | Counterterm parameter, prop. to :math:`{\cal L}_2(\mu)`, in units of :math:`L^2`                                        | ``c2``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c_4`                | Counterterm parameter, prop. to :math:`{\cal L}_4(\mu)`, in units of :math:`L^2`                                        | ``c4``   |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c_{\mathrm{nlo}}`   | Next-to-leading order counterterm parameter, in units of :math:`L^4`                                                    | ``cnlo`` |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c^B_{\mathrm{nlo}}` | | Next-to-leading order counterterm parameter for the bispectrum                                                        | ``cnloB``|
|                            | | (defined in Eggemeier et al. 2025), in units of :math:`L^2`                                                           |          |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c^B_1`              | | Next-to-leading order counterterm parameter for the bispectrum                                                        | ``cB1``  |
|                            | | (defined in `Ivanov et al. 2022 <https://doi.org/10.1103/PhysRevD.105.063512>`_), in units of :math:`L^2`             |          |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`c^B_2`              | | Next-to-leading order counterterm parameter for the bispectrum                                                        | ``cB2``  |
|                            | | (defined in `Ivanov et al. 2022 <https://doi.org/10.1103/PhysRevD.105.063512>`_), in units of :math:`L^2`             |          |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`a_{\mathrm{vir}}`   | | Parameter controlling the kurtosis in the :math:`\mathrm{VDG}_{\infty}`                                               | ``avir`` |
|                            | | damping function of the power spectrum, in units of :math:`L^2`                                                       |          |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`a_{\mathrm{vir}}^B` | | Parameter controlling the kurtosis in the :math:`\mathrm{VDG}_{\infty}`                                               | ``avirB``|
|                            | | damping function of the bispectrum, in units of :math:`L^2`                                                           |          |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`N^P_0`              | Constant shot noise, in units of :math:`L^3`                                                                            | ``NP0``  |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`N^P_{20}`           | Scale-dependent shot noise, in units of :math:`L^5`                                                                     | ``NP20`` |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`N^P_{22}`           | Scale-dependent shot noise, prop. to :math:`{\cal L}_2(\mu)`, in units of :math:`L^5`                                   | ``NP22`` |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`N^B_{0}`            | Constant bispectrum shot noise, in units of :math:`L^6`                                                                 | ``NB0``  |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+
| :math:`M^B_{0}`            | Bispectrum shot noise (proportional to power spectrum), in units of :math:`L^3`                                         | ``MB0``  |
+----------------------------+-------------------------------------------------------------------------------------------------------------------------+----------+

where :math:`L` either stands for :math:`\mathrm{Mpc}` or :math:`h^{-1}\,\mathrm{Mpc}`, depending on the unit configuration of COMET, see :ref:`examples`.
