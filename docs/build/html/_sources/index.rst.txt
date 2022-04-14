.. COMET documentation master file, created by
   sphinx-quickstart on Wed Apr 13 23:10:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to COMET's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

**COMET** - Cosmological Observables Modelled with/by/from Emulated Theory.
     The emulator makes use of evolution mapping (`Sanchez 2020
     <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511>`_,
     `Sanchez et al 2021 <https://arxiv.org/abs/2108.12710>`_,) to compress
     the information of evolution parameters :math:`\mathbf{\Theta_{e}}`
     (e.g. :math:`h,\,\Omega_\mathrm{K},\,w_0,\,w_\mathrm{a},\,A_\mathrm{s},\,
     \ldots`) into the single quantity :math:`\sigma_{12}`, defined as the rms
     fluctuation of the linear density contrast :math:`\delta` within spheres
     of radius :math:`R=8\,\mathrm{Mpc}`.

     This parameter, together with the parameters affecting the shape of the
     power spectrum :math:`\mathbf{\Theta_{s}}` (e.g.
     :math:`\omega_\mathrm{b},\,\omega_\mathrm{c},\,n_\mathrm{s}`), and
     the linear growth rate :math:`f`, are used as base of the emulator.

     The redshift-dependency of the multipoles can also be treated similarly to
     the impact that different evolution parameters have on the power spectrum,
     that is, by a simple rescaling of the amplitude of the power spectrum in
     order to match the desired value of :math:`\sigma_{12}`.

     Internally to the emulator, the pair :math:`\left[k,P(k)\right]` is
     expressed in :math:`\left[\mathrm{Mpc}^{-1},\mathrm{Mpc}^3\right]` units,
     since this is the only set of units for which the evolution parameter
     degeneracy is present. If the user wishes to use the more conventional
     unit set :math:`\left[h\,\mathrm{Mpc}^{-1},h^{-3}\,\mathrm{Mpc}^3\right]`,
     they can do so by specifying it in the proper class attribute flag. In this
     case, the input/output are converted into :math:`\mathrm{Mpc}` units
     before being used/returned.

     Geometrical distortions (AP corrections) are included a posteriori without
     the need of including them in the emulation. This process is carried out
     by first reconstructing the full anisotropic 2d galaxy power spectrum
     :math:`P_\mathrm{gg}(k,\mu)`, summing up all the even multipoles up to
     :math:`\ell=6`, applying distortions to :math:`k` and :math:`\mu`, and then
     projecting again over the Legendre polynomials.

     Our emulators are publicly available under MIT licence; please, follow the
     links above to be see te corresponding papers on the arXiv website, where
     you can find all the references to credit our work.

.. note::
  The comet emulator is under constant development and new versions of the
  emulator become available as we improve them. Follow our `public repository
  <https://gitlab.com/aegge/pt-emulator>`_ to make sure you are always up to
  date with our latest release.


Installation
============

Install the code is as easy as

::

  pip install comet


Then you can follow the `Jupyter Notebook <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
for a small example in how to make predictions, compare with data and estimate
the :math:`\chi^2` of you model.

**Developer version**

If you want to modify the code and play around with it, we provide a developer
version so that you can make it and test it. Also, could be possible that you
have your own theoretical predictions and you wish to train the emulator
with your own computations. You can install the developer
version as follow.

::

  git clone git@gitlab.com:aegge/pt-emulator.git
  cd pt-emulator
  pip install -e .


Then you can follow the `Jupyter Notebook <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
to learn how to train the *COMET* and make predictions.

.. warning::
   the comet emulator only works in a
   Python 3 environment; the data file at its core cannot
   be unpickled by Python 2.x; in case your ``pip``
   command doesn't link to a Python 3 pip executable, please
   modify the line above accordingly (e.g. with ``pip3`` instead of ``pip``)

Quick start example
===================


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
