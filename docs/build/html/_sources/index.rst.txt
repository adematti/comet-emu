.. COMET documentation master file, created by
   sphinx-quickstart on Wed Apr 13 23:10:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to COMET's documentation!
=================================

.. .. warning::
..    UNDER CONSTRUCTION |:construction_worker:| |:wrench:| |:nut_and_bolt:|

====================  =====
**Contributors**:     Alex Eggemeier, Benjamin Camacho-Quevedo, Andrea Pezzotta,
                      Martin Crocce, Román Scoccimarro, Ariel G. Sánchez
**Source**:           `Source code at GitLab <https://gitlab.com/aegge/comet-emu>`_
**Documentation**:    `Documentation at Readthedocs <https://comet-emu.readthedocs.io/en/latest/>`_
**Installation**:     ``pip install comet-emu``
**References**:       `Eggemeier et al. 2022 <https://academic.oup.com/mnras/article/519/2/2962/6912276>`_
====================  =====

|:dizzy:| **COMET** - Cosmological Observables Modelled by Emulated perturbation Theory
     COMET is a Python package that provides emulated predictions of large-scale
     structure observables from models that are based on perturbation theory.
     COMET substantially speeds up these analytic computations without any
     relevant sacrifice in accuracy, enabling an extremely efficient
     exploration of large-scale structure likelihoods.

     At its core, COMET exploits the evolution mapping approach of
     `Sanchez 2020 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511>`_
     and `Sanchez et al. 2021 <https://arxiv.org/abs/2108.12710>`_, which
     gives it a high degree of flexibility and allows it to cover a wide
     cosmology parameter space at continuous redshifts up to :math:`z \sim 3`.
     Specifically, the  current release of COMET supports the following
     parameters (for more details, see :ref:`here<spaceparams>`):

     ================================================  ====
     Phys. cold dark matter density                    :math:`\omega_c`
     Phys. baryon density                              :math:`\omega_b`
     Scalar spectral index                             :math:`n_s`
     Hubble expansion rate                             :math:`h`
     Amplitude of scalar fluctuations                  :math:`A_s`
     Constant dark energy equation of state parameter  :math:`w_0`
     Time-evolving equation of state parameter         :math:`w_a`
     Curvature density parameter                       :math:`\Omega_K`
     ================================================  ====

     Currently, COMET can be used to obtain the following quantities (the
     perturbation theory models are described :ref:`here<models>`):

     - the real-space galaxy power spectrum at one-loop order and bispectrum
       at tree-level order
     - multipoles (monopole, quadrupole, hexadecapole) of the redshift-space
       power spectrum at one-loop order and bispectrum at tree-level order for
       two different redshift-space distortion models
     - the linear matter power spectrum (with and without infrared resummation)
     - Gaussian covariance matrices for the real-space power spectrum and
       bispectrum and their redshift-space multipoles
     - :math:`\chi^2`'s for arbitrary combinations of multipoles

     COMET provides an easy-to-use interface for all of these computations, and
     we give quick-start as well as more in-depth examples on our
     :ref:`tutorial<examples>` pages.

     Our package is made publicly available under the MIT licence; please cite
     the papers listed above if you are making use of COMET in your own work.

.. note::
  The COMET emulator is under constant development and new versions of the
  emulator become available as we improve them. Follow our `public repository
  <https://gitlab.com/aegge/comet-emu>`_ to make sure you are always up to
  date with our latest release.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   ./Tutorial/examples
   model
   spaceparams

.. toctree::
   :maxdepth: 2
   :caption: Complete API:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
