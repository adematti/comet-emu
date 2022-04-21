.. COMET documentation master file, created by
   sphinx-quickstart on Wed Apr 13 23:10:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to COMET's documentation!
=================================

.. warning::
   UNDER CONSTRUCTION |:construction_worker:| |:wrench:| |:nut_and_bolt:|

====================  =====
**Contributors**:     Alex Eggemeier, Benjamin Camacho-Quevedo, Andrea Pezzotta,
                      Martin Crocce, Román Scoccimarro, Ariel Sánchez
**Source**:           `Source code at GitLab <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
**Documentation**:    `Documentation at Readthedocs <https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks>`_
**Installation**:     ``pip install comet``
**References**:       `Sanchez 2020 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511>`_, `Sanchez et al. 2021 <https://arxiv.org/abs/2108.12710>`_,
====================  =====

|:dizzy:| **COMET** - Cosmological Observables Modelled by Emulated perturbation
                      Theory.
     COMET is a Python package that provides emulated predictions of large-scale
     structure observables from models that are fundamentally based on
     perturbation theory. COMET speeds up these analytic computations by two
     orders of magnitude without any relevant sacrifice in accuracy, enabling
     an extremely efficient exploration of large-scale structure likelihoods.

     At its core, COMET exploits the evolution mapping approach of
     `Sanchez 2020 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511>`_
     and `Sanchez et al. 2021 <https://arxiv.org/abs/2108.12710>`_, which
     gives it a high degree of flexibility and allows it to cover a wide
     cosmology parameter space at continuous redshifts up to ``z ~ 3``.
     Specifically, the  current release of COMET supports the following
     parameters (for more details, see here):

     ====                                               ====
     Phys. cold dark matter density:                    ``omega_c``
     Phys. baryon density:                              ``omega_b``
     Scalar spectral index:                             ``n_s``
     Hubble expansion rate:                             ``h``
     Amplitude of scalar fluctuations:                  ``A_s``
     Constant dark energy equation of state parameter:  ``w_0c``
     Time-evolving equation of state parameter:         ``w_a``
     Curvature density parameter:                       ``Omega_K``
     ====                                               ====

     Currently, COMET can be used to obtain the following quantities (for the
     perturbation theory models described in more detail here):

     - the real-space galaxy power spectrum at one-loop order
     - multipoles (monopole, quadrupole, hexadecapole) of the redshift-space
       power spectrum at one-loop order
     - Gaussian covariances in real- and redshift-space
     - ``chi^2``'s for arbitrary combinations of multipoles

     COMET provides an easy-to-use interface for any of these computations, and
     we give quick-start as well as more in-depth examples on our Tutorial
     pages. For installation instructions, see here. 

     Our emulators are publicly available under MIT licence; please, follow the
     links above to be see te corresponding papers on the arXiv website, where
     you can find all the references to credit our work.

.. note::
  The comet emulator is under constant development and new versions of the
  emulator become available as we improve them. Follow our `public repository
  <https://gitlab.com/aegge/pt-emulator>`_ to make sure you are always up to
  date with our latest release.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples
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
