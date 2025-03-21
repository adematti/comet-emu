# Give a Welcome to the *COMET*

| | |
| ---      | ---      |
| **Author**:  |  Alex E. et al. |
| **Source:**  |  [Source code at GitLab](https://gitlab.com/aegge/pt-emulator)  |
| **Documentation**: | [Documentation at Readthedocs](https://comet-emu.readthedocs.io/en/latest/index.html)  |
| **Installation**:  |  `pip install comet-emu`|
| **References**:  | [Eggemeier et al 2022](https://academic.oup.com/mnras/article/519/2/2962/6912276), [Pezzotta et al 2025](https://arxiv.org/abs/2503.16160) |

---
## :dizzy: **COMET** - Cosmological Observables Modelled by Emulated perturbation Theory.

COMET is a Python package that provides emulated predictions of large-scale
structure observables from models that are based on perturbation theory.
COMET substantially speeds up these analytic computations without any
relevant sacrifice in accuracy, enabling an extremely efficient
exploration of large-scale structure likelihoods.

At its core, COMET exploits the evolution mapping approach of
[Sanchez 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511)
and [Sanchez et al. 2021](https://arxiv.org/abs/2108.12710), which
gives it a high degree of flexibility and allows it to cover a wide
cosmology parameter space at continuous redshifts up to $z \sim 3$.
Specifically, the  current release of COMET supports the following
parameters (for more details, see [here](https://comet-emulator-comet-emu.readthedocs-hosted.com/en/latest/spaceparams.html)):

| | |
| ---    | ---     |
| Phys. cold dark matter density   |                 $`\omega_c`$ |
| Phys. baryon density  |                            $`\omega_b`$ |
| Scalar spectral index |                            $`n_s`$ |
| Hubble expansion rate  |                           $`h`$ |
| Amplitude of scalar fluctuations  |                $`A_s`$ |
| Constant dark energy equation of state parameter | $`w_0`$ |
| Time-evolving equation of state parameter   |      $`w_a`$ |
| Curvature density parameter   |                    $`\Omega_K`$ |
| Total neutrino mass           |                    $`M_\nu`$ |

Currently, COMET can be used to obtain the following quantities (the
perturbation theory models are described [here](https://comet-emu.readthedocs.io/en/latest/model.html)):

- the real-space galaxy power spectrum at one-loop order
- multipoles (monopole, quadrupole, hexadecapole) of the redshift-space
  power spectrum at one-loop order
- the linear matter power spectrum (with and without infrared resummation)
- Gaussian covariance matrices for the real-space power spectrum and
  redshift-space multipoles
- $`\chi^2`$'s for arbitrary combinations of multipoles

COMET provides an easy-to-use interface for all of these computations, and
we give quick-start as well as more in-depth examples on our
[tutorial pages](https://comet-emu.readthedocs.io/en/latest/Tutorial/examples.html).

Our package is made publicly available under the MIT licence; please cite
the papers listed above if you are making use of COMET in your own work.

## Getting started

Install the code is as easy as

```
pip install comet-emu
```

Then you can follow the [Jupyter Notebook](https://gitlab.com/aegge/comet-emu/-/tree/main/notebooks)
for a small example on how to make predictions, compare with data and estimate
the $`\chi^2`$ of your model.

## Developer version

If you want to modify the code and play around with it, we provide a developer
version so that you can make it and test it. Also, could be possible that you
have your own theoretical predictions and you wish to train the emulator
with your own computations. You can install the developer
version as follow.

```
git clone git@gitlab.com:aegge/comet-emu.git
cd comet-emu
pip install -e .
```

Then you can follow the [Jupyter Notebook](https://gitlab.com/aegge/comet-emu/-/tree/main/notebooks)
to learn how to train the *COMET* and make predictions.


## License
MIT License

## Project status
.. note::
  The COMET emulator is under constant development and new versions of the
  emulator become available as we improve them. Follow our `public repository
  <https://gitlab.com/aegge/comet-emu>`_ to make sure you are always up to
  date with our latest release.
