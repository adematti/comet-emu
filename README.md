# Give a Welcome to the *COMET*

| | |
| ---      | ---      |
| **Author**:  |  Alex E. et al. |
| **Source:**  |  [Source code at GitLab](https://gitlab.com/aegge/pt-emulator)  |
| **Documentation**: | [Documentation at Readthedocs](https://gitlab.com/aegge/pt-emulator)  |
| **Installation**:  |  `pip install comet`|
| **References**:  |[Sanchez 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511), [Sanchez et al 2021](https://arxiv.org/abs/2108.12710) |

---
## :dizzy: **COMET** - Cosmological Observables Modelled with/by/from Emulated Theory.

 The emulator makes use of evolution mapping [Sanchez 2020](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.102.123511),
 [Sanchez et al 2021](https://arxiv.org/abs/2108.12710) to compress
 the information of evolution parameters $`\mathbf{\Theta_{e}}`$
 (e.g. $`h,\,\Omega_\mathrm{K},\,w_0,\,w_\mathrm{a},\,A_\mathrm{s},\,
 \ldots`$) into the single quantity $`\sigma_{12}`$, defined as the rms
 fluctuation of the linear density contrast $`\delta`$ within spheres
 of radius $`R=12\,\mathrm{Mpc}`$.

 This parameter, together with the parameters affecting the shape of the
 power spectrum $`\mathbf{\Theta_{s}}`$ (e.g.
 $`\omega_\mathrm{b},\,\omega_\mathrm{c},\,n_\mathrm{s}`$), and
 the linear growth rate $`f`$, are used as base of the emulator.

 The redshift-dependency of the multipoles can also be treated similarly to
 the impact that different evolution parameters have on the power spectrum,
 that is, by a simple rescaling of the amplitude of the power spectrum in
 order to match the desired value of $`\sigma_{12}`$.

 Internally to the emulator, the pair $`\left[k,P(k)\right]`$ is
 expressed in $`\left[\mathrm{Mpc}^{-1},\mathrm{Mpc}^3\right]`$ units,
 since this is the only set of units for which the evolution parameter
 degeneracy is present. If the user wishes to use the more conventional
 unit set $`\left[h\,\mathrm{Mpc}^{-1},h^{-3}\,\mathrm{Mpc}^3\right]`$,
 they can do so by specifying it in the proper class attribute flag. In this
 case, the input/output are converted into $`\mathrm{Mpc}`$ units
 before being used/returned.

 Geometrical distortions (AP corrections) are included a posteriori without
 the need of including them in the emulation. This process is carried out
 by first reconstructing the full anisotropic 2d galaxy power spectrum
 $`P_\mathrm{gg}(k,\mu)`$, summing up all the even multipoles up to
 $`\ell=6`$, applying distortions to $`k`$ and $`\mu`$, and then
 projecting again over the Legendre polynomials.

## Getting started

Install the code is as easy as

```
pip install comet
```

Then you can follow the [Jupyter Notebook](https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks)
for a small example in how to make predictions, compare with data and estimate
the $`\chi^2`$ of you model.

## Developer version

If you want to modify the code and play around with it, we provide a developer
version so that you can make it and test it. Also, could be possible that you
have your own theoretical predictions and you wish to train the emulator
with your own computations. You can install the developer
version as follow.

```
git clone git@gitlab.com:aegge/pt-emulator.git
cd pt-emulator
pip install -e .
```

Then you can follow the [Jupyter Notebook](https://gitlab.com/aegge/pt-emulator/-/tree/main/notebooks)
to learn how to train the *COMET* and make predictions.


## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
